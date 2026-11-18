"""Обучение и инференс собственного классификатора звуков.

Модель — многоклассовая логистическая регрессия над MFCC/спектральными
признаками. Она не требует GPU и хорошо подходит для пользовательских
датасетов: «речь», «крик», «птица», «дверь», «огонь», «шум» и т.д.

Структура датасета::

    data/sound_dataset/
        speech/        1.wav 2.wav ...
        yelling/       ...
        bird_singing/  ...
        door_opening/  ...
        fire_burning/  ...

Каждый подкаталог — отдельный класс. Файлы WAV/FLAC/OGG читаются через
``soundfile``; MP3/M4A и видео — через PyAV (если установлен).
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from src.sound_features import (
    FEATURE_CONFIG, FEATURE_VERSION, extract_features, feature_size,
)


AUDIO_SUFFIXES = (
    '.wav', '.flac', '.ogg', '.oga', '.opus', '.aiff', '.aif', '.au',
    '.mp3', '.m4a', '.aac', '.wma',
)


class DatasetError(RuntimeError):
    pass


class ModelError(RuntimeError):
    pass


def load_audio(path):
    """Читает аудиофайл в моно float32. Возвращает ``(audio, rate)``."""
    path = str(path)
    try:
        import soundfile as sf
        data, rate = sf.read(path, dtype='float32', always_2d=True)
        if data.size:
            audio = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
            return np.ascontiguousarray(audio, dtype=np.float32), int(rate)
    except Exception:
        pass
    try:
        from src.file_analyzer import decode_audio_file
        audio, rate, _channels, _fmt = decode_audio_file(path)
        return np.ascontiguousarray(audio, dtype=np.float32), int(rate)
    except Exception as exc:
        raise DatasetError(f'не удалось прочитать {path}: {exc}') from exc


def scan_dataset(dataset_dir):
    """Возвращает ``{class_name: [paths]}`` и предупреждения."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise DatasetError(f'Каталог датасета не найден: {dataset_dir}')
    classes = {}
    warnings = []
    for entry in sorted(dataset_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith(('.', '_')):
            continue
        files = []
        for path in sorted(entry.rglob('*')):
            if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES:
                files.append(path)
        if not files:
            warnings.append(f'класс «{entry.name}» пропущен: нет аудиофайлов')
            continue
        classes[entry.name] = files
    if len(classes) < 2:
        raise DatasetError(
            'Нужно минимум два класса — подкаталога с аудиофайлами. '
            f'Найдено: {len(classes)}')
    for name, files in classes.items():
        if len(files) < 2:
            warnings.append(
                f'класс «{name}» содержит всего {len(files)} файл(а) — '
                'качество обучения будет низким')
    return classes, warnings


def _augment_audio(audio, sample_rate, rng):
    """Простое усиление данных: сдвиг, громкость, шум, обрезка."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return audio
    max_samples = int(10.0 * sample_rate)
    if audio.size > max_samples:
        start = int(rng.integers(0, max(1, audio.size - max_samples)))
        audio = audio[start:start + max_samples]
    elif audio.size < max_samples and rng.random() < 0.5:
        pad = max_samples - audio.size
        left = int(rng.integers(0, pad + 1))
        audio = np.pad(audio, (left, pad - left))
    if rng.random() < 0.8:
        audio = audio * float(rng.uniform(0.5, 1.5))
    if rng.random() < 0.5:
        noise = rng.normal(0.0, float(rng.uniform(0.0005, 0.01)), audio.size)
        audio = audio + noise.astype(np.float32)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def _softmax(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def _accuracy(weights, bias, features, labels, classes):
    logits = features @ weights + bias
    predicted = np.argmax(logits, axis=1)
    return float(np.mean(predicted == labels)), predicted


def _metrics(labels, predicted, classes):
    labels = np.asarray(labels)
    predicted = np.asarray(predicted)
    per_class = {}
    for index, name in enumerate(classes):
        true_positive = int(np.sum((labels == index) & (predicted == index)))
        false_positive = int(np.sum((labels != index) & (predicted == index)))
        false_negative = int(np.sum((labels == index) & (predicted != index)))
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive else 0.0)
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative else 0.0)
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        per_class[name] = {
            'precision': round(float(precision), 4),
            'recall': round(float(recall), 4),
            'f1': round(float(f1), 4),
            'support': int(np.sum(labels == index)),
        }
    confusion = np.zeros((len(classes), len(classes)), dtype=int)
    for truth, guess in zip(labels, predicted):
        confusion[int(truth), int(guess)] += 1
    return per_class, confusion.tolist()


def _stratified_split(items, test_size, rng):
    """Делит ``[(path, class_index)]`` на train/val с сохранением классов."""
    by_class = {}
    for item in items:
        by_class.setdefault(item[1], []).append(item)
    train, validation = [], []
    for label, entries in by_class.items():
        entries = list(entries)
        rng.shuffle(entries)
        if len(entries) <= 2:
            validation.extend(entries[:1])
            train.extend(entries[1:])
            continue
        count = max(1, int(round(len(entries) * test_size)))
        validation.extend(entries[:count])
        train.extend(entries[count:])
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation


def train_model(dataset_dir, output_path=None, *, epochs=250,
                learning_rate=0.02, l2=1e-4, test_size=0.2, augment=True,
                seed=42, progress=None, max_files_per_class=None,
                config=None):
    """Обучает модель и сохраняет её в ``.npz``.

    Возвращает отчёт с точностью, метриками по классам и путём модели.
    """
    def report(fraction, message):
        if progress is not None:
            try:
                progress(max(0.0, min(1.0, float(fraction))), message)
            except Exception:
                pass

    started = time.monotonic()
    rng = np.random.default_rng(int(seed))
    classes, warnings = scan_dataset(dataset_dir)
    names = sorted(classes)
    class_index = {name: index for index, name in enumerate(names)}

    report(0.02, 'Чтение датасета...')
    raw_items = []
    for name in names:
        paths = classes[name]
        if max_files_per_class:
            paths = paths[:int(max_files_per_class)]
        raw_items.extend((path, class_index[name]) for path in paths)

    if len(raw_items) < 2:
        raise DatasetError('Слишком мало аудиофайлов для обучения')

    train_items, val_items = _stratified_split(raw_items, test_size, rng)
    if not val_items:
        val_items = list(train_items[:max(1, len(train_items) // 5)])

    feature_config = dict(FEATURE_CONFIG)
    if config:
        feature_config.update(config)

    def features_for(item, train):
        path, label = item
        audio, rate = load_audio(path)
        vectors = [extract_features(audio, rate, feature_config)]
        if train and augment and audio.size:
            copies = 2 if len(train_items) < 200 else 1
            for _ in range(copies):
                augmented = _augment_audio(audio, int(rate), rng)
                vectors.append(extract_features(augmented, rate, feature_config))
        return np.stack(vectors).astype(np.float32), label

    report(0.08, 'Извлечение признаков (обучающая выборка)...')
    train_x, train_y = [], []
    for index, item in enumerate(train_items):
        vectors, label = features_for(item, True)
        train_x.append(vectors)
        train_y.extend([label] * vectors.shape[0])
        if index % max(1, len(train_items) // 20) == 0:
            report(0.08 + 0.40 * (index / max(1, len(train_items))),
                   f'Признаки: {index + 1}/{len(train_items)}')

    report(0.50, 'Извлечение признаков (проверочная выборка)...')
    val_x, val_y = [], []
    for index, item in enumerate(val_items):
        vectors, label = features_for(item, False)
        val_x.append(vectors)
        val_y.extend([label] * vectors.shape[0])
        if index % max(1, len(val_items) // 10) == 0:
            report(0.50 + 0.15 * (index / max(1, len(val_items))),
                   f'Проверка: {index + 1}/{len(val_items)}')

    x_train = np.concatenate(train_x, axis=0).astype(np.float32)
    y_train = np.asarray(train_y, dtype=np.int64)
    x_val = np.concatenate(val_x, axis=0).astype(np.float32)
    y_val = np.asarray(val_y, dtype=np.int64)

    mean = x_train.mean(axis=0).astype(np.float32)
    std = x_train.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    n_features = x_train.shape[1]
    n_classes = len(names)
    weights = np.zeros((n_features, n_classes), dtype=np.float32)
    bias = np.zeros(n_classes, dtype=np.float32)

    # Веса классов компенсируют дисбаланс (например, «речь» встречается
    # гораздо чаще «пожара»).
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    class_weights = np.where(
        counts > 0, len(y_train) / (n_classes * np.maximum(counts, 1.0)), 0.0)
    sample_weights = class_weights[y_train]

    one_hot = np.zeros((len(y_train), n_classes), dtype=np.float32)
    one_hot[np.arange(len(y_train)), y_train] = 1.0

    # Adam: устойчивее обычного SGD на маленьких датасетах.
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    m_w = np.zeros_like(weights)
    v_w = np.zeros_like(weights)
    m_b = np.zeros_like(bias)
    v_b = np.zeros_like(bias)
    best_accuracy = -1.0
    best_weights = weights.copy()
    best_bias = bias.copy()
    best_epoch = 0
    history = []

    report(0.68, 'Обучение классификатора...')
    for epoch in range(1, int(epochs) + 1):
        logits = x_train @ weights + bias
        probabilities = _softmax(logits)
        weighted = -(sample_weights[:, None] * one_hot *
                     np.log(np.maximum(probabilities, 1e-12)))
        loss = float(weighted.mean())
        error = (probabilities - one_hot) * sample_weights[:, None]
        grad_w = (x_train.T @ error) / len(y_train) + l2 * weights
        grad_b = error.mean(axis=0)

        m_w = beta1 * m_w + (1 - beta1) * grad_w
        v_w = beta2 * v_w + (1 - beta2) * (grad_w ** 2)
        m_b = beta1 * m_b + (1 - beta1) * grad_b
        v_b = beta2 * v_b + (1 - beta2) * (grad_b ** 2)
        m_w_hat = m_w / (1 - beta1 ** epoch)
        v_w_hat = v_w / (1 - beta2 ** epoch)
        m_b_hat = m_b / (1 - beta1 ** epoch)
        v_b_hat = v_b / (1 - beta2 ** epoch)
        weights -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
        bias -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

        train_accuracy, _ = _accuracy(weights, bias, x_train, y_train, names)
        val_accuracy, _ = _accuracy(weights, bias, x_val, y_val, names)
        history.append({
            'epoch': epoch,
            'loss': round(loss, 6),
            'train_accuracy': round(train_accuracy, 4),
            'val_accuracy': round(val_accuracy, 4),
        })
        if x_val.shape[0] and val_accuracy >= best_accuracy:
            best_accuracy = val_accuracy
            best_weights = weights.copy()
            best_bias = bias.copy()
            best_epoch = epoch

        if epoch == 1 or epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            report(0.68 + 0.28 * (epoch / max(1, epochs)),
                   f'Эпоха {epoch}/{epochs}: loss={loss:.4f}, '
                   f'val={val_accuracy * 100:.1f}%')

    weights, bias = best_weights, best_bias
    val_accuracy, predicted = _accuracy(weights, bias, x_val, y_val, names)
    per_class, confusion = _metrics(y_val, predicted, names)
    train_accuracy, _ = _accuracy(weights, bias, x_train, y_train, names)

    model_path = Path(output_path or (
        Path(__file__).resolve().parent.parent / 'models' / 'sound_model.npz'))
    model_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        'version': FEATURE_VERSION,
        'created_at': datetime.now().astimezone().isoformat(timespec='seconds'),
        'dataset': str(Path(dataset_dir).resolve()),
        'classes': names,
        'feature_config': feature_config,
        'epochs': int(epochs),
        'best_epoch': int(best_epoch),
        'learning_rate': float(learning_rate),
        'l2': float(l2),
        'train_samples': int(x_train.shape[0]),
        'val_samples': int(x_val.shape[0]),
        'train_accuracy': round(float(train_accuracy), 4),
        'val_accuracy': round(float(val_accuracy), 4),
        'per_class': per_class,
        'confusion_matrix': confusion,
        'warnings': warnings,
        'duration_seconds': round(time.monotonic() - started, 2),
    }
    np.savez_compressed(
        model_path,
        weights=weights.astype(np.float32),
        bias=bias.astype(np.float32),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        classes=np.asarray(names, dtype=np.str_),
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )

    report(1.0, f'Модель сохранена: {model_path}')
    return {
        'model_path': str(model_path),
        'classes': names,
        'val_accuracy': round(float(val_accuracy), 4),
        'train_accuracy': round(float(train_accuracy), 4),
        'per_class': per_class,
        'confusion_matrix': confusion,
        'train_samples': int(x_train.shape[0]),
        'val_samples': int(x_val.shape[0]),
        'best_epoch': int(best_epoch),
        'warnings': warnings,
        'metadata': metadata,
    }


class CustomSoundModel:
    """Загруженная пользовательская модель звуков."""

    def __init__(self, path=None):
        self.path = str(path) if path else None
        self.loaded = False
        self.classes = []
        self.weights = None
        self.bias = None
        self.mean = None
        self.std = None
        self.feature_config = dict(FEATURE_CONFIG)
        self.metadata = {}
        self.error = None
        if path:
            self.load(path)

    def load(self, path=None):
        path = path or self.path
        if not path:
            raise ModelError('путь к модели не указан')
        path = str(path)
        if not os.path.isfile(path):
            raise ModelError(f'файл модели не найден: {path}')
        try:
            with np.load(path, allow_pickle=False) as data:
                self.weights = np.asarray(data['weights'], dtype=np.float32)
                self.bias = np.asarray(data['bias'], dtype=np.float32)
                self.mean = np.asarray(data['mean'], dtype=np.float32)
                self.std = np.asarray(data['std'], dtype=np.float32)
                self.classes = [str(item) for item in data['classes'].tolist()]
                raw = data['metadata']
                try:
                    self.metadata = json.loads(str(raw.item()))
                except Exception:
                    self.metadata = {}
        except Exception as exc:
            self.error = str(exc)
            raise ModelError(f'не удалось загрузить модель: {exc}') from exc
        self.feature_config = dict(FEATURE_CONFIG)
        self.feature_config.update(
            self.metadata.get('feature_config') or {})
        self.path = path
        self.loaded = True
        self.error = None
        return self

    def predict_proba(self, audio, sample_rate):
        if not self.loaded:
            raise ModelError('модель не загружена')
        vector = extract_features(audio, sample_rate, self.feature_config)
        if vector.size != self.weights.shape[0]:
            raise ModelError(
                f'размер признаков {vector.size} не совпадает с моделью '
                f'{self.weights.shape[0]}')
        normalized = (vector - self.mean) / np.where(
            np.abs(self.std) < 1e-6, 1.0, self.std)
        logits = normalized @ self.weights + self.bias
        probabilities = _softmax(logits.reshape(1, -1))[0]
        order = np.argsort(probabilities)[::-1]
        return [
            {'label': self.classes[index],
             'score': float(probabilities[index])}
            for index in order
        ]

    def info(self):
        return {
            'path': self.path,
            'loaded': self.loaded,
            'classes': list(self.classes),
            'val_accuracy': self.metadata.get('val_accuracy'),
            'created_at': self.metadata.get('created_at'),
            'error': self.error,
        }


def default_model_path():
    custom = os.environ.get('BLUETOOTH_MONITOR_SOUND_MODEL')
    if custom:
        return str(Path(custom).expanduser())
    return str(Path(__file__).resolve().parent.parent / 'models' / 'sound_model.npz')


def load_custom_model(path=None):
    """Загружает модель, если файл существует; иначе возвращает None."""
    path = path or default_model_path()
    if not os.path.isfile(path):
        return None
    try:
        return CustomSoundModel(path)
    except Exception:
        return None
