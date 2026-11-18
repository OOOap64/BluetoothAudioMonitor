"""Классификация источника звука.

Используются два взаимодополняющих источника:

* пользовательская модель (``models/sound_model.npz``), обученная на своих
  классах — речь, крик, птицы, дверь, огонь и т.п.;
* модель AudioSet (AST) из transformers — если она доступна.

Пользовательская модель имеет приоритет и работает без интернета. Если её
нет, приложение продолжает работать с AudioSet. Названия классов AudioSet
дополнительно сворачиваются в понятные категории.
"""

from __future__ import annotations

import threading

import numpy as np

from src.audio_utils import resample
from src.custom_sound_model import CustomSoundModel, default_model_path


_MODEL_CACHE = {}
_CACHE_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()


# Ключевые слова AudioSet -> понятная категория. Проверяются по порядку.
_CATEGORY_RULES = (
    ('yelling', (
        'scream', 'yell', 'shout', 'bellow', 'war cry', 'battle cry',
        'crying', 'sobbing', 'wail', 'groan', 'grunt',
    )),
    ('speech', (
        'speech', 'conversation', 'narration', 'monologue', 'dialogue',
        'whisper', 'audiobook', 'male singing', 'female singing',
    )),
    ('bird_singing', (
        'bird', 'chirp', 'tweet', 'crow', 'owl', 'duck', 'goose', 'rooster',
        'canary', 'parrot', 'pigeon', 'seagull',
    )),
    ('door', (
        'door', 'knock', 'slam', 'creak', 'drawer', 'cabinet', 'window',
        'gate',
    )),
    ('fire_burning', (
        'fire', 'burning', 'crackling', 'crackle', 'explosion', 'firework',
        'bonfire', 'smoke detector',
    )),
    ('alarm', (
        'alarm', 'buzzer', 'bell', 'ringtone', 'siren', 'beep', 'smoke',
        'detector', 'timer',
    )),
    ('music', (
        'music', 'singing', 'guitar', 'piano', 'drum', 'violin', 'synthesizer',
        'orchestra', 'choir', 'band', 'electronic', 'hip hop', 'rock',
        'pop music', 'classical',
    )),
    ('animal', (
        'dog', 'cat', 'animal', 'horse', 'cow', 'sheep', 'frog', 'insect',
        'cricket', 'bee', 'lion', 'tiger', 'elephant', 'monkey',
    )),
    ('water', (
        'water', 'rain', 'stream', 'river', 'ocean', 'waves', 'drip',
        'splash', 'pour',
    )),
    ('traffic', (
        'vehicle', 'car', 'truck', 'motorcycle', 'bus', 'train', 'airplane',
        'engine', 'traffic', 'road', 'horn', 'subway',
    )),
    ('applause', (
        'applause', 'clapping', 'cheering', 'crowd',
    )),
    ('noise', (
        'noise', 'static', 'hum', 'hiss', 'buzz', 'rumble', 'wind',
        'white noise', 'pink noise',
    )),
)


def category_for_label(label):
    """Сворачивает произвольное имя класса в понятную категорию."""
    text = str(label or '').strip().lower().replace('_', ' ').replace('-', ' ')
    if not text:
        return 'other'
    for category, keywords in _CATEGORY_RULES:
        for keyword in keywords:
            if keyword in text:
                return category
    return 'other'


class SoundClassifier:
    def __init__(self, model_name=None, device=None, custom_model_path=None,
                 use_audioset=True):
        self.model_name = model_name or 'MIT/ast-finetuned-audioset-10-10-0.4593'
        self.device = device
        self.feature_extractor = None
        self.model = None
        self.loaded = False
        self.use_audioset = bool(use_audioset)
        self.custom_model_path = custom_model_path or default_model_path()
        self.custom_model = None
        self.custom_error = None
        self.ast_error = None
        self._custom_tried = False
        self._ast_tried = False
        self._load_lock = threading.Lock()
        self._loading = False

    def _resolve_device(self):
        if self.device:
            return self.device
        try:
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except Exception:
            self.device = 'cpu'
        return self.device

    def _load_custom(self):
        if self._custom_tried:
            return self.custom_model is not None
        self._custom_tried = True
        if not self.custom_model_path:
            return False
        try:
            self.custom_model = CustomSoundModel(self.custom_model_path)
            self.custom_error = None
            return True
        except Exception as exc:
            self.custom_error = str(exc)
            self.custom_model = None
            return False

    def _load_ast(self):
        if not self.use_audioset:
            return False
        if self._ast_tried:
            return self.model is not None
        self._ast_tried = True
        device = self._resolve_device()
        key = (self.model_name, device)
        try:
            with _CACHE_LOCK:
                cached = _MODEL_CACHE.get(key)
                if cached is None:
                    from transformers import (
                        AutoFeatureExtractor,
                        AutoModelForAudioClassification,
                    )
                    feature_extractor = AutoFeatureExtractor.from_pretrained(
                        self.model_name)
                    model = AutoModelForAudioClassification.from_pretrained(
                        self.model_name)
                    model.to(device)
                    model.eval()
                    cached = (feature_extractor, model)
                    _MODEL_CACHE[key] = cached
                self.feature_extractor, self.model = cached
            self.ast_error = None
            return True
        except Exception as exc:
            self.ast_error = str(exc)
            self.feature_extractor = None
            self.model = None
            return False

    def load(self):
        """Загружает доступные модели. True — если есть хотя бы одна."""
        with self._load_lock:
            if self._loading:
                return self.loaded
            self._loading = True
        try:
            custom_ok = self._load_custom()
            if custom_ok:
                self.loaded = True
            ast_ok = self._load_ast()
            self.loaded = bool(custom_ok or ast_ok)
            if not self.loaded:
                details = []
                if self.custom_error:
                    details.append('своя модель: ' + self.custom_error)
                if self.ast_error:
                    details.append('AudioSet: ' + self.ast_error)
                raise RuntimeError(
                    'Классификатор звука недоступен. Положите модель в '
                    'models/sound_model.npz (обучение: '
                    'tools/train_sound_model.py) или установите transformers '
                    'и скачайте AST. ' + ' | '.join(details))
            return True
        finally:
            with self._load_lock:
                self._loading = False

    def reload_custom(self, path=None):
        self.custom_model_path = path or self.custom_model_path
        self.custom_model = None
        self._custom_tried = False
        return self._load_custom()

    def status(self):
        return {
            'custom': (
                self.custom_model.info() if self.custom_model is not None
                else {'loaded': False, 'path': self.custom_model_path,
                      'error': self.custom_error}),
            'audioset': {
                'loaded': self.model is not None,
                'model': self.model_name,
                'error': self.ast_error,
            },
            'loaded': self.loaded,
        }

    def _classify_custom(self, audio, sample_rate):
        if self.custom_model is None:
            return []
        try:
            predictions = self.custom_model.predict_proba(audio, sample_rate)
        except Exception as exc:
            self.custom_error = str(exc)
            return []
        results = []
        for item in predictions[:5]:
            label = item['label']
            results.append({
                'label': label,
                'category': label if label in (
                    'speech', 'yelling', 'bird_singing', 'door_opening',
                    'fire_burning', 'noise', 'music') else category_for_label(label),
                'score': float(item['score']),
                'source': 'custom',
            })
        return results

    def _classify_ast(self, audio, sample_rate):
        if not self.loaded or self.model is None or self.feature_extractor is None:
            return []

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return []

        if int(round(sample_rate)) != 16000:
            audio = resample(audio, sample_rate, 16000)
        audio = np.nan_to_num(audio.astype(np.float32, copy=False))

        # AST обучен на 10-секундных фрагментах 16 кГц.
        target = 16000 * 10
        if audio.size > target:
            audio = audio[:target]
        elif audio.size < target:
            audio = np.pad(audio, (0, target - audio.size))

        import torch

        with _INFERENCE_LOCK:
            inputs = self.feature_extractor(
                audio, sampling_rate=16000, return_tensors='pt')
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                logits = self.model(**inputs).logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
            topk = torch.topk(probabilities, k=min(5, probabilities.numel()))

            results = []
            id2label = self.model.config.id2label
            for score, index in zip(topk.values, topk.indices):
                label = id2label.get(index.item(), f'class_{index.item()}')
                results.append({
                    'label': label,
                    'category': category_for_label(label),
                    'score': float(score.item()),
                    'source': 'audioset',
                })
        return results

    def classify(self, audio, sample_rate=44100):
        if not self.loaded:
            self.load()

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return []

        results = []
        results.extend(self._classify_custom(audio, sample_rate))
        results.extend(self._classify_ast(audio, sample_rate))

        # Убираем дубли категорий от двух моделей, оставляя лучший score.
        merged = {}
        for item in results:
            key = (item.get('category'), item.get('label'))
            current = merged.get(key)
            if current is None or item['score'] > current['score']:
                merged[key] = item
        ordered = sorted(
            merged.values(), key=lambda item: item.get('score', 0.0),
            reverse=True)
        return ordered[:8]
