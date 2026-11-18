"""Общие функции обработки аудио.

Здесь собраны преобразования сигнала и расчёт характеристик, которые
используются и при захвате звука в реальном времени, и при анализе файлов.
"""

from __future__ import annotations

import os
import wave
from functools import lru_cache
from math import gcd
from pathlib import Path

import numpy as np


# Частотные диапазоны, которые показываются в интерфейсе.
BANDS = (
    (20, 60),       # суб-бас
    (60, 250),      # бас
    (250, 2000),    # середина
    (2000, 4000),   # верхняя середина
    (4000, 20000),  # высокие
)


def to_mono(audio):
    """Приводит массив к моно float32.

    Поддерживаются обе раскладки: (frames, channels) и (channels, frames).
    """
    arr = np.asarray(audio)
    if arr.ndim == 2:
        arr = arr.astype(np.float32, copy=False)
        # Для потокового аудио более привычна раскладка (frames, channels).
        if arr.shape[1] <= 8:
            arr = arr.mean(axis=1)
        else:
            arr = arr.mean(axis=0)
    audio = np.asarray(arr, dtype=np.float32).reshape(-1)
    return np.nan_to_num(audio, copy=False)


def resample(audio, src_rate, dst_rate):
    """Ресемплинг сигнала (через scipy, без тяжёлого librosa)."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    try:
        src_rate = int(round(float(src_rate)))
        dst_rate = int(round(float(dst_rate)))
    except (TypeError, ValueError):
        return audio
    if audio.size == 0 or src_rate <= 0 or dst_rate <= 0 or src_rate == dst_rate:
        return audio

    # scipy импортируется лениво: при старте GUI он не нужен.
    from scipy.signal import resample_poly

    divisor = gcd(src_rate, dst_rate)
    out = resample_poly(audio, dst_rate // divisor, src_rate // divisor)
    return np.ascontiguousarray(out, dtype=np.float32)


@lru_cache(maxsize=16)
def _hanning(size):
    return np.hanning(size).astype(np.float32)


def normalize_for_asr(audio, target_peak=0.5):
    """Осторожно усиливает тихую запись перед распознаванием речи."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak <= 0.0:
        return audio
    if peak < 0.1:
        audio = audio * (target_peak / peak)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def _spectrum(audio, rate, average=False):
    """Возвращает частоты и спектральную мощность."""
    if audio.size < 16:
        return None, None

    if average and audio.size > 32768:
        frame_size = 8192 if audio.size >= 65536 else 4096
        step = frame_size // 2
        window = _hanning(frame_size)
        accumulated = None
        count = 0
        for start in range(0, audio.size - frame_size + 1, step):
            chunk = audio[start:start + frame_size] * window
            power = np.abs(np.fft.rfft(chunk)) ** 2
            accumulated = power if accumulated is None else accumulated + power
            count += 1
        if count == 0 or accumulated is None:
            return None, None
        return (
            np.fft.rfftfreq(frame_size, 1.0 / rate),
            accumulated / count,
        )

    data = audio[-32768:] if audio.size > 32768 else audio
    window = _hanning(data.size)
    power = np.abs(np.fft.rfft(data * window)) ** 2
    return np.fft.rfftfreq(data.size, 1.0 / rate), power


def compute_characteristics(audio, rate, average=False, duration=None):
    """Считает набор характеристик сигнала.

    ``average=True`` усредняет спектр по кадрам — так удобнее показывать
    характеристики уже готовой записи. Для потока достаточно одного блока.
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return {
            'rms': 0.0,
            'peak': 0.0,
            'amplitude': 0.0,
            'db_level': -100.0,
            'level_percent': 0,
            'spectral_centroid': 0.0,
            'dominant_frequency': 0.0,
            'crest_factor': 0.0,
            'frequency_bands': [0.0] * len(BANDS),
            'duration': float(duration or 0.0),
        }

    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    peak = float(np.max(np.abs(audio)))
    amplitude = float(np.mean(np.abs(audio)))

    db_level = 20.0 * np.log10(rms) if rms > 0.0 else -100.0
    db_level = float(max(-100.0, db_level))
    # -60..0 dBFS -> 0..100 %
    level_percent = int(max(0, min(100, round((db_level + 60.0) / 60.0 * 100.0))))
    crest_factor = float(peak / rms) if rms > 0.0 else 0.0

    spectral_centroid = 0.0
    dominant_frequency = 0.0
    band_levels = [0.0] * len(BANDS)

    freqs, power = _spectrum(audio, rate, average=average)
    if freqs is not None and power is not None:
        total_power = float(np.sum(power))
        if total_power > 0.0:
            spectral_centroid = float(np.sum(freqs * power) / total_power)
            start = 1 if power.size > 1 else 0
            if power[start:].size:
                dominant_frequency = float(freqs[start + int(np.argmax(power[start:]))])
            for index, (low, high) in enumerate(BANDS):
                mask = (freqs >= low) & (freqs < high)
                if np.any(mask):
                    band_levels[index] = float(np.sum(power[mask]) / total_power * 100.0)

    if duration is None:
        duration = audio.size / float(rate) if rate else 0.0

    return {
        'rms': rms,
        'peak': peak,
        'amplitude': amplitude,
        'db_level': db_level,
        'level_percent': level_percent,
        'spectral_centroid': spectral_centroid,
        'dominant_frequency': dominant_frequency,
        'crest_factor': crest_factor,
        'frequency_bands': band_levels,
        'duration': float(duration),
    }


def _normalize_samples(array):
    if array.dtype == np.float32 or array.dtype == np.float64:
        return array.astype(np.float32, copy=False)
    if array.dtype == np.uint8:
        return (array.astype(np.float32) - 128.0) / 128.0
    if array.dtype == np.int16:
        return array.astype(np.float32) / 32768.0
    if array.dtype == np.int32:
        return array.astype(np.float32) / 2147483648.0
    return array.astype(np.float32)


def _read_audio_frames(path):
    """Читает файл в массив ``(frames, channels)`` float32.

    Сначала используется ``soundfile`` (WAV/FLAC/OGG/Opus и др.), затем PyAV
    как запасной декодер для форматов, которые libsndfile не поддерживает.
    """
    errors = []

    try:
        import soundfile as sf
        data, rate = sf.read(path, dtype='float32', always_2d=True)
        if data.size:
            return np.ascontiguousarray(data, dtype=np.float32), int(rate)
        errors.append('soundfile: файл не содержит аудиоданных')
    except Exception as exc:
        errors.append(f'soundfile: {exc}')

    try:
        import av
    except Exception as exc:
        raise RuntimeError(
            'не удалось прочитать аудиофайл: ' + ' | '.join(errors)
        ) from exc

    chunks = []
    rate = 0
    with av.open(str(path)) as container:
        stream = next(
            (item for item in container.streams if item.type == 'audio'), None)
        if stream is None:
            raise ValueError('в файле нет аудиодорожки')

        for frame in container.decode(stream):
            if rate <= 0:
                try:
                    rate = int(frame.sample_rate or 0)
                except Exception:
                    pass
            array = _normalize_samples(frame.to_ndarray())
            try:
                frame_channels = int(frame.layout.nb_channels or 1)
            except Exception:
                frame_channels = 1
            frame_channels = max(1, frame_channels)

            if array.ndim == 1:
                array = array.reshape(-1, 1)
            elif getattr(frame.format, 'is_planar', False):
                array = array.reshape(frame_channels, -1).T
            else:
                array = array.reshape(-1, frame_channels)
            chunks.append(np.ascontiguousarray(array, dtype=np.float32))

    if not chunks:
        errors.append('PyAV: не удалось прочитать ни одного аудиокадра')
        raise RuntimeError(
            'не удалось прочитать аудиофайл: ' + ' | '.join(errors))
    if rate <= 0:
        raise ValueError('не удалось определить частоту дискретизации')

    return np.concatenate(chunks, axis=0), rate


def convert_audio_to_wav(source_path, output_path=None, *, sample_rate=None,
                         channels=None):
    """Преобразует аудиофайл (например, запись ``.opus``) в WAV.

    По умолчанию WAV создаётся рядом с исходным файлом и сохраняет исходные
    частоту дискретизации и число каналов. ``sample_rate`` пересемплирует
    сигнал, ``channels`` приводит число каналов (моно смешивает каналы).
    Возвращает словарь с параметрами сохранённой записи.
    """
    source = Path(source_path)
    if not source.is_file():
        raise FileNotFoundError(f'аудиофайл не найден: {source}')

    destination = Path(output_path) if output_path else source.with_suffix('.wav')
    if os.path.abspath(str(source)) == os.path.abspath(str(destination)):
        raise ValueError('путь назначения совпадает с исходным файлом')
    if destination.suffix.lower() != '.wav':
        destination = destination.with_suffix('.wav')

    data, rate = _read_audio_frames(source)

    if channels is not None:
        channels = max(1, int(channels))
        if data.shape[1] > channels:
            if channels == 1:
                data = data.mean(axis=1, keepdims=True)
            else:
                data = data[:, :channels]
        elif data.shape[1] < channels:
            if data.shape[1] == 1:
                data = np.repeat(data, channels, axis=1)
            else:
                raise ValueError(
                    f'в записи {data.shape[1]} каналов, '
                    f'нельзя получить {channels}')

    if sample_rate is not None:
        sample_rate = int(sample_rate)
        if sample_rate <= 0:
            raise ValueError('частота дискретизации должна быть положительной')
        if sample_rate != rate:
            if data.shape[1] == 1:
                data = resample(
                    data[:, 0], rate, sample_rate).reshape(-1, 1)
            else:
                data = np.stack([
                    resample(data[:, index], rate, sample_rate)
                    for index in range(data.shape[1])
                ], axis=1)
            rate = sample_rate

    info = save_wav(
        destination, data, rate, channels=int(data.shape[1]))
    info['source'] = os.fspath(source)
    info['source_format'] = source.suffix.lower().lstrip('.') or 'unknown'
    return info


def opus_to_wav(source_path, output_path=None, **kwargs):
    """Преобразует запись Ogg/Opus в WAV (обёртка над convert_audio_to_wav)."""
    return convert_audio_to_wav(source_path, output_path, **kwargs)


convert_opus_to_wav = opus_to_wav


def save_wav(path, audio, rate, channels=1):
    """Сохраняет сигнал в WAV-файл (PCM 16 бит).

    Используется для сохранения проанализированного аудио. Сигнал приводится
    к float32, при необходимости нормализуется по пику и клипуется, после чего
    записывается как знаковые 16-битные отсчёты. Возвращает словарь с
    параметрами записи (путь, частота, каналы, длительность, размер).
    """
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    else:
        arr = arr.reshape(arr.shape[0], -1)

    try:
        rate = int(round(float(rate)))
    except (TypeError, ValueError):
        rate = 0

    try:
        channels = int(channels)
    except (TypeError, ValueError):
        channels = arr.shape[1] or 1
    channels = max(1, min(channels, arr.shape[1] or 1))
    arr = arr[:, :channels]

    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak > 1.0:
        arr = arr / peak

    samples = np.clip(arr, -1.0, 1.0)
    samples = (samples * 32767.0).astype(np.int16)

    path = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)

    with wave.open(path, 'wb') as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())

    frames = int(samples.shape[0])
    return {
        'status': 'ok',
        'path': path,
        'format': 'wav',
        'sample_rate': rate,
        'channels': channels,
        'duration_seconds': round(frames / float(rate), 3) if rate else 0.0,
        'size_bytes': os.path.getsize(path) if os.path.exists(path) else 0,
    }
