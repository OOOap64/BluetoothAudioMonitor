"""Признаки звука для собственного классификатора.

Лёгкий акустический «отпечаток» на основе мел-спектрограммы и MFCC.
Не требует нейросетей и GPU, поэтому модель можно обучать на обычном
ноутбуке за секунды/минуты. Для обучения используется
:mod:`src.custom_sound_model`, для инференса — :class:`SoundClassifier`.
"""

from __future__ import annotations

import numpy as np

from src.audio_utils import resample


FEATURE_VERSION = 1
FEATURE_CONFIG = {
    'n_mfcc': 20,
    'n_mels': 64,
    'n_fft': 1024,
    'win_ms': 25.0,
    'hop_ms': 10.0,
    'max_seconds': 10.0,
    'fmin': 20.0,
    'fmax': 8000.0,
}

_MEL_CACHE = {}


def _hz_to_mel(freq):
    return 2595.0 * np.log10(1.0 + np.asarray(freq, dtype=np.float64) / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(sample_rate, n_mels=64, n_fft=1024, fmin=20.0,
                   fmax=None):
    """Матрица треугольных мел-фильтров формы (n_mels, n_fft//2+1)."""
    sample_rate = int(sample_rate)
    fmax = min(float(fmax or sample_rate / 2.0), sample_rate / 2.0)
    key = (sample_rate, int(n_mels), int(n_fft), float(fmin), float(fmax))
    cached = _MEL_CACHE.get(key)
    if cached is not None:
        return cached

    n_bins = n_fft // 2 + 1
    mel_points = np.linspace(
        float(_hz_to_mel(fmin)), float(_hz_to_mel(fmax)), n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(np.int64)
    bins = np.clip(bins, 0, n_bins - 1)
    # Убираем возможные одинаковые границы на низких частотах.
    for index in range(1, len(bins)):
        if bins[index] <= bins[index - 1]:
            bins[index] = min(n_bins - 1, bins[index - 1] + 1)

    filters = np.zeros((n_mels, n_bins), dtype=np.float32)
    for mel in range(1, n_mels + 1):
        left, center, right = bins[mel - 1], bins[mel], bins[mel + 1]
        if center == left:
            center = min(n_bins - 1, left + 1)
        if right == center:
            right = min(n_bins, center + 1)
        if right <= center:
            filters[mel - 1, max(0, center - 1)] = 1.0
            continue
        for k in range(left, center):
            filters[mel - 1, k] = (k - left) / max(1, center - left)
        for k in range(center, right):
            filters[mel - 1, k] = (right - k) / max(1, right - center)
    _MEL_CACHE[key] = filters
    return filters


def _frame_signal(audio, frame_size, hop_size):
    if audio.size <= frame_size:
        audio = np.pad(audio, (0, frame_size - audio.size))
    frames = np.lib.stride_tricks.sliding_window_view(
        audio, frame_size)[::hop_size]
    return np.ascontiguousarray(frames, dtype=np.float32)


def _spectral_stats(power, freqs):
    total = np.maximum(power.sum(axis=1), 1e-12)
    centroid = (power * freqs).sum(axis=1) / total
    bandwidth = np.sqrt(
        (power * (freqs - centroid[:, None]) ** 2).sum(axis=1) / total)
    cumulative = np.cumsum(power, axis=1)
    rolloff_index = np.argmax(
        cumulative >= 0.85 * total[:, None], axis=1)
    rolloff = freqs[rolloff_index]
    flatness = np.exp(np.mean(np.log(np.maximum(power, 1e-12)), axis=1)) / total
    return np.stack([centroid, bandwidth, rolloff, flatness], axis=1)


def _delta(features):
    if features.shape[1] < 2:
        return np.zeros_like(features)
    padded = np.pad(features, ((0, 0), (1, 1)), mode='edge')
    return (padded[:, 2:] - padded[:, :-2]) / 2.0


def extract_features(audio, sample_rate, config=None):
    """Возвращает фиксированный вектор признаков (float32)."""
    cfg = dict(FEATURE_CONFIG)
    if config:
        cfg.update({key: value for key, value in config.items()
                    if key in cfg})
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    audio = np.nan_to_num(audio, copy=False)
    sample_rate = int(round(float(sample_rate or 16000)))
    if sample_rate <= 0:
        sample_rate = 16000

    max_samples = int(cfg['max_seconds'] * sample_rate)
    if audio.size > max_samples:
        # Берём середину записи: там обычно самое информативное событие.
        start = max(0, (audio.size - max_samples) // 2)
        audio = audio[start:start + max_samples]
    if audio.size == 0:
        audio = np.zeros(int(0.5 * sample_rate), dtype=np.float32)

    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak

    frame_size = max(64, int(sample_rate * cfg['win_ms'] / 1000.0))
    hop_size = max(1, int(sample_rate * cfg['hop_ms'] / 1000.0))
    n_fft = max(frame_size, int(cfg['n_fft']))
    if n_fft & (n_fft - 1):
        n_fft = 1 << int(np.ceil(np.log2(n_fft)))

    frames = _frame_signal(audio, frame_size, hop_size)
    window = np.hanning(frame_size).astype(np.float32)
    spectrum = np.fft.rfft(frames * window, n=n_fft, axis=1)
    power = (np.abs(spectrum) ** 2).astype(np.float32)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate).astype(np.float32)

    bank = mel_filterbank(
        sample_rate, n_mels=int(cfg['n_mels']), n_fft=n_fft,
        fmin=float(cfg['fmin']), fmax=float(cfg['fmax']))
    mel = np.maximum(np.dot(power, bank.T), 1e-10)
    log_mel = np.log(mel)

    # MFCC через DCT-II (scipy есть в requirements; fallback — numpy DCT).
    try:
        from scipy.fft import dct
        mfcc = dct(log_mel, type=2, axis=1, norm='ortho')[
            :, :int(cfg['n_mfcc'])]
    except Exception:
        length = log_mel.shape[1]
        n = np.arange(length)
        basis = np.cos(np.pi / length * (n[:, None] + 0.5) * n[None, :])
        mfcc = np.dot(log_mel, basis.T)[:, :int(cfg['n_mfcc'])]
        mfcc *= np.sqrt(2.0 / length)
    mfcc = mfcc.astype(np.float32)
    delta = _delta(mfcc).astype(np.float32)

    spectral = _spectral_stats(power, freqs).astype(np.float32)
    zcr = np.mean(
        np.abs(np.diff(np.signbit(frames).astype(np.int8), axis=1)), axis=1)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))

    def stats(matrix):
        return np.concatenate([
            matrix.mean(axis=0),
            matrix.std(axis=0),
            matrix.min(axis=0),
            matrix.max(axis=0),
            np.percentile(matrix, 10, axis=0),
            np.percentile(matrix, 90, axis=0),
        ])

    extra = np.stack([zcr, rms], axis=1).astype(np.float32)
    vector = np.concatenate([
        stats(mfcc),
        stats(delta),
        stats(extra),
        stats(spectral),
    ]).astype(np.float32)
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    return vector


def feature_size(config=None):
    """Размер вектора признаков при заданной конфигурации."""
    cfg = dict(FEATURE_CONFIG)
    if config:
        cfg.update({key: value for key, value in config.items()
                    if key in cfg})
    # 6 статистик у mfcc, delta, extra(2) и spectral(4).
    return int(6 * (cfg['n_mfcc'] * 2 + 2 + 4))
