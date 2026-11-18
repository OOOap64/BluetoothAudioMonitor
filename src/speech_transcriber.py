"""Распознавание речи через faster-whisper.

Модель Whisper ожидает моно-сигнал с частотой 16 кГц. Модуль приводит частоту
автоматически, убирает постоянную составляющую и низкочастотный гул, мягко
нормализует уровень и только затем отдаёт звук модели.

Для живого распознавания по умолчанию используется модель ``tiny``: на CPU она
даёт результат примерно за 0.3 с и позволяет уложиться в задержку до 1 с.
Размер можно переопределить переменной окружения
``BLUETOOTH_MONITOR_WHISPER_MODEL`` (``tiny``, ``base``, ``small``, ``medium``,
``large-v3``); более крупные модели точнее, но медленнее. Если выбранная модель
недоступна (например, нет интернета при первом запуске), автоматически
используется ``base``.

Скорость также регулируется переменными ``BLUETOOTH_MONITOR_WHISPER_BEAM``
(по умолчанию 1 вместо 5) и ``BLUETOOTH_MONITOR_WHISPER_THREADS``
(по умолчанию до 8 потоков CPU).

Точность конкретных фраз повышается без смены модели: список фраз передаётся
Whisper через ``initial_prompt`` и ``hotwords``. Фразы задаются в GUI, в файле
``data/asr_phrases.txt`` или переменными ``BLUETOOTH_MONITOR_PHRASES`` /
``BLUETOOTH_MONITOR_PHRASES_FILE``.
"""

from __future__ import annotations

import os
import re
import threading
import time

import numpy as np

from src.audio_utils import normalize_for_asr, resample
from src.text_corrector import correct_speech_text


_MODEL_CACHE = {}
_FAILED_LOADS = {}
_LOAD_LOCKS = {}
_CACHE_LOCK = threading.Lock()
_DEVICE_CACHE = {}
_DEVICE_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()
_LOAD_RETRY_SECONDS = 5.0
_HIGHPASS_SOS = None


def clear_model_cache():
    with _CACHE_LOCK:
        _MODEL_CACHE.clear()
        _FAILED_LOADS.clear()


# Фразы, которые пользователь произносит чаще всего. Whisper получает их как
# initial_prompt/hotwords: это повышает точность без смены модели и без
# дополнительной задержки. Список можно заменить в GUI или через переменные
# BLUETOOTH_MONITOR_PHRASES / BLUETOOTH_MONITOR_PHRASES_FILE.
_DEFAULT_PHRASES = (
    'Проверка живого распознавания речи. Открой окно и включи свет.',
    'Открой окно и включи свет.',
    'Привет, меня зовут.',
    'Где моя большая?',
)
_MAX_PHRASES = 20

# Служебные «звуки», которые Whisper иногда вставляет вместо речи.
_NON_SPEECH_RE = re.compile(
    r'\[(?:music|applause|laughter|blank_audio|silence|noise|'
    r'музыка|аплодисменты|смех|тишина|шум)[^\]]*\]|'
    r'\((?:music|applause|laughter|музыка|смех|тишина)\)|'
    r'[♪♫]+',
    re.IGNORECASE,
)


def _env_model_size():
    return (os.environ.get('BLUETOOTH_MONITOR_WHISPER_MODEL') or '').strip()


def _default_model_size():
    return _env_model_size() or 'tiny'


def _env_int(name, default, low, high):
    try:
        value = int(os.environ.get(name) or default)
    except (TypeError, ValueError):
        value = default
    return max(low, min(high, value))


def _default_cpu_threads():
    return _env_int(
        'BLUETOOTH_MONITOR_WHISPER_THREADS',
        min(8, os.cpu_count() or 4), 1, 32)


def _detect_device():
    with _DEVICE_LOCK:
        if 'auto' not in _DEVICE_CACHE:
            try:
                import torch
                _DEVICE_CACHE['auto'] = (
                    'cuda' if torch.cuda.is_available() else 'cpu')
            except Exception:
                _DEVICE_CACHE['auto'] = 'cpu'
        return _DEVICE_CACHE['auto']


def _compute_candidates(device, requested):
    if requested and str(requested) not in ('auto', 'default'):
        return [str(requested)]
    if device == 'cuda':
        return ['float16', 'int8_float16', 'int8', 'float32']
    return ['int8', 'float32']


def default_phrases():
    return list(_DEFAULT_PHRASES)


def _normalize_phrases(phrases):
    if phrases is None:
        return list(_DEFAULT_PHRASES)
    if isinstance(phrases, (list, tuple, set)):
        items = list(phrases)
    else:
        value = str(phrases)
        path = value.strip()
        if '\n' not in value and os.path.isfile(path):
            try:
                with open(path, 'r', encoding='utf-8', errors='replace') as fh:
                    items = fh.read().splitlines()
            except Exception:
                items = [value]
        else:
            items = re.split(r'[\n|;]+', value)
    result = []
    for item in items:
        text = ' '.join(str(item or '').split())
        if text and text not in result:
            result.append(text)
    return result[:_MAX_PHRASES]


def _resolve_phrases(phrases):
    if phrases is not None:
        return _normalize_phrases(phrases)
    env_value = os.environ.get('BLUETOOTH_MONITOR_PHRASES')
    if env_value:
        return _normalize_phrases(env_value)
    env_file = os.environ.get('BLUETOOTH_MONITOR_PHRASES_FILE')
    if env_file and os.path.isfile(env_file):
        return _normalize_phrases(env_file)
    default_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'asr_phrases.txt')
    if os.path.isfile(default_file):
        return _normalize_phrases(default_file)
    return _normalize_phrases(None)


def _phrase_prompt(phrases):
    if not phrases:
        return ''
    return 'Частые фразы, которые может произнести говорящий: ' + '; '.join(
        phrases)


def _prepare_audio(audio, sample_rate):
    """Готовит сигнал для Whisper: 16 кГц, без DC/гула, нормализованный."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return audio
    audio = resample(audio, sample_rate, 16000)
    if audio.size == 0:
        return audio
    audio = np.nan_to_num(audio, copy=False)
    # Постоянная составляющая и инфразвуковой гул заметно мешают Whisper.
    audio = audio - float(np.mean(audio, dtype=np.float64))
    try:
        from scipy.signal import butter, sosfilt
        global _HIGHPASS_SOS
        if _HIGHPASS_SOS is None:
            _HIGHPASS_SOS = butter(
                4, 80.0 / (16000.0 / 2.0), btype='highpass', output='sos')
        audio = sosfilt(_HIGHPASS_SOS, audio).astype(np.float32, copy=False)
    except Exception:
        pass
    audio = normalize_for_asr(audio, target_peak=0.7)
    return np.ascontiguousarray(np.clip(audio, -1.0, 1.0), dtype=np.float32)


def _clean_segment_text(text):
    text = str(text or '')
    text = _NON_SPEECH_RE.sub(' ', text)
    text = ' '.join(text.split())
    return text.strip()


class SpeechTranscriber:
    def __init__(self, model_size=None, device=None, compute_type='auto',
                 language='ru', autocorrect=True, beam_size=None,
                 cpu_threads=None, phrases=None):
        self.model_size = (
            model_size or _default_model_size()
        )
        # Импорт torch и выбор устройства откладываются до load(), чтобы
        # запуск захвата не задерживался на несколько секунд.
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.autocorrect = bool(autocorrect)
        self.beam_size = beam_size
        self.phrases = _resolve_phrases(phrases)
        self.phrase_prompt = _phrase_prompt(self.phrases)
        self.phrase_hotwords = '; '.join(self.phrases)
        self.cpu_threads = (
            _default_cpu_threads() if cpu_threads is None else
            max(1, min(32, int(cpu_threads))))
        self.model = None
        self.loaded = False
        self.model_label = ''

    def _resolve_device(self):
        if self.device:
            return self.device
        self.device = _detect_device()
        return self.device

    def _load_cached_model(self, model_size, device, compute_type):
        key = (model_size, device, compute_type, self.cpu_threads)
        now = time.monotonic()
        with _CACHE_LOCK:
            model = _MODEL_CACHE.get(key)
            if model is not None:
                return model
            failure = _FAILED_LOADS.get(key)
            if failure is not None and now - failure[0] < _LOAD_RETRY_SECONDS:
                raise failure[1]
            load_lock = _LOAD_LOCKS.setdefault(key, threading.RLock())
        with load_lock:
            now = time.monotonic()
            with _CACHE_LOCK:
                model = _MODEL_CACHE.get(key)
                if model is not None:
                    return model
                failure = _FAILED_LOADS.get(key)
                if failure is not None and now - failure[0] < _LOAD_RETRY_SECONDS:
                    raise failure[1]
            from faster_whisper import WhisperModel
            try:
                model = WhisperModel(
                    model_size,
                    device=device,
                    compute_type=compute_type,
                    cpu_threads=self.cpu_threads,
                    num_workers=1,
                )
            except Exception as exc:
                with _CACHE_LOCK:
                    _FAILED_LOADS[key] = (time.monotonic(), exc)
                raise
            with _CACHE_LOCK:
                _MODEL_CACHE[key] = model
                _FAILED_LOADS.pop(key, None)
            return model

    def load(self):
        if self.loaded and self.model is not None:
            return True

        device = self._resolve_device()
        sizes = [self.model_size]
        fallback = (
            os.environ.get('BLUETOOTH_MONITOR_WHISPER_FALLBACK', 'base')
            or 'base'
        ).strip()
        if fallback and fallback not in sizes:
            sizes.append(fallback)

        errors = []
        for size in sizes:
            for compute_type in _compute_candidates(device, self.compute_type):
                try:
                    model = self._load_cached_model(size, device, compute_type)
                except Exception as exc:
                    errors.append(f'{size}/{compute_type}: {exc}')
                    continue
                self.model = model
                self.model_size = size
                self.compute_type = compute_type
                self.model_label = f'{size} ({device}, {compute_type})'
                self.loaded = True
                return True

        self.loaded = False
        raise RuntimeError(
            'Не удалось загрузить модель распознавания речи. '
            + '; '.join(errors[-4:]))

    def transcribe_segments(self, audio, sample_rate=16000, language=None,
                            beam_size=None, vad_filter=False,
                            condition_on_previous_text=False, progress=None,
                            autocorrect=None, initial_prompt=None,
                            hotwords=None):
        """Возвращает список сегментов: [{'start', 'end', 'text'}]."""
        if autocorrect is None:
            autocorrect = self.autocorrect
        if not self.loaded or self.model is None:
            self.load()

        audio = _prepare_audio(audio, sample_rate)
        if audio.size == 0:
            return []

        if language in (None, '', 'auto', 'Auto'):
            language = None
        else:
            language = str(language)

        requested_beam = (
            self.beam_size if beam_size is None else beam_size)
        beam = _env_int('BLUETOOTH_MONITOR_WHISPER_BEAM', 1, 1, 10)
        try:
            if requested_beam is not None:
                beam = max(1, min(10, int(requested_beam)))
        except (TypeError, ValueError):
            pass

        options = {
            'language': language,
            'beam_size': beam,
            'best_of': beam,
            'vad_filter': bool(vad_filter),
            'condition_on_previous_text': bool(condition_on_previous_text),
            'no_speech_threshold': 0.6,
            'log_prob_threshold': -1.0,
            'compression_ratio_threshold': 2.4,
        }
        if vad_filter:
            options['vad_parameters'] = {
                'min_silence_duration_ms': 500,
                'speech_pad_ms': 300,
            }

        prompt = (
            initial_prompt if initial_prompt is not None else
            (self.phrase_prompt or
             os.environ.get('BLUETOOTH_MONITOR_WHISPER_PROMPT')))
        if prompt:
            options['initial_prompt'] = str(prompt)

        hint = (
            hotwords if hotwords is not None else
            (self.phrase_hotwords or
             os.environ.get('BLUETOOTH_MONITOR_WHISPER_HOTWORDS')))
        if hint:
            options['hotwords'] = str(hint)

        duration = max(audio.size / 16000.0, 0.001)
        segments = []
        with _INFERENCE_LOCK:
            raw_segments, _info = self.model.transcribe(audio, **options)
            collected = tuple(
                (
                    float(getattr(segment, 'start', 0.0)),
                    float(getattr(segment, 'end', 0.0)),
                    getattr(segment, 'text', ''),
                    float(getattr(segment, 'no_speech_prob', 0.0) or 0.0),
                    float(getattr(segment, 'avg_logprob', 0.0) or 0.0),
                )
                for segment in raw_segments
            )
        for start, end, raw_text, no_speech, logprob in collected:
            text = _clean_segment_text(raw_text)
            if not text:
                continue
            if no_speech > 0.75 and logprob < -1.0:
                continue
            if autocorrect:
                text = correct_speech_text(text)
            if not text:
                continue
            segments.append({
                'start': start,
                'end': end,
                'text': text,
            })
            if progress is not None:
                try:
                    progress(min(1.0, end / duration), text)
                except Exception:
                    pass
        return segments

    def transcribe(self, audio, sample_rate=44100, language=None,
                   autocorrect=None, **kwargs):
        # Сегменты намеренно не корректируем здесь по отдельности: итоговый
        # текст исправляется один раз, чтобы не добавлять лишние точки между
        # фрагментами одной фразы.
        use_autocorrect = (
            self.autocorrect if autocorrect is None else bool(autocorrect))
        segments = self.transcribe_segments(
            audio, sample_rate=sample_rate, language=language,
            autocorrect=False, **kwargs)
        text = ' '.join(segment['text'] for segment in segments).strip()
        if use_autocorrect:
            text = correct_speech_text(text)
        return text.strip()
