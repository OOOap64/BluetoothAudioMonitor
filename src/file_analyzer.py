"""Анализ готовых аудиозаписей.

Файл декодируется через PyAV (mp3, m4a/aac, ogg/opus, flac, wav, wma,
видеофайлы с аудиодорожкой и т.д.), а при его отсутствии — через soundfile.
Затем считаются характеристики сигнала и распознаётся речь.
"""

from __future__ import annotations

import os

import numpy as np

from src.audio_utils import compute_characteristics
from src.sound_classifier import SoundClassifier
from src.speech_transcriber import SpeechTranscriber


MAX_FILE_SECONDS = 30 * 60  # защита от переполнения памяти
AUDIO_EXTENSIONS = (
    '*.wav', '*.mp3', '*.flac', '*.ogg', '*.oga', '*.opus', '*.m4a',
    '*.aac', '*.wma', '*.aiff', '*.aif', '*.au', '*.mp4', '*.mkv',
    '*.webm', '*.mov', '*.avi',
)


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


def _decode_with_av(path):
    import av

    chunks = []
    sample_rate = 0
    channels = 1
    codec_name = 'unknown'
    sample_format = 'unknown'
    total_frames = 0

    with av.open(path) as container:
        stream = next(
            (item for item in container.streams if item.type == 'audio'), None)
        if stream is None:
            raise ValueError('в файле нет аудиодорожки')

        context = stream.codec_context
        try:
            sample_rate = int(context.sample_rate or 0)
        except Exception:
            sample_rate = 0
        try:
            channels = int(context.layout.nb_channels or 0)
        except Exception:
            channels = int(getattr(context, 'channels', 1) or 1)
        codec_name = getattr(context, 'name', None) or 'unknown'
        try:
            sample_format = context.format.name if context.format else 'unknown'
        except Exception:
            sample_format = 'unknown'

        for frame in container.decode(stream):
            if sample_rate <= 0:
                try:
                    sample_rate = int(frame.sample_rate or 0)
                except Exception:
                    pass
            array = frame.to_ndarray()
            array = _normalize_samples(array)

            try:
                frame_channels = int(frame.layout.nb_channels or channels or 1)
            except Exception:
                frame_channels = max(1, channels)
            frame_channels = max(1, frame_channels)

            if array.ndim == 1:
                array = array.reshape(1, -1)
            if getattr(frame.format, 'is_planar', False):
                array = array.reshape(frame_channels, -1)
            else:
                array = array.reshape(-1, frame_channels).T

            chunks.append(array.astype(np.float32, copy=False))
            total_frames += array.shape[1]

            if sample_rate > 0 and total_frames / sample_rate > MAX_FILE_SECONDS:
                raise ValueError(
                    f'файл длиннее {MAX_FILE_SECONDS // 60} минут; '
                    'анализ такого файла может занять слишком много памяти')

    if not chunks:
        raise ValueError('не удалось прочитать ни одного аудиокадра')
    if sample_rate <= 0:
        raise ValueError('не удалось определить частоту дискретизации')

    data = np.concatenate(chunks, axis=1)
    audio = data.mean(axis=0) if data.shape[0] > 1 else data[0]
    channels = int(data.shape[0])
    format_label = f'{codec_name} ({sample_format})' if sample_format else codec_name
    return np.ascontiguousarray(audio, dtype=np.float32), sample_rate, channels, format_label


def _decode_with_soundfile(path):
    import soundfile as sf

    info = sf.info(path)
    data, sample_rate = sf.read(path, dtype='float32', always_2d=True)
    if data.size == 0:
        raise ValueError('файл не содержит аудиоданных')
    if sample_rate > 0 and data.shape[0] / float(sample_rate) > MAX_FILE_SECONDS:
        raise ValueError(
            f'файл длиннее {MAX_FILE_SECONDS // 60} минут; '
            'анализ такого файла может занять слишком много памяти')
    audio = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    format_label = info.format or 'audio'
    if info.subtype:
        format_label = f'{format_label} ({info.subtype})'
    return np.ascontiguousarray(audio, dtype=np.float32), int(sample_rate), int(data.shape[1]), format_label


def decode_audio_file(path):
    """Декодирует файл любого поддерживаемого формата в моно float32."""
    errors = []

    try:
        return _decode_with_av(path)
    except Exception as exc:
        errors.append(f'PyAV: {exc}')

    try:
        return _decode_with_soundfile(path)
    except Exception as exc:
        errors.append(f'soundfile: {exc}')

    raise ValueError(
        'Не удалось прочитать аудиофайл. Поддерживаются WAV, MP3, FLAC, OGG/Opus, '
        'M4A/AAC, WMA и видеофайлы с аудиодорожкой. '
        + ' | '.join(errors))


def _format_time(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f'{hours:02d}:{minutes:02d}:{secs:02d}'
    return f'{minutes:02d}:{secs:02d}'


def format_segments(segments):
    return '\n'.join(
        f"[{_format_time(segment['start'])}] {segment['text']}"
        for segment in segments
    )


def analyze_audio_file(path, language='ru', progress=None, autocorrect=True,
                       model_size=None, phrases=None):
    """Полный анализ файла: характеристики + речь + классификация звука."""

    def report(fraction, message):
        if progress is not None:
            try:
                progress(max(0.0, min(1.0, float(fraction))), message)
            except Exception:
                pass

    report(0.01, 'Чтение аудиофайла...')
    audio, sample_rate, channels, format_label = decode_audio_file(path)
    duration = audio.size / float(sample_rate) if sample_rate else 0.0

    report(0.08, 'Расчёт характеристик звука...')
    characteristics = compute_characteristics(
        audio, sample_rate, average=True, duration=duration)

    result = {
        'file_path': path,
        'file_name': os.path.basename(path),
        'sample_rate': int(sample_rate),
        'channels': int(channels),
        'format': format_label,
        'duration': float(duration),
        'transcribed_text': '',
        'transcription_error': None,
        'autocorrect': bool(autocorrect),
        'classifications': [],
        'classification_error': None,
    }
    result.update(characteristics)

    # --- распознавание речи ---
    report(0.12, 'Загрузка модели распознавания речи...')
    transcriber = SpeechTranscriber(
        model_size=model_size, language=language, autocorrect=autocorrect,
        phrases=phrases)
    try:
        transcriber.load()
        report(0.22, f'Распознавание речи: {transcriber.model_label}...')

        def on_segment(fraction, _text):
            report(0.22 + 0.63 * fraction, 'Распознавание речи...')

        segments = transcriber.transcribe_segments(
            audio,
            sample_rate=sample_rate,
            language=language,
            vad_filter=True,
            progress=on_segment,
        )
        result['segments'] = segments
        result['transcribed_text'] = format_segments(segments)
    except Exception as exc:
        result['transcription_error'] = str(exc)
        report(0.85, f'Распознавание недоступно: {exc}')

    # --- классификация источника звука ---
    report(0.88, 'Определение источника звука...')
    try:
        classifier = SoundClassifier()
        classifier.load()
        # Для классификации достаточно 10 секунд из середины записи.
        window = 10 * sample_rate
        if audio.size > window:
            start = max(0, (audio.size - window) // 2)
            sample = audio[start:start + window]
        else:
            sample = audio
        result['classifications'] = classifier.classify(sample, sample_rate)
    except Exception as exc:
        result['classification_error'] = str(exc)

    report(1.0, 'Готово')
    return result
