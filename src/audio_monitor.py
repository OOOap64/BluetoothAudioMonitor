"""Захват и анализ аудио в реальном времени.

Поддерживаются два пути захвата:

* ``soundcard`` — WASAPI loopback (системный звук, в том числе Bluetooth);
* ``sounddevice`` — обычные микрофоны и другие устройства ввода.

Тяжёлые операции (Whisper, классификатор) выполняются в отдельном потоке,
поэтому аудиопоток не прерывается во время распознавания.
"""

from __future__ import annotations

import re
import threading
import time
import uuid
import warnings
from collections import deque, namedtuple
from datetime import datetime

import numpy as np
import sounddevice as sd

from src.audio_devices import (
    com_initialize, default_loopback_source, describe_source, sc
)
from src.audio_utils import compute_characteristics, save_wav, to_mono
from src.sound_classifier import SoundClassifier
from src.session_log import default_recordings_dir
from src.speech_transcriber import SpeechTranscriber
from src.text_corrector import correct_speech_text

try:
    from soundcard import SoundcardRuntimeWarning
    warnings.filterwarnings('ignore', category=SoundcardRuntimeWarning)
except Exception:
    pass


Utterance = namedtuple('Utterance', 'audio partial phrase_id captured_at')


def _normalize_merge_words(text):
    return re.sub(r'[^\w]+', '', str(text or '').lower())


def merge_partial_text(previous, new):
    """Склеивает перекрывающиеся частичные распознавания без дублей."""
    previous = ' '.join(str(previous or '').split())
    new = ' '.join(str(new or '').split())
    if not previous:
        return new
    if not new:
        return previous
    norm_previous = _normalize_merge_words(previous)
    norm_new = _normalize_merge_words(new)
    if norm_new in norm_previous:
        return previous
    if norm_previous in norm_new:
        return new
    old_words = previous.split()
    new_words = new.split()
    old_norm = [_normalize_merge_words(word) for word in old_words]
    new_norm = [_normalize_merge_words(word) for word in new_words]
    for overlap in range(min(len(old_words), len(new_words)), 0, -1):
        if old_norm[-overlap:] == new_norm[:overlap]:
            return ' '.join(old_words + new_words[overlap:])
    return ' '.join(old_words + new_words)


class _UtteranceDetector:
    """Энергетический детектор с быстрыми частичными результатами.

    Частичные окна отправляются каждые ``partial_interval`` секунд речи,
    поэтому текст появляется ещё до паузы. Полная фраза отправляется после
    короткой паузы и заменяет частичный текст.
    """

    def __init__(self, rate, frame_ms=20, start_frames=1,
                 end_silence=0.20, max_utterance=8.0,
                 pre_roll=0.15, min_utterance=0.2,
                 partial_interval=0.5, partial_window=1.0):
        self.rate = int(rate)
        self.frame_ms = int(frame_ms)
        self.frame_size = max(1, int(self.rate * frame_ms / 1000.0))
        self._leftover = np.zeros(0, dtype=np.float32)

        self._pre = deque(maxlen=max(1, int(pre_roll * 1000.0 / frame_ms)))
        self._noise = 0.001
        self._speech_run = 0
        self._start_frames = max(1, int(start_frames))
        self._end_frames = max(1, int(round(end_silence * 1000.0 / frame_ms)))
        self._max_frames = max(1, int(round(max_utterance * 1000.0 / frame_ms)))
        self._min_frames = max(1, int(round(min_utterance * 1000.0 / frame_ms)))
        self._partial_frames = max(
            1, int(round(partial_interval * 1000.0 / frame_ms)))
        self._window_frames = max(
            self._partial_frames,
            int(round(partial_window * 1000.0 / frame_ms)))

        self._in_speech = False
        self._frames = []
        self._frame_count = 0
        self._silence_frames = 0
        self._since_partial = 0
        self._phrase_id = 0

    def _reset(self):
        self._in_speech = False
        self._frames = []
        self._frame_count = 0
        self._silence_frames = 0
        self._speech_run = 0
        self._since_partial = 0
        self._pre.clear()

    def _threshold(self):
        return max(0.0035, self._noise * 3.0)

    def push(self, audio):
        """Добавляет аудио, возвращает частичные и завершённые фразы."""
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return []
        if self._leftover.size:
            audio = np.concatenate((self._leftover, audio))

        completed = []
        frames_total = audio.size // self.frame_size
        threshold = self._threshold()

        for index in range(frames_total):
            frame = audio[index * self.frame_size:(index + 1) * self.frame_size]
            rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
            is_speech = rms > threshold

            if not self._in_speech:
                if not is_speech:
                    # Медленно подстраиваем порог под фоновый шум.
                    self._noise = min(0.05, 0.98 * self._noise + 0.02 * max(rms, 1e-6))
                    threshold = self._threshold()
                self._pre.append(frame)
                if is_speech:
                    self._speech_run += 1
                    if self._speech_run >= self._start_frames:
                        self._in_speech = True
                        self._phrase_id += 1
                        self._frames = list(self._pre)
                        self._frame_count = len(self._frames)
                        self._silence_frames = 0
                        self._since_partial = 0
                        self._pre.clear()
                else:
                    self._speech_run = 0
                continue

            self._frames.append(frame)
            self._frame_count += 1
            self._since_partial += 1
            if is_speech:
                self._silence_frames = 0
            else:
                self._silence_frames += 1

            if (self._since_partial >= self._partial_frames and
                    self._frame_count >= self._min_frames):
                count = min(self._window_frames, len(self._frames))
                window = np.concatenate(self._frames[-count:])
                completed.append(Utterance(
                    window.astype(np.float32, copy=False), True,
                    self._phrase_id, time.time()))
                self._since_partial = 0

            if (self._silence_frames >= self._end_frames or
                    self._frame_count >= self._max_frames):
                utterance = np.concatenate(self._frames) if self._frames else None
                if utterance is not None and self._silence_frames:
                    # Убираем хвостовую тишину.
                    keep = utterance.size - self._silence_frames * self.frame_size
                    utterance = utterance[:max(0, keep)]
                if (utterance is not None and
                        self._frame_count >= self._min_frames and
                        utterance.size > 0):
                    completed.append(Utterance(
                        utterance.astype(np.float32, copy=False), False,
                        self._phrase_id, time.time()))
                self._reset()
                threshold = self._threshold()

        self._leftover = audio[frames_total * self.frame_size:].copy()
        return completed


class AudioMonitor:
    def __init__(self, source=None, chunk=1024, rate=48000, channels=1,
                 language='ru', log_callback=None, autocorrect=True,
                 model_size=None, phrases=None):
        self.source = source
        self.chunk = max(64, int(chunk))
        self.rate = int(rate)
        self.channels = int(channels)
        self.language = language
        self.log_callback = log_callback
        self.autocorrect = bool(autocorrect)
        self.model_size = model_size
        self.phrases = phrases

        self.stream = None
        self.running = False
        self.format = 'Float32'

        self.classifier = SoundClassifier()
        self.transcriber = SpeechTranscriber(
            model_size=model_size, language=language,
            autocorrect=self.autocorrect, phrases=phrases)

        self._lock = threading.RLock()
        self._pending_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._stream_error = None

        self._buffer = deque(maxlen=1024)
        self._pending = deque()
        self._pending_frames = 0
        self._total_frames = 0

        self._transcribe_queue = deque()
        self._transcript_parts = []
        self._last_text = ''
        self._partial_text = ''
        self._active_phrase_id = None
        self._last_recognition_ms = 0

        self._reader_thread = None
        self._worker_thread = None
        self._recorder = None
        self._soundcard_ready = threading.Event()
        self._soundcard_error = None

        self._ema = None
        self._peak_hold = 0.0
        self._last_status_log = 0.0

        self.latest_data = {}

    # ------------------------------------------------------------------ log
    def _log(self, message):
        if self.log_callback:
            try:
                self.log_callback(message)
                return
            except Exception:
                pass
        try:
            print(message)
        except Exception:
            pass

    def _fail(self, message):
        self._stream_error = message
        with self._lock:
            data = dict(self.latest_data)
            data['error'] = message
            self.latest_data = data
        self._log(message)
        self.running = False

    # ------------------------------------------------------------- opening
    def start(self):
        if self.running:
            return True

        # Поток, в котором запускается захват, может быть без COM.
        com_initialize()

        self._stop_event.clear()
        self._stream_error = None
        self._ema = None
        self._peak_hold = 0.0
        self._last_text = ''
        self._transcript_parts = []
        self._partial_text = ''
        self._active_phrase_id = None
        self._last_recognition_ms = 0
        self._transcribe_queue.clear()
        with self._lock:
            self.latest_data = {
                'sample_rate': 0,
                'channels': 0,
                'format': '--',
                'classifications': [],
                'transcribed_text': '',
                'live_text': '',
            }

        source = self.source or default_loopback_source()
        self.source = source
        self._log(f'Источник аудио: {describe_source(source)}')

        self.running = True
        try:
            if source and source.get('backend') == 'soundcard' and sc is not None:
                self._open_soundcard(source)
            else:
                self._open_sounddevice(source)
        except Exception:
            self.running = False
            self._stop_event.set()
            raise

        self._worker_thread = threading.Thread(
            target=self._analysis_loop, name='audio-analysis', daemon=True)
        self._worker_thread.start()

        self._log(
            f'Захват запущен: {describe_source(source)}; '
            f'SR={self.rate} Гц, каналов={self.channels}, формат={self.format}')
        self._log(
            'Автокоррекция речи: '
            + ('включена' if self.autocorrect else 'выключена'))
        return True

    def _init_buffer(self):
        # ~30 секунд истории достаточно для классификации и отображения.
        maxlen = max(64, int(30.0 * self.rate / self.chunk) + 4)
        with self._lock:
            self._buffer = deque(maxlen=maxlen)
            self._total_frames = 0
        with self._pending_lock:
            self._pending.clear()
            self._pending_frames = 0

    def _samplerates(self, preferred):
        rates = []
        for rate in (preferred, 48000, 44100, 16000):
            try:
                rate = int(rate)
            except (TypeError, ValueError):
                continue
            if rate > 0 and rate not in rates:
                rates.append(rate)
        return rates

    def _open_soundcard(self, source):
        microphone = sc.get_microphone(
            id=source.get('id'), include_loopback=bool(source.get('is_loopback', True)))
        if microphone is None:
            raise RuntimeError('WASAPI loopback устройство не найдено')

        try:
            available_channels = max(1, int(getattr(microphone, 'channels', 1) or 1))
        except Exception:
            available_channels = 2
        channel_options = [min(2, available_channels)]
        if channel_options[0] > 1:
            channel_options.append(1)

        self._soundcard_ready.clear()
        self._soundcard_error = None
        self._reader_thread = threading.Thread(
            target=self._soundcard_reader,
            args=(
                microphone,
                channel_options,
                self._samplerates(source.get('samplerate')),
            ),
            name='audio-capture',
            daemon=True,
        )
        self._reader_thread.start()

        # soundcard/MediaFoundation требует, чтобы recorder создавался,
        # использовался и закрывался в одном и том же потоке. Поэтому
        # открытие устройства выполняется внутри потока чтения, а здесь
        # мы только ждём результат.
        if not self._soundcard_ready.wait(timeout=8.0):
            self.running = False
            self._stop_event.set()
            raise RuntimeError(
                'Не удалось открыть WASAPI loopback: превышено время ожидания')

        if self._soundcard_error:
            self.running = False
            self._stop_event.set()
            message = self._soundcard_error
            self._soundcard_error = None
            raise RuntimeError(message)

    def _open_sounddevice(self, source):
        device = source.get('device') if source else None
        try:
            info = sd.query_devices(device, 'input')
        except Exception as exc:
            raise RuntimeError(f'Устройство ввода недоступно: {exc}') from exc

        max_input = int(info.get('max_input_channels', 1) or 1)
        preferred = source.get('samplerate') if source else None

        channel_options = [min(2, max_input)]
        if channel_options[0] > 1:
            channel_options.append(1)

        errors = []
        for channels in channel_options:
            for rate in self._samplerates(preferred):
                stream = None
                try:
                    stream = sd.InputStream(
                        device=device,
                        samplerate=rate,
                        channels=channels,
                        blocksize=self.chunk,
                        dtype='float32',
                        callback=self._sounddevice_callback,
                    )
                    self.rate = int(stream.samplerate)
                    self.channels = int(stream.channels)
                    self.format = 'Float32'
                    self._init_buffer()
                    stream.start()
                except Exception as exc:
                    errors.append(f'{rate} Гц / {channels} кан.: {exc}')
                    try:
                        if stream is not None:
                            stream.close()
                    except Exception:
                        pass
                    continue

                self.stream = stream
                return

        details = '; '.join(errors[-4:]) if errors else 'нет доступных форматов'
        raise RuntimeError(
            'Не удалось открыть устройство ввода. '
            f'Устройство: {source.get("name") if source else "по умолчанию"}. {details}')

    # ------------------------------------------------------------- capture
    def _sounddevice_callback(self, indata, frames, time_info, status):
        if status:
            now = time.time()
            if now - self._last_status_log > 2.0:
                self._last_status_log = now
                self._log(f'Статус аудиопотока: {status}')
        try:
            self._handle_block(indata)
        except Exception as exc:
            self._fail(f'Ошибка обработки аудиоблока: {exc}')

    def _soundcard_reader(self, microphone, channel_options, rates):
        # soundcard работает через COM/MediaFoundation; в новом потоке
        # библиотеку нужно инициализировать заново.
        com_initialize()
        errors = []
        recorder = None
        try:
            for channels in channel_options:
                for rate in rates:
                    if self._stop_event.is_set():
                        break
                    candidate = None
                    try:
                        candidate = microphone.recorder(
                            samplerate=rate, channels=channels, blocksize=self.chunk)
                        candidate.__enter__()
                    except Exception as exc:
                        errors.append(f'{rate} Гц / {channels} кан.: {exc}')
                        try:
                            if candidate is not None:
                                candidate.__exit__(None, None, None)
                        except Exception:
                            pass
                        continue

                    recorder = candidate
                    self.rate = int(rate)
                    self.channels = int(channels)
                    self.format = 'Float32 (WASAPI loopback)'
                    self._init_buffer()
                    self._recorder = recorder
                    self._soundcard_ready.set()
                    break
                if recorder is not None:
                    break

            if recorder is None:
                details = '; '.join(errors[-4:]) if errors else 'нет доступных форматов'
                self._soundcard_error = (
                    'Не удалось открыть WASAPI loopback. '
                    f'Устройство: {getattr(microphone, "name", "?")}. {details}')
                return

            while self.running and not self._stop_event.is_set():
                data = recorder.record(numframes=self.chunk)
                if data is None or len(data) == 0:
                    continue
                self._handle_block(data)
        except Exception as exc:
            if not self._soundcard_ready.is_set():
                self._soundcard_error = f'Ошибка открытия аудиопотока: {exc}'
            elif self.running and not self._stop_event.is_set():
                self._fail(f'Ошибка чтения аудиопотока: {exc}')
        finally:
            self._recorder = None
            if recorder is not None:
                try:
                    recorder.__exit__(None, None, None)
                except Exception:
                    pass
            self._soundcard_ready.set()

    def _handle_block(self, data):
        audio = to_mono(data)
        if audio.size == 0:
            return

        with self._lock:
            self._buffer.append(audio.copy())
            self._total_frames += audio.size
            total_frames = self._total_frames

        with self._pending_lock:
            self._pending.append(audio.copy())
            self._pending_frames += audio.size
            # Не даём очереди расти бесконечно, если распознавание занято.
            while (self._pending_frames > 120 * self.rate and self._pending):
                dropped = self._pending.popleft()
                self._pending_frames -= dropped.size

        characteristics = compute_characteristics(audio, self.rate)
        self._update_realtime(characteristics, total_frames)

    def _update_realtime(self, characteristics, total_frames):
        with self._lock:
            if self._ema is None:
                self._ema = {
                    'rms': characteristics['rms'],
                    'amplitude': characteristics['amplitude'],
                    'db_level': characteristics['db_level'],
                    'spectral_centroid': characteristics['spectral_centroid'],
                    'frequency_bands': list(characteristics['frequency_bands']),
                }
            else:
                alpha = 0.25
                for key in ('rms', 'amplitude', 'db_level', 'spectral_centroid'):
                    self._ema[key] = (
                        (1.0 - alpha) * self._ema[key] + alpha * characteristics[key])
                self._ema['frequency_bands'] = [
                    (1.0 - alpha) * old + alpha * new
                    for old, new in zip(
                        self._ema['frequency_bands'], characteristics['frequency_bands'])
                ]

            # Пиковый уровень держится около секунды и плавно спадает.
            self._peak_hold = max(characteristics['peak'], self._peak_hold * 0.92)

            rms = self._ema['rms']
            db_level = max(-100.0, min(0.0, self._ema['db_level']))
            data = dict(self.latest_data)
            data.update({
                'sample_rate': self.rate,
                'channels': self.channels,
                'format': self.format,
                'rms': rms,
                'peak': self._peak_hold,
                'amplitude': self._ema['amplitude'],
                'db_level': db_level,
                'level_percent': int(max(0, min(100, round((db_level + 60.0) / 60.0 * 100.0)))),
                'spectral_centroid': self._ema['spectral_centroid'],
                'dominant_frequency': characteristics['dominant_frequency'],
                'crest_factor': (self._peak_hold / rms) if rms > 0 else 0.0,
                'frequency_bands': list(self._ema['frequency_bands']),
                'duration': total_frames / float(self.rate) if self.rate else 0.0,
                'timestamp': time.time(),
            })
            data.pop('error', None)
            self.latest_data = data

    # ------------------------------------------------------------ analysis
    def _snapshot(self, seconds):
        with self._lock:
            blocks = list(self._buffer)
        if not blocks:
            return None
        needed = max(1, int(seconds * self.rate))
        parts = []
        total = 0
        for block in reversed(blocks):
            parts.append(block)
            total += block.size
            if total >= needed:
                break
        if not parts:
            return None
        audio = np.concatenate(parts[::-1])
        if audio.size > needed:
            audio = audio[-needed:]
        return audio

    def snapshot_audio(self, seconds=None):
        """Возвращает накопленный в буфере сигнал (моно float32).

        Без ``seconds`` берётся весь буфер (около 30 секунд последнего звука),
        с ``seconds`` — только последние N секунд.
        """
        with self._lock:
            blocks = list(self._buffer)
        if not blocks:
            return None
        audio = blocks[0] if len(blocks) == 1 else np.concatenate(blocks)
        if seconds is not None and seconds > 0 and self.rate:
            needed = max(1, int(seconds * self.rate))
            if audio.size > needed:
                audio = audio[-needed:]
        return audio

    def save_recording(self, path=None, seconds=None):
        """Сохраняет проанализированное аудио в WAV-файл.

        Если ``path`` не задан, файл кладётся в каталог записей журнала
        (``data/read_aloud/recordings``) с именем ``capture-*.wav``.
        Возвращает словарь со статусом, путём, длительностью и размером.
        """
        audio = self.snapshot_audio(seconds)
        if audio is None or audio.size == 0:
            return {
                'status': 'empty',
                'path': None,
                'format': 'wav',
                'sample_rate': int(self.rate),
                'channels': 1,
                'duration_seconds': 0.0,
                'size_bytes': 0,
                'error': 'Нет данных для сохранения',
            }

        if not path:
            try:
                directory = default_recordings_dir()
                directory.mkdir(parents=True, exist_ok=True)
                name = (
                    f'capture-{time.strftime("%Y%m%d-%H%M%S")}-'
                    f'{uuid.uuid4().hex[:8]}.wav')
                path = directory / name
            except Exception as exc:
                return {
                    'status': 'error',
                    'path': None,
                    'format': 'wav',
                    'sample_rate': int(self.rate),
                    'channels': 1,
                    'duration_seconds': 0.0,
                    'size_bytes': 0,
                    'error': f'не удалось выбрать каталог записи: {exc}',
                }

        try:
            info = save_wav(path, audio, self.rate, channels=1)
        except Exception as exc:
            return {
                'status': 'error',
                'path': None,
                'format': 'wav',
                'sample_rate': int(self.rate),
                'channels': 1,
                'duration_seconds': 0.0,
                'size_bytes': 0,
                'error': f'не удалось сохранить запись: {exc}',
            }
        return info

    def _load_classifier(self):
        try:
            self.classifier.load()
            self._log('Модель классификации звука загружена')
        except Exception as exc:
            self._log(f'Классификация звука недоступна: {exc}')

    def _analysis_loop(self):
        self._log('Загрузка модели распознавания речи...')
        try:
            self.transcriber.load()
            self._log(
                'Модель распознавания речи загружена: '
                + self.transcriber.model_label)
        except Exception as exc:
            self._log(f'Распознавание речи недоступно: {exc}')
        threading.Thread(
            target=self._load_classifier, name='sound-classifier-loader',
            daemon=True).start()

        detector = _UtteranceDetector(self.rate)
        last_classify = 0.0

        while self.running and not self._stop_event.is_set():
            self._drain_transcribe_queue()

            with self._pending_lock:
                blocks = list(self._pending)
                self._pending.clear()
                self._pending_frames = 0

            if blocks:
                audio = blocks[0] if len(blocks) == 1 else np.concatenate(blocks)
                for utterance in detector.push(audio):
                    self._enqueue_utterance(utterance)
                self._drain_transcribe_queue()

            now = time.time()
            if now - last_classify >= 2.5:
                last_classify = now
                if self.classifier.loaded:
                    window = self._snapshot(4.0)
                    if window is not None and window.size:
                        rms = float(np.sqrt(np.mean(np.square(window, dtype=np.float64))))
                        if rms > 0.002:
                            try:
                                results = self.classifier.classify(window, self.rate)
                            except Exception as exc:
                                self._log(f'Ошибка классификации: {exc}')
                                results = []
                            if results:
                                with self._lock:
                                    data = dict(self.latest_data)
                                    data['classifications'] = results
                                    self.latest_data = data

            self._stop_event.wait(0.05 if blocks else 0.1)

    def _drain_transcribe_queue(self, limit=6, budget_ms=30):
        deadline = time.monotonic() + max(0, budget_ms) / 1000.0
        for processed in range(limit):
            if self._stop_event.is_set() or not self._transcribe_queue:
                return
            self._transcribe_utterance(self._transcribe_queue.popleft())
            if processed >= 1 and time.monotonic() >= deadline:
                return

    def _enqueue_utterance(self, utterance):
        if not utterance.partial:
            self._transcribe_queue = deque(
                item for item in self._transcribe_queue
                if item.phrase_id != utterance.phrase_id)
            self._transcribe_queue.appendleft(utterance)
            return
        last = next(
            (item for item in reversed(self._transcribe_queue)
             if item.partial and item.phrase_id == utterance.phrase_id),
            None)
        if (last is not None and last.audio.size == utterance.audio.size and
                np.array_equal(last.audio, utterance.audio)):
            return
        while len(self._transcribe_queue) >= 6:
            for index, item in enumerate(self._transcribe_queue):
                if item.partial:
                    del self._transcribe_queue[index]
                    break
            else:
                self._transcribe_queue.popleft()
        self._transcribe_queue.append(utterance)

    def _correct_text(self, text):
        text = ' '.join(str(text or '').split()).strip()
        if not text or not self.autocorrect:
            return text
        try:
            return correct_speech_text(text) or text
        except Exception:
            return text

    def _update_partial(self, phrase_id, text):
        if phrase_id != self._active_phrase_id:
            self._active_phrase_id = phrase_id
            self._partial_text = ''
        if not text:
            return
        self._partial_text = merge_partial_text(
            self._partial_text, text)[:500]
        with self._lock:
            data = dict(self.latest_data)
            data['live_text'] = self._correct_text(self._partial_text)
            data['last_recognition_ms'] = self._last_recognition_ms
            self.latest_data = data

    def _finish_phrase(self, phrase_id, text, elapsed_ms=0):
        if phrase_id == self._active_phrase_id and not text and self._partial_text:
            text = self._partial_text
        final = self._correct_text(text)
        self._partial_text = ''
        self._active_phrase_id = None
        with self._lock:
            data = dict(self.latest_data)
            data['live_text'] = ''
            data['last_recognition_ms'] = elapsed_ms
            self.latest_data = data
        if not final or final == self._last_text:
            return
        self._last_text = final

        timestamp = datetime.now().strftime('%H:%M:%S')
        with self._lock:
            self._transcript_parts.append(f'[{timestamp}] {final}')
            joined = ' '.join(self._transcript_parts)
            while len(joined) > 12000 and len(self._transcript_parts) > 1:
                self._transcript_parts.pop(0)
                joined = ' '.join(self._transcript_parts)
            data = dict(self.latest_data)
            data['transcribed_text'] = joined
            self.latest_data = data
        self._log(f'Распознано за {elapsed_ms / 1000.0:.2f} с: {final}')

    def _transcribe_utterance(self, utterance):
        if not self.transcriber.loaded:
            try:
                self.transcriber.load()
            except Exception:
                return
        try:
            text = self.transcriber.transcribe(
                utterance.audio, self.rate, language=self.language,
                autocorrect=False)
        except Exception as exc:
            self._log(f'Ошибка распознавания речи: {exc}')
            return

        text = ' '.join(str(text).split()).strip()
        elapsed_ms = max(0, int((time.time() - utterance.captured_at) * 1000))
        self._last_recognition_ms = elapsed_ms
        if utterance.partial:
            self._update_partial(utterance.phrase_id, text)
        else:
            self._finish_phrase(utterance.phrase_id, text, elapsed_ms)

    # --------------------------------------------------------------- state
    def get_analysis(self):
        with self._lock:
            if not self.latest_data:
                return None
            data = dict(self.latest_data)
            data['frequency_bands'] = list(data.get('frequency_bands', []))
            data['classifications'] = list(data.get('classifications', []))
            return data

    def stop(self):
        was_running = self.running
        self.running = False
        self._stop_event.set()

        stream = self.stream
        self.stream = None
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass

        current = threading.current_thread()
        # Поток чтения сам закрывает recorder в своём finally — так soundcard
        # не приходится освобождать объект из чужого потока.
        reader = self._reader_thread
        if reader is not None and reader is not current and reader.is_alive():
            reader.join(timeout=3.0)

        # Если поток не успел завершиться, закрываем устройство принудительно.
        if reader is not None and reader is not current and reader.is_alive():
            recorder = self._recorder
            self._recorder = None
            if recorder is not None:
                try:
                    recorder.__exit__(None, None, None)
                except Exception:
                    pass
            reader.join(timeout=1.0)

        worker = self._worker_thread
        if worker is not None and worker is not current and worker.is_alive():
            worker.join(timeout=0.5)

        self._reader_thread = None
        self._worker_thread = None
        self._soundcard_ready.set()

        if was_running:
            self._log('Захват аудио остановлен')

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass
