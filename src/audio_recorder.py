"""Запись микрофона во время озвучивания.

Запись необязательна: если пакет ``sounddevice``/PortAudio или микрофон
недоступны, сеанс всё равно сохраняется, а в поле ``recording`` попадает
понятная ошибка. Это важно для портативных запусков и тестов.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
import wave
from pathlib import Path

import numpy as np

from src.session_log import default_recordings_dir


class RecordingError(RuntimeError):
    pass


class AudioRecorder:
    def __init__(self, output_dir=None, samplerate=48000, channels=1,
                 max_seconds=600.0):
        self.output_dir = Path(output_dir or default_recordings_dir())
        self.samplerate = int(samplerate or 48000)
        self.channels = max(1, int(channels or 1))
        self.max_seconds = float(max_seconds)
        self.path = None
        self.error = None
        self._stream = None
        self._frames = []
        self._frame_count = 0
        self._lock = threading.Lock()
        self._started_at = None
        self._stopped_at = None
        self._status = ''

    @property
    def active(self):
        return self._stream is not None

    def start(self):
        if self._stream is not None:
            return self.info()
        try:
            import sounddevice as sd
        except Exception as exc:
            self.error = f'sounddevice недоступен: {exc}'
            return self.info(status='unavailable')

        self.error = None
        self._frames = []
        self._frame_count = 0
        self._status = ''
        self._started_at = time.monotonic()

        def callback(indata, frames, time_info, status):
            if status:
                self._status = str(status)
            with self._lock:
                # Ограничиваем память даже при сбое остановки.
                count = int(indata.shape[0])
                if self._frame_count < self.max_seconds * self.samplerate:
                    self._frames.append(
                        np.asarray(indata, dtype=np.float32).copy())
                    self._frame_count += count

        try:
            stream = sd.InputStream(
                samplerate=self.samplerate,
                channels=self.channels,
                dtype='float32',
                callback=callback,
            )
            stream.start()
        except Exception as exc:
            self.error = f'не удалось открыть микрофон: {exc}'
            try:
                stream.close()
            except Exception:
                pass
            return self.info(status='error')

        self._stream = stream
        return self.info(status='recording')

    def stop(self):
        stream = self._stream
        self._stream = None
        self._stopped_at = time.monotonic()
        if stream is not None:
            try:
                stream.stop()
            except Exception as exc:
                self.error = self.error or f'ошибка остановки записи: {exc}'
            try:
                stream.close()
            except Exception:
                pass

        with self._lock:
            frames = self._frames
            self._frames = []

        if not frames:
            if self.error:
                return self.info(status='error')
            return self.info(status='empty')

        try:
            audio = np.concatenate(frames, axis=0)
        except Exception as exc:
            self.error = f'ошибка объединения аудиоблоков: {exc}'
            return self.info(status='error')

        if audio.ndim > 1:
            audio = audio[:, :self.channels]
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1.0:
            audio = audio / peak

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = str(
            self.output_dir / f'mic-{time.strftime("%Y%m%d-%H%M%S")}-'
            f'{uuid.uuid4().hex[:8]}.wav')
        try:
            samples = np.clip(audio, -1.0, 1.0)
            samples = (samples * 32767.0).astype(np.int16)
            with wave.open(self.path, 'wb') as handle:
                handle.setnchannels(int(samples.shape[1]) if samples.ndim > 1 else 1)
                handle.setsampwidth(2)
                handle.setframerate(self.samplerate)
                handle.writeframes(samples.tobytes())
        except Exception as exc:
            self.error = f'не удалось сохранить запись: {exc}'
            self.path = None
            return self.info(status='error')

        return self.info(status='ok')

    def info(self, status=None):
        duration = 0.0
        if self._started_at is not None:
            end = self._stopped_at if self._stopped_at else time.monotonic()
            duration = max(0.0, float(end - self._started_at))
        result = {
            'status': status or (
                'recording' if self._stream is not None else 'idle'),
            'path': self.path,
            'duration_seconds': round(duration, 3),
            'sample_rate': self.samplerate,
            'channels': self.channels,
            'error': self.error or '',
        }
        if self._status:
            result['device_status'] = self._status
        if self.path and os.path.exists(self.path):
            try:
                result['size_bytes'] = os.path.getsize(self.path)
            except OSError:
                pass
        return result

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False
