"""Чтение произвольного текста вслух с фиксацией результата.

Сервис не зависит от PyQt: его можно запускать в фоновом потоке GUI,
из командной строки и в тестах. Он умеет:

* синтезировать речь в файл (если TTS-движок это поддерживает);
* проиграть готовый файл;
* параллельно записать микрофон;
* вернуть единый результат для журнала сеансов.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path

from src.session_log import default_session_dir


def _artifact(path, status='ok', error='', duration=None, **extra):
    result = {
        'path': str(path) if path else None,
        'status': status,
        'error': error or '',
    }
    if path and os.path.exists(str(path)):
        try:
            result['size_bytes'] = os.path.getsize(str(path))
            result['format'] = os.path.splitext(str(path))[1].lstrip('.').lower()
        except OSError:
            pass
    if duration is not None:
        result['duration_seconds'] = round(float(duration), 3)
    for key, value in extra.items():
        if value is not None:
            result[key] = value
    return result


def _is_temp_artifact(path):
    """Проверяет, что файл лежит во временном каталоге (TEMP или ASCII-кэш)."""
    try:
        resolved = Path(path).resolve()
    except (OSError, RuntimeError):
        return False
    roots = {
        Path(tempfile.gettempdir()),
        Path(os.environ.get('TEMP') or '/tmp'),
    }
    try:
        from src.tts_engine import ascii_work_dir
        roots.add(Path(ascii_work_dir()))
    except Exception:
        pass
    for root in roots:
        try:
            resolved.relative_to(root.resolve())
            return True
        except (OSError, ValueError):
            continue
    return False


def _copy_artifact(path, destination_dir, prefix='tts'):
    if not path or not os.path.isfile(str(path)):
        return None
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    suffix = os.path.splitext(str(path))[1] or '.wav'
    target = destination_dir / (
        f'{prefix}-{time.strftime("%Y%m%d-%H%M%S")}-'
        f'{uuid.uuid4().hex[:8]}{suffix}')
    try:
        shutil.copy2(str(path), str(target))
        return target
    except OSError:
        return None


class ReadAloudService:
    """Выполняет один сеанс чтения и возвращает отчёт."""

    def __init__(self, tts, recorder_factory=None, data_dir=None,
                 clock=None):
        self.tts = tts
        self.recorder_factory = recorder_factory
        self.data_dir = Path(data_dir or default_session_dir())
        self.clock = clock or time.monotonic

    def _new_recorder(self):
        if self.recorder_factory is not None:
            return self.recorder_factory()
        try:
            from src.audio_recorder import AudioRecorder
            return AudioRecorder(output_dir=self.data_dir / 'recordings')
        except Exception:
            return None

    def read(self, text, voice=None, *, language='ru', record=False,
             save_tts=True, session_dir=None):
        text = (text or '').strip()
        session_dir = Path(session_dir or self.data_dir)
        started_wall = datetime.now().astimezone().isoformat(timespec='seconds')
        started = self.clock()
        tts_audio = None
        recording = None
        error = ''
        status = 'ok'
        recorder = None
        synthesized_path = None

        if not text:
            return {
                'status': 'error',
                'error': 'пустой текст',
                'started_at': started_wall,
                'duration_seconds': 0.0,
                'text': '',
                'voice': voice,
                'language': language,
                'tts_audio': None,
                'recording': None,
            }

        if self.tts is None:
            return {
                'status': 'error',
                'error': 'TTS недоступен',
                'started_at': started_wall,
                'duration_seconds': 0.0,
                'text': text,
                'voice': voice,
                'language': language,
                'tts_audio': None,
                'recording': None,
            }

        try:
            # 1. Синтез. Для Piper/Edge получаем готовый файл, который
            #    можно сохранить и проиграть без повторного синтеза.
            if hasattr(self.tts, 'synthesize'):
                temp_dir = session_dir / 'tts'
                temp_dir.mkdir(parents=True, exist_ok=True)
                synth = self.tts.synthesize(text, voice)
                if not isinstance(synth, dict):
                    synth = {'path': synth}
                synthesized_path = synth.get('path')
                if synth.get('status') == 'error':
                    # Движок может не поддерживать файловый вывод. Не
                    # сдаёмся сразу: ниже попробуем прямое озвучивание.
                    error = str(synth.get('error') or 'ошибка синтеза речи')
                    synthesized_path = None
                elif not synthesized_path or not os.path.isfile(str(synthesized_path)):
                    # Некоторые движки (pyttsx3) синтезируют только при
                    # проигрывании. Переходим к прямому озвучиванию.
                    synthesized_path = None

            # 2. Необязательная запись микрофона начинается до проигрывания.
            if record:
                recorder = self._new_recorder()
                if recorder is not None:
                    try:
                        recorder.start()
                    except Exception as exc:
                        recording = {
                            'status': 'error',
                            'error': f'ошибка запуска записи: {exc}',
                        }

            # 3. Проигрывание.
            if synthesized_path:
                if hasattr(self.tts, 'play_file'):
                    played = self.tts.play_file(synthesized_path)
                else:
                    played = True
                if not played:
                    status = 'stopped'
                if save_tts:
                    saved = _copy_artifact(
                        synthesized_path, session_dir / 'tts', prefix='tts')
                    tts_audio = _artifact(saved, status='saved')
                else:
                    tts_audio = _artifact(
                        None, status='not_saved')
            else:
                # Прямое озвучивание (pyttsx3/SAPI): блокирует до конца.
                result = self.tts.speak(text, voice)
                if isinstance(result, dict):
                    if result.get('status') == 'error':
                        raise RuntimeError(
                            result.get('error') or 'ошибка озвучивания')
                    if result.get('status') == 'stopped':
                        status = 'stopped'
                tts_audio = _artifact(None, status='not_saved')

        except Exception as exc:
            status = 'error'
            error = str(exc)
        finally:
            if recorder is not None:
                try:
                    recording = recorder.stop()
                except Exception as exc:
                    recording = {
                        'status': 'error',
                        'error': f'ошибка остановки записи: {exc}',
                    }
            # Временный файл синтеза не является пользовательской записью:
            # при сохранении уже есть копия, иначе он больше не нужен.
            saved_copy = bool(tts_audio and tts_audio.get('path'))
            if synthesized_path and (not save_tts or saved_copy):
                try:
                    if _is_temp_artifact(synthesized_path):
                        os.unlink(synthesized_path)
                except OSError:
                    pass

        elapsed = max(0.0, float(self.clock() - started))
        if status == 'ok':
            error = ''
        return {
            'status': status,
            'error': error,
            'started_at': started_wall,
            'duration_seconds': round(elapsed, 3),
            'text': text,
            'voice': voice,
            'language': language,
            'tts_audio': tts_audio,
            'recording': recording,
        }
