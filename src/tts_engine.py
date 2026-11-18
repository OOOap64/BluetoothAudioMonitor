"""
Text-to-Speech module for Bluetooth Audio Monitor.

Поддерживаются три источника голоса:

* ``pyttsx3`` — офлайн-голоса Windows/SAPI;
* ``edge-tts`` — нейроголоса Microsoft Edge (нужен интернет);
* ``piper`` — локальная модель Piper (``.onnx``), в том числе обученная
  на собственном голосе.

Программа Piper и русские голоса могут лежать в каталоге ``piper`` рядом с
проектом — они находятся автоматически. Пользовательские профили голосов
хранятся в :mod:`src.voice_profiles` и передаются в ``speak``/``speak_async``
как словарь.
"""

from __future__ import annotations

import asyncio
import ctypes
import hashlib
import importlib.util
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import wave
from pathlib import Path


def app_root():
    """Каталог проекта (рядом с ``main.py``)."""
    return Path(__file__).resolve().parent.parent


def piper_search_dirs():
    """Каталоги, где ищется установленный Piper."""
    dirs = []
    custom = os.environ.get('BLUETOOTH_MONITOR_PIPER_DIR')
    if custom:
        dirs.append(Path(custom).expanduser())
    dirs.append(app_root() / 'piper')
    dirs.append(app_root() / 'tools' / 'piper')
    return dirs


def find_piper_executable():
    """Возвращает путь к ``piper.exe`` или пустую строку."""
    custom = os.environ.get('BLUETOOTH_MONITOR_PIPER_EXE')
    if custom and Path(custom).is_file():
        return str(Path(custom))
    for directory in piper_search_dirs():
        for name in ('piper.exe', 'piper'):
            candidate = directory / name
            if candidate.is_file():
                return str(candidate)
    return shutil.which('piper') or shutil.which('piper.exe') or ''


def piper_model_dirs():
    """Каталоги с моделями Piper (``.onnx``)."""
    dirs = []
    custom = os.environ.get('BLUETOOTH_MONITOR_PIPER_VOICES')
    if custom:
        dirs.append(Path(custom).expanduser())
    for directory in piper_search_dirs():
        dirs.append(directory / 'voices')
        dirs.append(directory / 'models')
    return dirs


def _pretty_piper_name(stem):
    """``ru_RU-irina-medium`` -> ``Irina (ru-RU, medium)``."""
    parts = str(stem).split('-')
    if len(parts) >= 3 and parts[0]:
        locale = parts[0].replace('_', '-')
        name = parts[1][:1].upper() + parts[1][1:]
        quality = parts[2]
        return f'{name} (Piper, {locale}, {quality})'
    return f'{stem} (Piper)'


def find_piper_models():
    """Находит модели Piper в стандартных каталогах."""
    models = []
    seen = set()
    for directory in piper_model_dirs():
        try:
            if not directory.is_dir():
                continue
            candidates = sorted(directory.glob('*.onnx'))
        except OSError:
            continue
        for onnx in candidates:
            try:
                resolved = str(onnx.resolve())
            except OSError:
                resolved = str(onnx)
            if resolved in seen:
                continue
            seen.add(resolved)
            config = onnx.with_suffix('.onnx.json')
            models.append({
                'model_path': str(onnx),
                'config_path': str(config) if config.is_file() else '',
                'name': _pretty_piper_name(onnx.stem),
            })
    return models


# ---------------------------------------------------------------------------
# Обходной путь для Piper и не-ASCII путей.
#
# Piper 1.2.0 (espeak-ng + onnxruntime) на Windows аварийно завершается
# (0xC0000409), если путь к piper.exe, модели, конфигурации или выходному WAV
# содержит символы вне ASCII. Типичный случай — русское имя пользователя
# (C:\Users\Пользователь\...). Поэтому для Piper создаётся ASCII-зеркало
# каталога (junction) и все файлы передаются через него.
# ---------------------------------------------------------------------------

def _is_ascii_path(path):
    try:
        str(path).encode('ascii')
        return True
    except (UnicodeEncodeError, TypeError):
        return False


def _is_writable_dir(directory):
    try:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / f'.probe-{uuid.uuid4().hex}'
        probe.write_bytes(b'0')
        probe.unlink()
        return True
    except OSError:
        return False


_ASCII_WORK_DIR = None
_ASCII_WORK_LOCK = threading.Lock()


def ascii_work_dir():
    """Возвращает записываемый каталог с ASCII-путём."""
    global _ASCII_WORK_DIR
    with _ASCII_WORK_LOCK:
        if _ASCII_WORK_DIR is not None:
            return _ASCII_WORK_DIR
        candidates = []
        custom = os.environ.get('BLUETOOTH_MONITOR_ASCII_DIR')
        if custom:
            candidates.append(Path(custom).expanduser())
        if os.name == 'nt':
            candidates.append(
                Path(os.environ.get('PUBLIC') or r'C:\Users\Public')
                / 'BluetoothAudioMonitor')
            candidates.append(
                Path(os.environ.get('PROGRAMDATA') or r'C:\ProgramData')
                / 'BluetoothAudioMonitor')
        candidates.append(Path(tempfile.gettempdir()) / 'BluetoothAudioMonitor')
        for directory in candidates:
            if _is_ascii_path(directory) and _is_writable_dir(directory):
                _ASCII_WORK_DIR = directory
                return directory
        _ASCII_WORK_DIR = Path(tempfile.gettempdir())
        return _ASCII_WORK_DIR


def ascii_temp_path(prefix, suffix):
    """Временный файл с ASCII-путём (нужен Piper/MCI на Windows)."""
    directory = ascii_work_dir() / 'tmp'
    directory.mkdir(parents=True, exist_ok=True)
    return str(directory / f'{prefix}-{uuid.uuid4().hex}{suffix}')


def _remove_link(path):
    try:
        path = Path(path)
        if getattr(path, 'is_junction', lambda: False)():
            os.rmdir(path)
        elif path.is_symlink():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    except OSError:
        pass


def _ascii_mirror(source_dir):
    """Создаёт junction на каталог, если его путь не ASCII."""
    source = Path(source_dir).resolve()
    if _is_ascii_path(source):
        return source
    if os.name != 'nt':
        return None
    root = ascii_work_dir() / 'mirrors'
    if not _is_writable_dir(root):
        return None
    key = hashlib.sha1(str(source).lower().encode('utf-8')).hexdigest()[:12]
    # Имя ссылки тоже должно быть ASCII, иначе смысл обходного пути теряется.
    safe_name = ''.join(
        ch if ch.isascii() and (ch.isalnum() or ch in '._-') else '_'
        for ch in (source.name or 'dir'))
    link = root / f'{safe_name or "dir"}-{key}'
    if link.exists() or getattr(link, 'is_junction', lambda: False)():
        try:
            if os.path.samefile(link, source):
                return link
        except OSError:
            pass
        _remove_link(link)
    try:
        result = subprocess.run(
            ['cmd', '/c', 'mklink', '/J', str(link), str(source)],
            capture_output=True, text=True,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
        )
        if result.returncode == 0 and os.path.isdir(link):
            return link
    except Exception as exc:
        print(f"Cannot create ASCII junction for Piper: {exc}")
    return None


def _copy_piper_runtime(source_dir, target_dir):
    """Копия piper.exe, DLL и espeak-ng-data в ASCII-каталог."""
    source = Path(source_dir)
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.name == 'voices':
            continue
        destination = target / item.name
        try:
            if item.is_dir():
                shutil.copytree(item, destination, dirs_exist_ok=True)
            elif item.is_file():
                shutil.copy2(item, destination)
        except OSError:
            continue


def _ascii_piper_dir(executable):
    """Каталог с piper.exe, гарантированно пригодный для запуска."""
    source = Path(executable).resolve().parent
    if _is_ascii_path(source):
        return source
    mirror = _ascii_mirror(source)
    if mirror is not None:
        return mirror
    key = hashlib.sha1(str(source).lower().encode('utf-8')).hexdigest()[:12]
    target = ascii_work_dir() / 'piper' / key
    if not (target / Path(executable).name).is_file():
        _copy_piper_runtime(source, target)
    if (target / Path(executable).name).is_file():
        return target
    return source


def _ascii_file(path, source_root, ascii_root):
    """Возвращает ASCII-путь к файлу; при необходимости копирует его."""
    original = Path(path)
    if _is_ascii_path(original):
        return str(original)
    try:
        relative = original.resolve().relative_to(Path(source_root).resolve())
    except (OSError, ValueError):
        relative = None
    if relative is not None:
        candidate = Path(ascii_root) / relative
        if _is_ascii_path(candidate):
            if candidate.is_file():
                return str(candidate)
            try:
                candidate.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, candidate)
                return str(candidate)
            except OSError:
                pass
    try:
        stat = original.stat()
        key = hashlib.sha1(
            f'{original.resolve()}|{stat.st_size}|{stat.st_mtime_ns}'.encode(
                'utf-8')).hexdigest()[:16]
    except OSError:
        return str(original)
    target = ascii_work_dir() / 'files' / f'{key}{original.suffix}'
    if not target.is_file():
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target)
        except OSError:
            return str(original)
    return str(target)


def _sanitized_piper_config(config_path):
    """Убирает из конфигурации Piper фонемы из нескольких символов.

    Piper 1.2.0 умеет сопоставлять только односимвольные фонемы, а некоторые
    новые голоса (например, ru_RU-denis-medium) содержат в phoneme_id_map
    дифтонги вроде ``aɪ``. Без этой чистки Piper падает с ошибкой
    "is not a single codepoint". Для русской речи такие фонемы не нужны.
    """
    path = Path(config_path)
    if not path.is_file():
        return ''
    try:
        with path.open(encoding='utf-8') as handle:
            config = json.load(handle)
    except Exception:
        return str(path)
    phoneme_map = config.get('phoneme_id_map')
    if not isinstance(phoneme_map, dict):
        return str(path)
    unsupported = [key for key in phoneme_map if len(str(key)) != 1]
    if not unsupported:
        return str(path)
    for key in unsupported:
        phoneme_map.pop(key, None)
    try:
        stat = path.stat()
        key = hashlib.sha1(
            f'{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}'.encode(
                'utf-8')).hexdigest()[:16]
        target = ascii_work_dir() / 'configs' / f'{key}.json'
        if not target.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('w', encoding='utf-8') as handle:
                json.dump(config, handle, ensure_ascii=False)
        return str(target)
    except OSError:
        return str(path)


class TTSEngine:
    def __init__(self, engine_type='auto'):
        self.engine_type = engine_type
        self.engine = None
        self.edge_tts = None
        self.lock = threading.Lock()
        self._stop_flag = False
        self._current_thread = None
        self._current_process = None
        self._mci_alias = 'bamtts'
        self._mci_open = False
        # Очередь не даёт сотням вызовов speak_async создать сотни потоков
        # и наложиться друг на друга. Один рабочий поток читает по порядку.
        self._queue = queue.Queue()
        self._queue_lock = threading.Lock()
        self._worker_thread = None
        self._generation = 0
        self._generation_lock = threading.Lock()
        self.last_result = None
        # SAPI/pyttsx3 на Windows нельзя вызывать из фонового потока:
        # runAndWait() там блокируется и молча не даёт звука. Операции с
        # системным голосом выполняются в потоке, где создан движок (обычно
        # главный поток GUI). Остальные голоса работают в отдельном потоке.
        self._main_thread_id = threading.get_ident()
        self._dispatcher = None
        self._init_engine()

    def set_main_thread_dispatcher(self, dispatcher):
        """Регистрирует функцию, выполняющую callable в главном потоке GUI."""
        self._dispatcher = dispatcher

    def _run_on_main_thread(self, func):
        if (threading.get_ident() == self._main_thread_id or
                self._dispatcher is None or os.name != 'nt'):
            return func()
        result = {}
        done = threading.Event()

        def task():
            try:
                result['value'] = func()
            except Exception as exc:
                result['error'] = exc
            finally:
                done.set()

        self._dispatcher(task)
        if not done.wait(120):
            raise RuntimeError('Не удалось выполнить озвучивание в главном потоке')
        if 'error' in result:
            raise result['error']
        return result['value']

    # ------------------------------------------------------------- init
    def _init_engine(self):
        if self.engine_type in ['auto', 'pyttsx3']:
            try:
                import pyttsx3
                self.engine = pyttsx3.init()
                self.engine_type = 'pyttsx3'
                voices = self.engine.getProperty('voices') or []
                selected = None
                for voice in voices:
                    name = str(getattr(voice, 'name', '') or '')
                    voice_id = str(getattr(voice, 'id', '') or '')
                    if 'russian' in name.lower() or 'ru' in voice_id.lower():
                        selected = voice_id
                        break
                if selected:
                    self.engine.setProperty('voice', selected)
                elif voices:
                    self.engine.setProperty(
                        'voice', getattr(voices[0], 'id', ''))
                self.engine.setProperty('rate', 165)
                self.engine.setProperty('volume', 1.0)
            except Exception as exc:
                print(f"pyttsx3 init failed: {exc}")
                self.engine = None

        try:
            if importlib.util.find_spec('edge_tts') is not None:
                import edge_tts
                self.edge_tts = edge_tts
                if self.engine is None and self.engine_type in ['auto', 'edge-tts']:
                    self.engine_type = 'edge-tts'
        except Exception as exc:
            print(f"edge-tts init failed: {exc}")
            self.edge_tts = None

        # Последний офлайн-вариант для Windows.
        if self.engine is None and self.engine_type in ['auto', 'sapi']:
            try:
                import win32com.client
                self.engine = win32com.client.Dispatch("SAPI.SpVoice")
                self.engine_type = 'sapi'
            except Exception:
                pass

        if (self.engine is None and self.edge_tts is None
                and not find_piper_executable()):
            self.engine_type = None

    # ---------------------------------------------------------- helpers
    def _ensure_pyttsx3(self):
        if self.engine_type == 'pyttsx3' and self.engine is not None:
            return True
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine_type = 'pyttsx3'
            return True
        except Exception as exc:
            print(f"pyttsx3 unavailable: {exc}")
            return False

    def _ensure_edge(self):
        if self.edge_tts is not None:
            return True
        try:
            if importlib.util.find_spec('edge_tts') is None:
                return False
            import edge_tts
            self.edge_tts = edge_tts
            return True
        except Exception as exc:
            print(f"edge-tts unavailable: {exc}")
            return False

    @staticmethod
    def _split_voice_id(voice):
        """Разбирает строковый id вида ``backend::voice``."""
        if not isinstance(voice, str):
            return None, voice
        if '::' in voice:
            backend, value = voice.split('::', 1)
            return backend, value
        return None, voice

    def _resolve_voice(self, voice):
        """Возвращает ``(backend, profile)`` для запрошенного голоса."""
        profile = {}
        if isinstance(voice, dict):
            profile = dict(voice)
            backend = str(profile.get('backend') or '').lower()
            if not backend:
                backend = self.engine_type or 'pyttsx3'
        else:
            backend, value = self._split_voice_id(voice)
            if backend is None:
                # Совместимость со старым API, где передавался просто id.
                text = str(value or '')
                if text.startswith('edge-') or 'Neural' in text:
                    backend = 'edge'
                elif self.engine_type in ('pyttsx3', 'sapi'):
                    backend = self.engine_type
                elif self.edge_tts is not None:
                    backend = 'edge'
                else:
                    backend = 'pyttsx3'
            profile['voice'] = str(value or '')
            if not profile['voice']:
                for key in ('voice_id', 'id'):
                    if isinstance(voice, dict) and voice.get(key):
                        profile['voice'] = str(voice[key])
                        break
        if backend in ('sapi', 'system', 'windows'):
            backend = 'sapi'
        if backend in ('edge-tts', 'edgetts', 'edge_tts'):
            backend = 'edge'
        if backend in ('piper-tts', 'piper_tts', 'custom'):
            backend = 'piper'
        return backend, profile

    # ---------------------------------------------------------- speaking
    def speak(self, text, voice=None, reset_stop=True):
        """Синтез и проигрывание. Возвращает отчёт о результате.

        ``reset_stop=False`` используется рабочим потоком очереди: команда
        «Стоп» не должна отменяться самим фактом начала следующего элемента.
        """
        if not text or not text.strip():
            result = {'status': 'skipped', 'error': 'пустой текст'}
            self.last_result = result
            return result
        if reset_stop:
            self._stop_flag = False

        backend, profile = self._resolve_voice(voice)
        with self.lock:
            if self._stop_flag:
                result = {'status': 'stopped', 'error': ''}
                self.last_result = result
                return result
            try:
                if backend == 'piper':
                    self._speak_piper(text, profile)
                elif backend == 'edge':
                    self._speak_edge(text, profile)
                elif backend == 'sapi':
                    self._speak_sapi(text, profile)
                else:
                    self._speak_pyttsx3(text, profile)
                result = {
                    'status': 'stopped' if self._stop_flag else 'ok',
                    'error': '',
                    'backend': backend,
                }
            except Exception as exc:
                print(f"TTS error: {exc}")
                result = {
                    'status': 'error',
                    'error': str(exc),
                    'backend': backend,
                }
        self.last_result = result
        return result

    def _speak_pyttsx3(self, text, profile):
        def speak_now():
            if not self._ensure_pyttsx3():
                raise RuntimeError('pyttsx3 недоступен')
            voice_id = str(profile.get('voice') or '').strip()
            if voice_id:
                try:
                    self.engine.setProperty('voice', voice_id)
                except Exception as exc:
                    print(f"Cannot set voice {voice_id!r}: {exc}")
            try:
                self.engine.setProperty(
                    'rate', int(profile.get('rate') or 165))
                self.engine.setProperty(
                    'volume', float(profile.get('volume') or 1.0))
            except Exception:
                pass
            self.engine.say(text)
            self.engine.runAndWait()

        self._run_on_main_thread(speak_now)

    def _speak_sapi(self, text, profile):
        def speak_now():
            if self.engine is None:
                raise RuntimeError('SAPI недоступен')
            voice_name = str(profile.get('voice') or '').strip()
            if voice_name and voice_name != 'default':
                try:
                    for token in self.engine.GetVoices():
                        description = str(token.GetDescription() or '')
                        if voice_name.lower() in description.lower():
                            self.engine.Voice = token
                            break
                except Exception as exc:
                    print(f"Cannot select SAPI voice: {exc}")
            self.engine.Speak(text)

        self._run_on_main_thread(speak_now)

    def _speak_edge(self, text, profile):
        if not self._ensure_edge():
            raise RuntimeError('edge-tts недоступен (установите edge-tts)')
        voice_id = (
            str(profile.get('voice') or '').strip() or
            'ru-RU-DmitryNeural'
        )
        temp_file = ascii_temp_path('bluetooth-monitor-tts', '.mp3')

        async def _save():
            communicate = self.edge_tts.Communicate(text, voice_id)
            await communicate.save(temp_file)

        try:
            try:
                asyncio.run(_save())
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(_save())
                finally:
                    loop.close()
            if not self._stop_flag:
                played = self._play_media_file(temp_file)
                if played:
                    self._safe_unlink(temp_file)
        finally:
            # Если файл открыт внешним плеером, оставляем его плееру.
            pass

    def _piper_command_for(self, profile, output_path):
        """Собирает команду Piper с ASCII-путями и рабочим каталогом."""
        model_path = str(profile.get('model_path') or '').strip()
        if not model_path:
            raise RuntimeError('Не указана модель Piper (.onnx)')
        if not os.path.isfile(model_path):
            raise RuntimeError(f'Файл модели Piper не найден: {model_path}')

        executable = str(profile.get('executable_path') or '').strip()
        if not executable:
            executable = find_piper_executable()
        if not executable:
            raise RuntimeError(
                'Не найден Piper. Установите piper.exe в папку piper проекта '
                'или укажите путь в настройках голоса.')

        source_root = Path(executable).resolve().parent
        ascii_root = _ascii_piper_dir(executable)
        ascii_executable = str(Path(ascii_root) / Path(executable).name)
        if not _is_ascii_path(ascii_executable):
            raise RuntimeError(
                'Piper не может работать с путём, содержащим не-ASCII символы. '
                'Задайте ASCII-каталог в BLUETOOTH_MONITOR_ASCII_DIR.')

        ascii_model = _ascii_file(model_path, source_root, ascii_root)
        if not _is_ascii_path(ascii_model):
            raise RuntimeError(
                'Не удалось подготовить ASCII-путь к модели Piper. '
                'Скопируйте piper в каталог с латинскими буквами.')

        config_path = str(profile.get('config_path') or '').strip()
        if not config_path:
            candidate = model_path + '.json'
            if os.path.isfile(candidate):
                config_path = candidate
        ascii_config = ''
        if config_path and os.path.isfile(config_path):
            ascii_config = _ascii_file(config_path, source_root, ascii_root)
            ascii_config = _sanitized_piper_config(ascii_config)

        command = self._build_piper_command(
            ascii_executable,
            ascii_model,
            config_path=ascii_config or None,
            output_path=output_path,
        )
        return command, str(ascii_root)

    @staticmethod
    def _piper_failure_message(returncode, stderr):
        details = (stderr or '').strip()[:500]
        if returncode in (3221226505, -1073740791, -1073741819):
            hint = (
                'Piper аварийно завершился. Обычно это несовместимый голос '
                'или путь с не-ASCII символами. Попробуйте другой голос '
                '(например, Irina/Dmitri) или задайте BLUETOOTH_MONITOR_ASCII_DIR.')
            return hint + (f' Детали: {details}' if details else '')
        return 'Piper завершился с ошибкой: ' + (details or 'код %s' % returncode)

    def _speak_piper(self, text, profile):
        output_path = ascii_temp_path('bluetooth-monitor-piper', '.wav')
        command, cwd = self._piper_command_for(profile, output_path)
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                cwd=cwd or None,
            )
            self._current_process = process
            try:
                _, stderr = process.communicate(text, timeout=180)
            finally:
                self._current_process = None
            if self._stop_flag:
                return
            if process.returncode != 0:
                raise RuntimeError(
                    self._piper_failure_message(process.returncode, stderr))
            if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
                raise RuntimeError('Piper не создал аудиофайл')
            played = self._play_media_file(output_path)
            if played:
                self._safe_unlink(output_path)
        finally:
            if self._stop_flag:
                self._safe_unlink(output_path)

    def _synthesize_piper(self, text, profile, output_path):
        command, cwd = self._piper_command_for(profile, output_path)
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            cwd=cwd or None,
        )
        self._current_process = process
        try:
            _, stderr = process.communicate(text, timeout=180)
        finally:
            self._current_process = None
        if self._stop_flag:
            raise RuntimeError('озвучивание остановлено')
        if process.returncode != 0:
            raise RuntimeError(
                self._piper_failure_message(process.returncode, stderr))

    def _synthesize_edge(self, text, profile, output_path):
        if not self._ensure_edge():
            raise RuntimeError('edge-tts недоступен (установите edge-tts)')
        voice_id = (
            str(profile.get('voice') or '').strip() or
            'ru-RU-DmitryNeural'
        )

        async def _save():
            communicate = self.edge_tts.Communicate(text, voice_id)
            await communicate.save(output_path)

        try:
            asyncio.run(_save())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(_save())
            finally:
                loop.close()
        if self._stop_flag:
            raise RuntimeError('озвучивание остановлено')

    def _synthesize_pyttsx3(self, text, profile, output_path):
        def synth_now():
            if not self._ensure_pyttsx3():
                raise RuntimeError('pyttsx3 недоступен')
            voice_id = str(profile.get('voice') or '').strip()
            if voice_id:
                try:
                    self.engine.setProperty('voice', voice_id)
                except Exception as exc:
                    print(f"Cannot set voice {voice_id!r}: {exc}")
            try:
                self.engine.setProperty('rate', int(profile.get('rate') or 165))
                self.engine.setProperty(
                    'volume', float(profile.get('volume') or 1.0))
            except Exception:
                pass
            # save_to_file поддерживается SAPI5/espeak. Если движок не умеет —
            # выбрасываем исключение, и сервис перейдёт к прямому speak().
            self.engine.save_to_file(text, str(output_path))
            self.engine.runAndWait()
            if (not os.path.isfile(str(output_path)) or
                    os.path.getsize(str(output_path)) == 0):
                raise RuntimeError('движок не создал аудиофайл')

        self._run_on_main_thread(synth_now)

    def synthesize(self, text, voice=None):
        """Создаёт аудиофайл без проигрывания.

        Возвращает словарь ``{'status', 'path', 'backend', 'format', 'error'}``.
        Для движков без файлового вывода возвращается ``status='unsupported'``;
        вызывающий код должен использовать :meth:`speak`.
        """
        text = (text or '').strip()
        if not text:
            return {'status': 'error', 'error': 'пустой текст', 'path': None}
        # Новый запрос синтеза снимает предыдущую команду «Стоп».
        self._stop_flag = False

        backend, profile = self._resolve_voice(voice)
        suffix = '.mp3' if backend == 'edge' else '.wav'
        output_path = ascii_temp_path('bluetooth-monitor-synth', suffix)
        try:
            if backend == 'piper':
                self._synthesize_piper(text, profile, output_path)
            elif backend == 'edge':
                self._synthesize_edge(text, profile, output_path)
            elif backend == 'sapi':
                return {
                    'status': 'unsupported',
                    'error': 'SAPI не поддерживает файловый вывод',
                    'backend': backend,
                    'path': None,
                }
            else:
                self._synthesize_pyttsx3(text, profile, output_path)

            if self._stop_flag:
                self._safe_unlink(output_path)
                return {'status': 'stopped', 'error': '', 'backend': backend,
                        'path': None}
            if (not os.path.isfile(output_path) or
                    os.path.getsize(output_path) == 0):
                return {
                    'status': 'error',
                    'error': 'движок не создал аудиофайл',
                    'backend': backend,
                    'path': None,
                }
            return {
                'status': 'ok',
                'error': '',
                'backend': backend,
                'path': output_path,
                'format': suffix.lstrip('.'),
                'size_bytes': os.path.getsize(output_path),
            }
        except Exception as exc:
            self._safe_unlink(output_path)
            return {
                'status': 'error',
                'error': str(exc),
                'backend': backend,
                'path': None,
            }

    def play_file(self, path):
        """Проигрывает готовый WAV/MP3. Возвращает False, если остановлено."""
        if not path or not os.path.isfile(str(path)):
            return False
        if self._stop_flag:
            return False
        try:
            return bool(self._play_media_file(str(path)))
        except Exception as exc:
            print(f"Cannot play TTS audio: {exc}")
            return False

    @staticmethod
    def _build_piper_command(executable, model_path, config_path=None,
                             output_path=None):
        """Собирает команду Piper; вынесено отдельно для тестов."""
        executable = str(executable or '').strip()
        if executable.lower().endswith('.py'):
            command = [sys.executable, executable]
        else:
            command = [executable]
        command.extend(['--model', str(model_path)])
        if config_path:
            command.extend(['--config', str(config_path)])
        if output_path:
            command.extend(['--output_file', str(output_path)])
        return command

    # -------------------------------------------------------- playback
    def _play_media_file(self, path):
        """Проигрывает файл; возвращает True, если его можно удалять."""
        if self._stop_flag:
            return False

        # Windows MCI умеет и WAV, и MP3 (Edge TTS), поэтому пробуем первым.
        if os.name == 'nt' and self._play_with_mci(path):
            return True

        try:
            import pygame
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy() and not self._stop_flag:
                time.sleep(0.05)
            pygame.mixer.music.stop()
            pygame.mixer.quit()
            return True
        except Exception:
            pass

        if str(path).lower().endswith('.wav'):
            try:
                import winsound
                duration = 0.0
                try:
                    with wave.open(str(path), 'rb') as handle:
                        duration = (
                            handle.getnframes() /
                            float(handle.getframerate() or 1)
                        )
                except Exception:
                    duration = 0.0
                winsound.PlaySound(
                    str(path),
                    winsound.SND_FILENAME | winsound.SND_ASYNC)
                deadline = time.time() + max(0.2, duration)
                while time.time() < deadline and not self._stop_flag:
                    time.sleep(0.05)
                winsound.PlaySound(None, 0)
                return True
            except Exception:
                pass

        try:
            if hasattr(os, 'startfile'):
                os.startfile(str(path))
                # Внешний плеер ещё может читать файл.
                return False
        except Exception as exc:
            print(f"Cannot play TTS audio: {exc}")
        return False

    def _play_with_mci(self, path):
        """Проигрывание WAV/MP3 через Windows MCI (без сторонних пакетов)."""
        if os.name != 'nt':
            return False
        try:
            winmm = ctypes.WinDLL('winmm')
        except Exception:
            return False

        buffer = ctypes.create_unicode_buffer(512)

        def send(command):
            try:
                return winmm.mciSendStringW(
                    command, buffer, len(buffer), None)
            except Exception:
                return -1

        alias = self._mci_alias
        send(f'close {alias}')
        opened = send(f'open "{path}" type mpegvideo alias {alias}')
        if opened != 0:
            opened = send(f'open "{path}" alias {alias}')
        if opened != 0:
            return False
        self._mci_open = True
        try:
            send(f'set {alias} time format milliseconds')
            if send(f'play {alias}') != 0:
                return False
            while not self._stop_flag:
                if send(f'status {alias} mode') != 0:
                    break
                if (buffer.value or '').lower() in ('stopped', ''):
                    break
                time.sleep(0.04)
        finally:
            send(f'close {alias}')
            self._mci_open = False
        return True

    @staticmethod
    def _safe_unlink(path):
        try:
            if path and os.path.exists(path):
                os.unlink(path)
        except Exception:
            pass

    def _ensure_worker(self):
        with self._queue_lock:
            if (self._worker_thread is None or
                    not self._worker_thread.is_alive()):
                worker = threading.Thread(
                    target=self._queue_worker, name='tts-queue',
                    daemon=True)
                self._worker_thread = worker
                worker.start()
            return self._worker_thread

    def _queue_worker(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            generation, text, voice, callback = item
            with self._generation_lock:
                current = self._generation
            if generation != current:
                result = {'status': 'stopped', 'error': 'отменено'}
            else:
                result = self.speak(text, voice, reset_stop=False)
            if callback is not None:
                try:
                    callback(result)
                except Exception:
                    pass

    def speak_async(self, text, voice=None, on_done=None):
        """Ставит текст в очередь одного рабочего потока.

        В отличие от прежней реализации, сотни последовательных вызовов не
        создают сотни потоков и не пытаются говорить одновременно. Это
        устраняет гонки pyttsx3/Piper и утечку потоков при стресс-тестах.
        """
        self._stop_flag = False
        with self._generation_lock:
            generation = self._generation
        worker = self._ensure_worker()
        self._queue.put((generation, text, voice, on_done))
        self._current_thread = worker
        return worker

    def stop(self):
        # Новая генерация отменяет всё, что уже стоит в очереди.
        with self._generation_lock:
            self._generation += 1
        self._stop_flag = True

        cancelled = []
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is not None:
                cancelled.append(item)
        for _generation, _text, _voice, callback in cancelled:
            if callback is not None:
                try:
                    callback({'status': 'stopped', 'error': 'отменено'})
                except Exception:
                    pass

        process = self._current_process
        if process is not None:
            try:
                process.terminate()
            except Exception:
                pass
        if self._mci_open:
            try:
                winmm = ctypes.WinDLL('winmm')
                buffer = ctypes.create_unicode_buffer(64)
                winmm.mciSendStringW(
                    f'stop {self._mci_alias}', buffer, len(buffer), None)
                winmm.mciSendStringW(
                    f'close {self._mci_alias}', buffer, len(buffer), None)
            except Exception:
                pass
        try:
            if self.engine_type == 'pyttsx3' and self.engine:
                self.engine.stop()
        except Exception:
            pass
        try:
            import pygame
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
        except Exception:
            pass

    # ----------------------------------------------------------- voices
    def get_system_voices(self):
        """Системные голоса pyttsx3 (для диалога импорта)."""
        voices = []
        if self._ensure_pyttsx3():
            try:
                for voice in self.engine.getProperty('voices') or []:
                    voice_id = str(getattr(voice, 'id', '') or '')
                    name = str(getattr(voice, 'name', '') or voice_id)
                    voices.append({'id': voice_id, 'name': name})
            except Exception:
                pass
        return voices

    def get_piper_voices(self):
        """Готовые локальные голоса Piper, найденные на диске."""
        executable = find_piper_executable()
        voices = []
        for model in find_piper_models():
            voices.append({
                'id': 'piper::' + os.path.basename(model['model_path']),
                'name': model['name'],
                'backend': 'piper',
                'voice': '',
                'model_path': model['model_path'],
                'config_path': model['config_path'],
                'executable_path': executable,
                'rate': 165,
                'volume': 1.0,
            })
        return voices

    def get_available_voices(self):
        """Возвращает встроенные голоса с указанием backend.

        Piper-голоса из папки проекта добавляются автоматически, поэтому
        импорт локального голоса сводится к выбору из списка.
        Пользовательские профили добавляет интерфейс из ``VoiceProfileStore``.
        """
        voices = []

        for voice in self.get_system_voices():
            voices.append({
                'id': 'pyttsx3::' + voice['id'],
                'name': voice['name'] + ' (системный)',
                'backend': 'pyttsx3',
                'voice': voice['id'],
            })

        if self.edge_tts is not None or importlib.util.find_spec('edge_tts') is not None:
            edge_voices = [
                ('ru-RU-DmitryNeural', 'Дмитрий (Edge, мужской)'),
                ('ru-RU-SvetlanaNeural', 'Светлана (Edge, женский)'),
                ('en-US-AriaNeural', 'Aria (Edge, English)'),
                ('en-US-GuyNeural', 'Guy (Edge, English)'),
            ]
            for voice_id, name in edge_voices:
                voices.append({
                    'id': 'edge::' + voice_id,
                    'name': name,
                    'backend': 'edge',
                    'voice': voice_id,
                })

        voices.extend(self.get_piper_voices())

        if self.engine_type == 'sapi' and not voices:
            voices.append({
                'id': 'sapi::default',
                'name': 'Системный голос (SAPI)',
                'backend': 'sapi',
                'voice': 'default',
            })

        if not voices:
            voices.append({
                'id': 'default',
                'name': 'По умолчанию',
                'backend': 'pyttsx3',
                'voice': '',
            })
        return voices


_tts_instance = None


def get_tts():
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = TTSEngine()
    return _tts_instance
