"""Журнал сеансов чтения вслух.

Каждый сеанс сохраняется отдельной строкой JSON (JSONL) вместе с текстом,
голосом, устройствами, языком, координатами, состоянием аудио и путями к
записям. Формат JSONL выбран специально: файл можно читать построчно,
дописывать без перезаписи и легко экспортировать в CSV/JSON.

Каталог данных по умолчанию — ``data`` рядом с проектом. Его можно
переопределить переменной окружения ``BLUETOOTH_MONITOR_DATA_DIR``.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path


SESSION_VERSION = 1


def default_data_dir():
    """Каталог для журналов/записей проекта."""
    custom = os.environ.get('BLUETOOTH_MONITOR_DATA_DIR')
    if custom:
        return Path(custom).expanduser()
    return Path(__file__).resolve().parent.parent / 'data'


def default_session_dir(data_dir=None):
    return Path(data_dir or default_data_dir()) / 'read_aloud'


def default_sessions_path(data_dir=None):
    return default_session_dir(data_dir) / 'sessions.jsonl'


def default_recordings_dir(data_dir=None):
    return default_session_dir(data_dir) / 'recordings'


def _now():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')


def _clean_string(value, limit=4000):
    if value is None:
        return ''
    text = str(value)
    if len(text) > limit:
        text = text[:limit]
    return text


def _clean_location(location):
    if not isinstance(location, dict):
        return None
    result = {}
    for key in ('lat', 'lon', 'accuracy', 'source', 'city', 'country'):
        value = location.get(key)
        if value is None or value == '':
            continue
        result[key] = value
    if 'lat' not in result or 'lon' not in result:
        return None
    try:
        result['lat'] = round(float(result['lat']), 6)
        result['lon'] = round(float(result['lon']), 6)
    except (TypeError, ValueError):
        return None
    if not (-90.0 <= result['lat'] <= 90.0 and
            -180.0 <= result['lon'] <= 180.0):
        return None
    if result['lat'] == 0.0 and result['lon'] == 0.0:
        return None
    return result


def _clean_voice(voice):
    if isinstance(voice, dict):
        return {
            'id': _clean_string(voice.get('id'), 300),
            'name': _clean_string(voice.get('name') or voice.get('voice'), 300),
            'backend': _clean_string(voice.get('backend'), 50),
            'language': _clean_string(voice.get('language'), 20),
        }
    return {
        'id': _clean_string(voice, 300),
        'name': _clean_string(voice, 300),
        'backend': '',
        'language': '',
    }


def build_session_record(
    *,
    text,
    recognized_text='',
    voice=None,
    language='ru',
    location=None,
    microphone='',
    speaker='',
    audio_status='spoken',
    recording=None,
    tts_audio=None,
    started_at=None,
    duration_seconds=None,
    error='',
    source='custom',
):
    """Собирает запись сеанса в едином формате.

    ``recording`` и ``tts_audio`` — словари, возвращённые рекордером и
    TTS-движком. В них сохраняются путь, длительность, частота и статус.
    """
    record = {
        'version': SESSION_VERSION,
        'id': uuid.uuid4().hex,
        'created_at': _now(),
        'started_at': _clean_string(started_at, 60) or None,
        'source': _clean_string(source, 40),
        'text': _clean_string(text, 200000),
        'recognized_text': _clean_string(recognized_text, 200000),
        'voice': _clean_voice(voice),
        'language': _clean_string(language or 'auto', 30),
        'location': _clean_location(location),
        'devices': {
            'microphone': _clean_string(microphone, 500),
            'speaker': _clean_string(speaker, 500),
        },
        'audio': {
            'status': _clean_string(audio_status, 30),
            'duration_seconds': (
                round(float(duration_seconds), 3)
                if duration_seconds is not None else None
            ),
            'error': _clean_string(error, 2000),
        },
        'recording': _clean_artifact(recording),
        'tts_audio': _clean_artifact(tts_audio),
    }
    return record


def _clean_artifact(value):
    if not isinstance(value, dict):
        return None
    result = {}
    for key in (
        'path', 'status', 'error', 'format', 'sample_rate', 'channels',
        'duration_seconds', 'size_bytes',
    ):
        if key not in value or value.get(key) is None:
            continue
        item = value.get(key)
        if key == 'path':
            item = _clean_string(item, 2000)
        elif key in ('sample_rate', 'channels', 'size_bytes'):
            try:
                item = int(item)
            except (TypeError, ValueError):
                continue
        elif key == 'duration_seconds':
            try:
                item = round(float(item), 3)
            except (TypeError, ValueError):
                continue
        else:
            item = _clean_string(item, 500)
        result[key] = item
    return result or None


class SessionStore:
    """Потокобезопасное хранилище сеансов (JSONL)."""

    def __init__(self, path=None):
        self.path = Path(path or default_sessions_path())
        self._lock = threading.RLock()

    def append(self, record):
        """Дописывает запись, назначая id/created_at при необходимости."""
        if not isinstance(record, dict):
            raise ValueError('Запись сеанса должна быть словарём')
        data = dict(record)
        data.setdefault('version', SESSION_VERSION)
        data.setdefault('id', uuid.uuid4().hex)
        data.setdefault('created_at', _now())
        data.setdefault('text', '')
        line = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, 'a', encoding='utf-8') as handle:
                handle.write(line + '\n')
                handle.flush()
                try:
                    os.fsync(handle.fileno())
                except OSError:
                    pass
        return data

    def load(self, limit=None):
        """Читает записи, пропуская повреждённые строки."""
        records = []
        with self._lock:
            try:
                with open(self.path, 'r', encoding='utf-8') as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                        except Exception:
                            continue
                        if isinstance(item, dict):
                            records.append(item)
            except FileNotFoundError:
                return []
        if limit is not None:
            return records[-int(limit):]
        return records

    def last(self):
        records = self.load(limit=1)
        return records[0] if records else None

    def count(self):
        return len(self.load())

    def export_csv(self, destination):
        """Экспортирует журнал в CSV с плоскими основными полями."""
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        records = self.load()
        fields = [
            'id', 'created_at', 'source', 'text', 'recognized_text',
            'voice_name', 'voice_backend', 'language', 'latitude', 'longitude',
            'location_source', 'microphone', 'speaker', 'audio_status',
            'duration_seconds', 'recording_path', 'tts_audio_path', 'error',
        ]
        with open(destination, 'w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for record in records:
                voice = record.get('voice') or {}
                devices = record.get('devices') or {}
                location = record.get('location') or {}
                audio = record.get('audio') or {}
                recording = record.get('recording') or {}
                tts_audio = record.get('tts_audio') or {}
                writer.writerow({
                    'id': record.get('id', ''),
                    'created_at': record.get('created_at', ''),
                    'source': record.get('source', ''),
                    'text': record.get('text', ''),
                    'recognized_text': record.get('recognized_text', ''),
                    'voice_name': voice.get('name', ''),
                    'voice_backend': voice.get('backend', ''),
                    'language': record.get('language', ''),
                    'latitude': location.get('lat', ''),
                    'longitude': location.get('lon', ''),
                    'location_source': location.get('source', ''),
                    'microphone': devices.get('microphone', ''),
                    'speaker': devices.get('speaker', ''),
                    'audio_status': audio.get('status', ''),
                    'duration_seconds': audio.get('duration_seconds', ''),
                    'recording_path': recording.get('path', ''),
                    'tts_audio_path': tts_audio.get('path', ''),
                    'error': audio.get('error', ''),
                })
        return destination

    def export_json(self, destination):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {'version': SESSION_VERSION, 'sessions': self.load()}
        fd, temp_name = tempfile.mkstemp(
            prefix=destination.name + '.', suffix='.tmp',
            dir=str(destination.parent))
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write('\n')
            os.replace(temp_name, destination)
        finally:
            try:
                if os.path.exists(temp_name):
                    os.unlink(temp_name)
            except OSError:
                pass
        return destination
