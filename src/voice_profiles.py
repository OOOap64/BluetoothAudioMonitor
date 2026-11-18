"""Хранилище пользовательских голосов для озвучивания.

Пользователь может добавить собственный голос одним из способов:

* ``pyttsx3`` — любой голос, установленный в системе (SAPI/Windows);
* ``edge`` — голос Edge TTS по его идентификатору (например,
  ``ru-RU-DmitryNeural``);
* ``piper`` — локальная модель Piper (``.onnx`` + ``.json``), в том числе
  обученная на собственном голосе.

Профили сохраняются в JSON в каталоге пользователя, поэтому не теряются
между запусками. Модуль намеренно не импортирует PyQt и не запускает
озвучивание: его можно использовать в тестах и в фоновых потоках.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
import uuid
from pathlib import Path


BACKENDS = ('pyttsx3', 'edge', 'piper')
_BACKEND_ALIASES = {
    'sapi': 'pyttsx3',
    'system': 'pyttsx3',
    'windows': 'pyttsx3',
    'pyttsx': 'pyttsx3',
    'edge-tts': 'edge',
    'edgetts': 'edge',
    'edge_tts': 'edge',
    'piper-tts': 'piper',
    'piper_tts': 'piper',
    'custom': 'piper',
}


def default_store_path():
    """Возвращает путь к файлу профилей голосов.

    Переменная окружения ``BLUETOOTH_MONITOR_VOICE_DIR`` позволяет хранить
    профили рядом с проектом или на переносимом носителе.
    """
    custom_dir = os.environ.get('BLUETOOTH_MONITOR_VOICE_DIR')
    if custom_dir:
        return str(Path(custom_dir).expanduser() / 'voices.json')

    if os.name == 'nt':
        base = os.environ.get('APPDATA') or str(Path.home() / 'AppData' / 'Roaming')
    else:
        base = os.environ.get('XDG_CONFIG_HOME') or str(Path.home() / '.config')
    return str(Path(base) / 'BluetoothAudioMonitor' / 'voices.json')


def _clean_text(value, limit=200):
    if value is None:
        return ''
    text = str(value).strip()
    if len(text) > limit:
        text = text[:limit].rstrip()
    return text


def _clean_number(value, default, minimum, maximum):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:  # NaN
        return default
    return max(minimum, min(maximum, number))


def normalize_profile(profile):
    """Проверяет и приводит профиль голоса к единому виду.

    Возвращает новый словарь. При некорректных данных выбрасывает
    ``ValueError`` с понятным сообщением.
    """
    if not isinstance(profile, dict):
        raise ValueError('Профиль голоса должен быть словарём')

    name = _clean_text(profile.get('name'))
    if not name:
        raise ValueError('Укажите название голоса')

    backend = _clean_text(profile.get('backend') or 'pyttsx3').lower()
    backend = _BACKEND_ALIASES.get(backend, backend)
    if backend not in BACKENDS:
        raise ValueError(
            f'Неизвестный тип голоса: {profile.get("backend")!r}. '
            f'Допустимо: {", ".join(BACKENDS)}')

    result = {
        'id': _clean_text(profile.get('id'), limit=100),
        'name': name,
        'backend': backend,
        'voice': _clean_text(
            profile.get('voice') or profile.get('voice_id'), limit=300),
        'rate': int(_clean_number(profile.get('rate'), 165, 50, 400)),
        'volume': round(
            _clean_number(profile.get('volume'), 1.0, 0.0, 1.0), 3),
        'model_path': _clean_text(
            profile.get('model_path') or profile.get('model'), limit=1000),
        'config_path': _clean_text(
            profile.get('config_path') or profile.get('config'), limit=1000),
        'executable_path': _clean_text(
            profile.get('executable_path') or profile.get('executable'),
            limit=1000),
    }

    if backend == 'pyttsx3' and not result['voice']:
        raise ValueError('Выберите системный голос')
    if backend == 'edge' and not result['voice']:
        # Edge TTS умеет голос по умолчанию, но лучше сохранить явный.
        result['voice'] = 'ru-RU-DmitryNeural'
    if backend == 'piper' and not result['model_path']:
        raise ValueError('Укажите файл модели Piper (.onnx)')

    # Пути не обязаны существовать на момент загрузки: файл могли
    # перенести. Проверка существования выполняется перед озвучиванием.
    return result


def _safe_id(profile_id):
    text = _clean_text(profile_id, limit=100)
    return text or ('custom-' + uuid.uuid4().hex[:12])


class VoiceProfileStore:
    """Список пользовательских голосов с сохранением в JSON."""

    def __init__(self, path=None, autoload=True):
        self.path = Path(path or default_store_path())
        self._profiles = []
        if autoload:
            self.load()

    @property
    def profiles(self):
        return copy.deepcopy(self._profiles)

    def load(self):
        self._profiles = []
        try:
            with open(self.path, 'r', encoding='utf-8') as handle:
                data = json.load(handle)
        except FileNotFoundError:
            return self
        except Exception:
            # Повреждённый файл не должен ломать запуск приложения.
            return self

        if isinstance(data, dict):
            data = data.get('voices', [])
        if not isinstance(data, list):
            return self

        seen_ids = set()
        for item in data:
            try:
                profile = normalize_profile(item)
            except Exception:
                continue
            if profile['id'] in seen_ids:
                profile['id'] = ''
            if not profile['id']:
                profile['id'] = self._new_id(seen_ids)
            seen_ids.add(profile['id'])
            self._profiles.append(profile)
        return self

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {'version': 1, 'voices': self._profiles}
        # Атомарная запись: не оставляем полупустой файл при сбое.
        handle = tempfile.NamedTemporaryFile(
            'w', encoding='utf-8', delete=False,
            dir=str(self.path.parent), suffix='.tmp')
        try:
            with handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write('\n')
            os.replace(handle.name, self.path)
        finally:
            try:
                if os.path.exists(handle.name):
                    os.unlink(handle.name)
            except Exception:
                pass
        return self

    def _new_id(self, existing=None):
        existing = existing if existing is not None else {
            profile['id'] for profile in self._profiles}
        while True:
            candidate = 'custom-' + uuid.uuid4().hex[:12]
            if candidate not in existing:
                return candidate

    def add(self, profile):
        """Добавляет профиль; одноимённый пользовательский голос заменяется."""
        data = normalize_profile(profile)
        data['id'] = _safe_id(data.get('id'))
        while any(item['id'] == data['id'] for item in self._profiles):
            data['id'] = self._new_id()
        # Одинаковые названия сбивают с толку в выпадающем списке.
        self._profiles = [
            item for item in self._profiles
            if item['name'].casefold() != data['name'].casefold()
        ]
        self._profiles.append(data)
        self.save()
        return copy.deepcopy(data)

    def replace(self, profile_id, profile):
        self.remove(profile_id, save=False)
        data = normalize_profile(profile)
        data['id'] = _safe_id(profile_id)
        self._profiles.append(data)
        self.save()
        return copy.deepcopy(data)

    def remove(self, profile_id, save=True):
        before = len(self._profiles)
        self._profiles = [
            item for item in self._profiles if item['id'] != profile_id]
        changed = len(self._profiles) != before
        if changed and save:
            self.save()
        return changed

    def get(self, profile_id):
        for item in self._profiles:
            if item['id'] == profile_id:
                return copy.deepcopy(item)
        return None

    def find_by_name(self, name):
        target = _clean_text(name).casefold()
        for item in self._profiles:
            if item['name'].casefold() == target:
                return copy.deepcopy(item)
        return None

    def __len__(self):
        return len(self._profiles)

    def __iter__(self):
        return iter(self.profiles)
