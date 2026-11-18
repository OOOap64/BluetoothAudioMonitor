"""Определение координат места записи.

Источники координат (по приоритету):

1. явно указанные пользователем широта/долгота;
2. переменные окружения ``BLUETOOTH_MONITOR_LATITUDE`` /
   ``BLUETOOTH_MONITOR_LONGITUDE``;
3. сохранённый кеш (``data/location.json``);
4. необязательный запрос к IP-геолокации.

Сетевой запрос никогда не выполняется автоматически при сохранении сеанса:
его запускает отдельная кнопка в интерфейсе. Поэтому сохранение результатов
не зависает без интернета.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path

from src.session_log import default_data_dir


CACHE_TTL_SECONDS = 24 * 60 * 60
_PROVIDERS = (
    'https://ipapi.co/json/',
    'http://ip-api.com/json/?fields=status,lat,lon,city,country,query',
)


def cache_path(data_dir=None):
    return Path(data_dir or default_data_dir()) / 'location.json'


def _valid_coordinates(lat, lon):
    try:
        lat = float(lat)
        lon = float(lon)
    except (TypeError, ValueError):
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None
    if lat == 0.0 and lon == 0.0:
        # Часто это означает «координаты не определены».
        return None
    return round(lat, 6), round(lon, 6)


def _from_environment():
    lat = os.environ.get('BLUETOOTH_MONITOR_LATITUDE')
    lon = os.environ.get('BLUETOOTH_MONITOR_LONGITUDE')
    coords = _valid_coordinates(lat, lon)
    if not coords:
        return None
    return {
        'lat': coords[0],
        'lon': coords[1],
        'source': 'environment',
        'city': os.environ.get('BLUETOOTH_MONITOR_CITY', ''),
        'country': os.environ.get('BLUETOOTH_MONITOR_COUNTRY', ''),
    }


def manual_location(lat, lon, source='manual', **extra):
    coords = _valid_coordinates(lat, lon)
    if not coords:
        return None
    location = {'lat': coords[0], 'lon': coords[1], 'source': source}
    for key in ('city', 'country', 'accuracy'):
        value = extra.get(key)
        if value not in (None, ''):
            location[key] = value
    return location


def load_cached(data_dir=None, max_age=CACHE_TTL_SECONDS):
    """Читает кеш координат. Возвращает None, если он устарел."""
    path = cache_path(data_dir)
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            data = json.load(handle)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    timestamp = data.get('timestamp')
    if max_age is not None and timestamp:
        try:
            if time.time() - float(timestamp) > max_age:
                return None
        except (TypeError, ValueError):
            pass
    coords = _valid_coordinates(data.get('lat'), data.get('lon'))
    if not coords:
        return None
    return {
        'lat': coords[0],
        'lon': coords[1],
        'source': data.get('source') or 'cache',
        'city': data.get('city') or '',
        'country': data.get('country') or '',
        'timestamp': data.get('timestamp'),
    }


def save_cached(location, data_dir=None):
    if not location:
        return None
    path = cache_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(location)
    payload.setdefault('timestamp', time.time())
    try:
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
    except OSError:
        return None
    return payload


def cached_location(data_dir=None):
    """Локальные координаты без сети: env -> кеш."""
    return _from_environment() or load_cached(data_dir)


def _query_provider(url, timeout):
    request = urllib.request.Request(
        url,
        headers={
            'User-Agent': 'BluetoothAudioMonitor/1.0',
            'Accept': 'application/json',
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read(65536).decode('utf-8', errors='replace')
    return json.loads(payload)


def detect_location(timeout=3.0, data_dir=None):
    """Определяет координаты по IP. Может блокировать до ``timeout`` секунд.

    Возвращает словарь или None. Не выбрасывает исключения — вызывающий код
    должен показать понятное сообщение.
    """
    environment = _from_environment()
    if environment:
        return environment

    for url in _PROVIDERS:
        try:
            data = _query_provider(url, timeout)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        if data.get('status') not in (None, 'success'):
            continue
        coords = _valid_coordinates(data.get('lat'), data.get('lon'))
        if not coords:
            continue
        location = {
            'lat': coords[0],
            'lon': coords[1],
            'source': 'ip:' + url.split('/')[2],
            'city': str(data.get('city') or ''),
            'country': str(data.get('country') or data.get('country_name') or ''),
            'timestamp': time.time(),
        }
        save_cached(location, data_dir)
        return location

    # Последний шанс — старый кеш.
    return load_cached(data_dir, max_age=None)
