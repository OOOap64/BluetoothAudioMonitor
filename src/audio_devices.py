"""Список доступных источников аудио.

Основной путь для системного звука на Windows — WASAPI loopback через
пакет ``soundcard``. Он не требует изменения ``sounddevice.default`` и
работает независимо от версии PortAudio. Обычные микрофоны/линейные
входы по-прежнему открываются через ``sounddevice``.
"""

from __future__ import annotations

import ctypes
import os

import sounddevice as sd

try:  # soundcard нужен только для loopback, без него приложение работает.
    import soundcard as sc
except Exception:  # pragma: no cover - зависит от окружения
    sc = None


COINIT_MULTITHREADED = 0x0
_RPC_E_CHANGED_MODE = 0x80010106


def com_initialize():
    """Инициализирует COM в текущем потоке для soundcard/MediaFoundation.

    COM инициализируется отдельно для каждого потока. Если sounddevice
    (PortAudio) был импортирован раньше soundcard, фоновые потоки Qt
    остаются без COM, и ``sc.default_speaker()`` падает с ошибкой
    0x800401f0. Поэтому нужные потоки инициализируются явно.
    """
    if os.name != 'nt':
        return False
    try:
        ole32 = ctypes.windll.ole32
    except Exception:
        return False
    try:
        result = int(ole32.CoInitializeEx(None, COINIT_MULTITHREADED))
    except Exception:
        return False
    # S_OK = 0, S_FALSE = 1, RPC_E_CHANGED_MODE — COM уже инициализирован
    # в другом режиме, работать с ним всё равно можно.
    return result in (0, 1, _RPC_E_CHANGED_MODE, _RPC_E_CHANGED_MODE - 2 ** 32)


def _mic_channels(mic):
    try:
        channels = int(getattr(mic, 'channels', 0) or 0)
    except Exception:
        channels = 0
    return max(1, channels)


def _source_from_loopback_mic(mic):
    return {
        'backend': 'soundcard',
        'id': mic.id,
        'name': str(mic.name),
        'channels': _mic_channels(mic),
        'samplerate': 48000,
        'is_loopback': True,
    }


def list_loopback_sources():
    """Возвращает устройства WASAPI loopback (захват системного звука)."""
    if sc is None:
        return []

    sources = []
    seen = set()
    default_id = None
    try:
        default_speaker = sc.default_speaker()
        if default_speaker is not None:
            default_id = default_speaker.id
    except Exception:
        default_speaker = None

    try:
        microphones = sc.all_microphones(include_loopback=True)
    except Exception:
        return []

    for mic in microphones:
        try:
            if not getattr(mic, 'isloopback', False):
                continue
            source = _source_from_loopback_mic(mic)
        except Exception:
            continue
        if source['id'] in seen:
            continue
        seen.add(source['id'])
        source['is_default'] = source['id'] == default_id
        sources.append(source)

    # Устройство по умолчанию — первым в списке.
    sources.sort(key=lambda item: (not item.get('is_default'), item['name'].lower()))
    return sources


def default_loopback_source():
    """Loopback-источник динамика по умолчанию (или None)."""
    if sc is None:
        return None
    try:
        speaker = sc.default_speaker()
        if speaker is None:
            return None
        mic = sc.get_microphone(id=speaker.id, include_loopback=True)
        if mic is None:
            return None
        source = _source_from_loopback_mic(mic)
        source['is_default'] = True
        return source
    except Exception:
        return None


def list_input_sources():
    """Список обычных устройств ввода (микрофоны, линейные входы и т.п.)."""
    sources = []
    try:
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
    except Exception:
        return sources

    try:
        default_input = int(sd.default.device[0])
    except Exception:
        default_input = -1

    for index, device in enumerate(devices):
        try:
            max_input = int(device.get('max_input_channels', 0) or 0)
        except Exception:
            max_input = 0
        if max_input < 1:
            continue

        hostapi_name = ''
        try:
            hostapi_index = int(device.get('hostapi', -1))
            if 0 <= hostapi_index < len(hostapis):
                hostapi_name = str(hostapis[hostapi_index].get('name', ''))
        except Exception:
            pass

        try:
            default_rate = int(round(float(device.get('default_samplerate', 0) or 0)))
        except Exception:
            default_rate = 0

        sources.append({
            'backend': 'sounddevice',
            'device': index,
            'name': str(device.get('name', f'Устройство {index}')),
            'channels': min(2, max_input),
            'samplerate': default_rate,
            'is_loopback': False,
            'hostapi': hostapi_name,
            'is_default': index == default_input,
        })

    return sources


def list_all_sources():
    """Loopback-устройства сверху, затем обычные входы."""
    return list_loopback_sources() + list_input_sources()


def resolve_auto_source():
    """Источник для пункта «Авто».

    Сначала пытаемся захватить системный звук с динамика по умолчанию,
    иначе используем микрофон/вход по умолчанию.
    """
    return default_loopback_source()


def describe_source(source):
    if not source:
        return 'авто (системный звук или микрофон по умолчанию)'
    name = source.get('name', 'неизвестное устройство')
    if source.get('is_loopback'):
        return f'{name} [WASAPI loopback]'
    hostapi = source.get('hostapi')
    return f'{name} [{hostapi}]' if hostapi else name


def soundcard_available():
    return sc is not None


def default_input_name():
    """Имя устройства ввода по умолчанию (микрофона)."""
    if sc is not None:
        try:
            microphone = sc.default_microphone()
            if microphone is not None:
                return str(microphone.name)
        except Exception:
            pass
    try:
        device = sd.query_devices(kind='input')
        return str(device.get('name', '') or '')
    except Exception:
        return ''


def default_output_name():
    """Имя устройства вывода по умолчанию (динамиков/наушников)."""
    if sc is not None:
        try:
            speaker = sc.default_speaker()
            if speaker is not None:
                return str(speaker.name)
        except Exception:
            pass
    try:
        device = sd.query_devices(kind='output')
        return str(device.get('name', '') or '')
    except Exception:
        return ''


def describe_audio_pair(source=None):
    """Возвращает имена микрофона и динамика для журнала сеансов.

    Если выбран loopback-источник, он считается динамиком, а микрофон
    берётся по умолчанию. Если выбран вход — наоборот.
    """
    microphone = ''
    speaker = ''
    if source:
        if source.get('is_loopback'):
            speaker = str(source.get('name') or '')
        else:
            microphone = str(source.get('name') or '')
    if not microphone:
        microphone = default_input_name()
    if not speaker:
        speaker = default_output_name()
    return {'microphone': microphone, 'speaker': speaker}
