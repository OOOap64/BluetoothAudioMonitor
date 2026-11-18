#!/usr/bin/env python3
"""Стресс-тест чтения вслух.

Скрипт много раз прогоняет полный цикл: синтез -> сохранение аудио ->
проигрывание -> (необязательно) запись микрофона -> запись сеанса.
Первый сбой печатается с трассировкой, после чего тест останавливается
(если не указан ``--keep-going``).

Примеры::

    python tools/stress_read_aloud.py --iterations 500
    python tools/stress_read_aloud.py --iterations 100000 --backend fake
    python tools/stress_read_aloud.py --iterations 100 --fail-at 42
    python tools/stress_read_aloud.py --iterations 20 --backend tts --voice "Piper"

По умолчанию используется детерминированный fake-движок: тест проверяет
логику очереди, сохранения и отчётности без реального аудиоустройства.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
import traceback
import wave
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.read_aloud import ReadAloudService  # noqa: E402
from src.session_log import (  # noqa: E402
    SessionStore, build_session_record, default_sessions_path,
)


class FakeTTS:
    """Быстрый TTS, создающий настоящий WAV и не требующий звуковой карты."""

    def __init__(self, fail_at=None, play_delay=0.0):
        self.fail_at = fail_at
        self.play_delay = play_delay
        self.calls = 0
        self.lock = __import__('threading').Lock()
        self.stop_requested = False

    def synthesize(self, text, voice=None):
        with self.lock:
            self.calls += 1
            calls = self.calls
        if self.fail_at is not None and calls == self.fail_at:
            return {'status': 'error', 'error': 'искусственный сбой синтеза',
                    'path': None}
        fd, path = tempfile.mkstemp(suffix='.wav', prefix='stress-tts-')
        os.close(fd)
        with wave.open(path, 'wb') as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(b'\x00\x00' * 160)
        return {'status': 'ok', 'path': path, 'backend': 'fake'}

    def play_file(self, path):
        if self.play_delay:
            time.sleep(self.play_delay)
        return not self.stop_requested

    def speak(self, text, voice=None):
        return {'status': 'ok', 'error': ''}

    def stop(self):
        self.stop_requested = True


class FakeRecorder:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.running = False
        self.path = None

    def start(self):
        self.running = True
        return {'status': 'recording'}

    def stop(self):
        self.running = False
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = str(self.output_dir / f'fake-{time.time_ns()}.wav')
        with wave.open(self.path, 'wb') as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(b'\x00\x00' * 800)
        return {
            'status': 'ok',
            'path': self.path,
            'duration_seconds': 0.05,
            'sample_rate': 16000,
            'channels': 1,
        }


def run_iteration(index, args, tts, store, temp_dir):
    iteration_dir = Path(temp_dir) / f'run-{index:06d}'
    recorder_factory = None
    if args.record:
        recorder_factory = lambda: FakeRecorder(iteration_dir / 'recordings')
    service = ReadAloudService(
        tts, recorder_factory=recorder_factory,
        data_dir=iteration_dir / 'read_aloud')
    result = service.read(
        args.text,
        voice={'id': 'fake', 'name': 'Fake voice', 'backend': 'fake'},
        language=args.language,
        record=args.record,
        save_tts=args.save_tts,
    )
    record = build_session_record(
        text=result['text'],
        recognized_text=args.recognized_text,
        voice=result['voice'],
        language=result['language'],
        location={'lat': 55.75, 'lon': 37.62, 'source': 'stress-test'},
        microphone='Fake microphone',
        speaker='Fake speaker',
        audio_status=result['status'],
        recording=result.get('recording'),
        tts_audio=result.get('tts_audio'),
        started_at=result.get('started_at'),
        duration_seconds=result.get('duration_seconds'),
        error=result.get('error', ''),
    )
    store.append(record)
    if result['status'] != 'ok':
        raise RuntimeError(result.get('error') or 'неизвестная ошибка озвучивания')
    if args.record:
        recording = result.get('recording') or {}
        if recording.get('status') != 'ok':
            raise RuntimeError(
                'запись микрофона не удалась: ' + str(recording.get('error')))
    audio = result.get('tts_audio') or {}
    if args.save_tts and not audio.get('path'):
        raise RuntimeError('аудио TTS не сохранено')
    return record


def build_parser():
    parser = argparse.ArgumentParser(description='Стресс-тест озвучивания')
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--backend', choices=('fake', 'tts'), default='fake')
    parser.add_argument('--text', default='Проверка озвучивания номер раз.')
    parser.add_argument('--recognized-text', default='Тест распознавания')
    parser.add_argument('--language', default='ru')
    parser.add_argument('--record', action='store_true',
                        help='включить fake-запись микрофона')
    parser.add_argument('--no-save-tts', dest='save_tts',
                        action='store_false')
    parser.add_argument('--keep-going', action='store_true',
                        help='продолжать после ошибок')
    parser.add_argument('--fail-at', type=int, default=None,
                        help='искусственно сломать указанную итерацию')
    parser.add_argument('--play-delay', type=float, default=0.0)
    parser.add_argument('--data-dir', default=None,
                        help='каталог для журнала (по умолчанию temp)')
    parser.add_argument('--voice', default=None,
                        help='id/JSON голоса для --backend tts')
    parser.add_argument('--verbose', action='store_true')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.iterations < 1:
        print('--iterations должно быть >= 1', file=sys.stderr)
        return 2

    temp_dir = tempfile.mkdtemp(prefix='bam-stress-')
    data_dir = Path(args.data_dir) if args.data_dir else (
        Path(temp_dir) / 'read_aloud')
    store = SessionStore(data_dir / 'sessions.jsonl')

    if args.backend == 'tts':
        try:
            from src.tts_engine import get_tts
            tts = get_tts()
            if tts is None:
                raise RuntimeError('TTS не инициализирован')
        except Exception as exc:
            print(f'Не удалось запустить реальный TTS: {exc}', file=sys.stderr)
            return 2
    else:
        tts = FakeTTS(fail_at=args.fail_at, play_delay=args.play_delay)

    failures = 0
    attempted = 0
    started = time.monotonic()
    for index in range(1, args.iterations + 1):
        attempted = index
        try:
            if args.backend == 'tts':
                # Для реального движка сервис использует тот же код.
                from src.read_aloud import ReadAloudService
                service = ReadAloudService(
                    tts, data_dir=data_dir / f'run-{index:06d}')
                result = service.read(
                    args.text, args.voice, language=args.language,
                    record=args.record, save_tts=args.save_tts)
                if result['status'] != 'ok':
                    raise RuntimeError(
                        result.get('error') or result['status'])
            else:
                run_iteration(index, args, tts, store, temp_dir)
        except Exception as exc:
            failures += 1
            print(f'\nОШИБКА на итерации {index}: {exc}', file=sys.stderr)
            traceback.print_exc()
            if not args.keep_going:
                break
        if args.verbose and index % 100 == 0:
            elapsed = max(1e-9, time.monotonic() - started)
            print(f'  {index}/{args.iterations} '
                  f'({index / elapsed:.1f} итераций/с)', flush=True)

    elapsed = max(1e-9, time.monotonic() - started)
    print()
    print(f'Итераций: {attempted} из {args.iterations}')
    print(f'Успешно:  {attempted - failures}')
    print(f'Ошибок:   {failures}')
    print(f'Время:    {elapsed:.2f} с ({attempted / elapsed:.1f} итер./с)')
    print(f'Журнал:   {store.path} (записей: {store.count()})')
    print(f'Временный каталог: {temp_dir}')
    return 0 if failures == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
