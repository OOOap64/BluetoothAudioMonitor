# -*- coding: utf-8 -*-
"""Тесты чтения вслух, сохранения аудио и стресс-устойчивости."""

import os
import sys
import tempfile
import threading
import time
import unittest
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.read_aloud import ReadAloudService
from src.session_log import SessionStore, build_session_record


class FakeTTS:
    def __init__(self, fail_synth=False, fail_speak=False, stopped=False):
        self.fail_synth = fail_synth
        self.fail_speak = fail_speak
        self.stopped = stopped
        self.calls = 0
        self.last_path = None
        self.lock = threading.Lock()

    def synthesize(self, text, voice=None):
        with self.lock:
            self.calls += 1
        if self.fail_synth:
            return {'status': 'error', 'error': 'нет файлового вывода',
                    'path': None}
        fd, path = tempfile.mkstemp(suffix='.wav', prefix='tts-test-')
        os.close(fd)
        with wave.open(path, 'wb') as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(b'\x00\x00' * 80)
        self.last_path = path
        return {'status': 'ok', 'path': path, 'backend': 'fake'}

    def play_file(self, path):
        return not self.stopped

    def speak(self, text, voice=None):
        if self.fail_speak:
            return {'status': 'error', 'error': 'нет голоса'}
        return {'status': 'ok', 'error': ''}


class FakeRecorder:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.path = None
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True
        return {'status': 'recording'}

    def stop(self):
        self.stopped = True
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / 'mic.wav'
        with wave.open(str(self.path), 'wb') as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(b'\x00\x00' * 160)
        return {
            'status': 'ok', 'path': str(self.path),
            'duration_seconds': 0.1, 'sample_rate': 16000, 'channels': 1,
        }


class ReadAloudServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.service = ReadAloudService(
            FakeTTS(), data_dir=self.root / 'read_aloud')

    def tearDown(self):
        self.temp.cleanup()

    def test_read_saves_tts_audio(self):
        result = self.service.read(
            'Привет! Это тест чтения вслух.',
            voice={'name': 'Тестовый голос', 'backend': 'piper'},
            language='ru',
        )
        self.assertEqual(result['status'], 'ok', result)
        self.assertEqual(result['error'], '')
        self.assertIsNotNone(result['tts_audio'])
        self.assertTrue(Path(result['tts_audio']['path']).is_file())
        self.assertEqual(result['tts_audio']['status'], 'saved')
        self.assertGreaterEqual(result['duration_seconds'], 0.0)
        self.assertFalse(os.path.exists(self.service.tts.last_path))

    def test_temp_synthesis_file_is_removed_when_not_saved(self):
        result = self.service.read('Без сохранения', save_tts=False)
        self.assertEqual(result['status'], 'ok', result)
        self.assertFalse(os.path.exists(self.service.tts.last_path))

    def test_record_microphone(self):
        service = ReadAloudService(
            FakeTTS(), recorder_factory=lambda: FakeRecorder(
                self.root / 'recordings'),
            data_dir=self.root / 'read_aloud')
        result = service.read('Запись', record=True)
        self.assertEqual(result['status'], 'ok', result)
        recording = result['recording']
        self.assertEqual(recording['status'], 'ok')
        self.assertTrue(Path(recording['path']).is_file())
        self.assertEqual(recording['sample_rate'], 16000)

    def test_empty_text_is_error_without_calling_tts(self):
        result = self.service.read('   ')
        self.assertEqual(result['status'], 'error')
        self.assertIn('пустой', result['error'])
        self.assertEqual(self.service.tts.calls, 0)

    def test_synthesis_failure_falls_back_to_speak(self):
        service = ReadAloudService(FakeTTS(fail_synth=True),
                                   data_dir=self.root / 'read_aloud')
        result = service.read('Резервное озвучивание')
        self.assertEqual(result['status'], 'ok', result)
        self.assertEqual(result['error'], '')

    def test_both_synthesis_and_speak_failure_is_reported(self):
        service = ReadAloudService(
            FakeTTS(fail_synth=True, fail_speak=True),
            data_dir=self.root / 'read_aloud')
        result = service.read('Не должно прочитаться')
        self.assertEqual(result['status'], 'error')
        self.assertIn('нет голоса', result['error'])

    def test_stopped_playback_has_stopped_status(self):
        service = ReadAloudService(FakeTTS(stopped=True),
                                   data_dir=self.root / 'read_aloud')
        result = service.read('Остановка')
        self.assertEqual(result['status'], 'stopped')

    def test_stress_500_sequential_reads(self):
        store = SessionStore(self.root / 'sessions.jsonl')
        for index in range(500):
            result = self.service.read(
                f'Стресс-тест, итерация {index}.',
                voice={'name': 'Fake', 'backend': 'fake'},
                language='ru',
            )
            self.assertEqual(result['status'], 'ok', (index, result))
            store.append(build_session_record(
                text=result['text'], voice=result['voice'],
                language=result['language'], audio_status=result['status'],
                tts_audio=result['tts_audio'],
                duration_seconds=result['duration_seconds'],
                error=result['error']))
        self.assertEqual(store.count(), 500)
        self.assertEqual(store.last()['audio']['status'], 'ok')

    def test_stress_parallel_reads(self):
        results = []

        def worker(index):
            return self.service.read(f'Параллельно {index}')

        with ThreadPoolExecutor(max_workers=8) as pool:
            for result in pool.map(worker, range(160)):
                results.append(result)
        self.assertEqual(len(results), 160)
        self.assertTrue(all(item['status'] == 'ok' for item in results))
        for item in results:
            self.assertTrue(Path(item['tts_audio']['path']).is_file())

    def test_build_record_from_service_result(self):
        result = self.service.read('Запись в журнал')
        record = build_session_record(
            text=result['text'], recognized_text='распознано',
            voice=result['voice'], language=result['language'],
            location={'lat': 1.0, 'lon': 2.0, 'source': 'test'},
            microphone='mic', speaker='spk',
            audio_status=result['status'], recording=result['recording'],
            tts_audio=result['tts_audio'], started_at=result['started_at'],
            duration_seconds=result['duration_seconds'],
            error=result['error'])
        self.assertEqual(record['audio']['status'], 'ok')
        self.assertIsNotNone(record['tts_audio'])
        self.assertEqual(record['location']['lat'], 1.0)


class TTSQueueTests(unittest.TestCase):
    """Проверяет, что очередь TTS не создаёт поток на каждый вызов."""

    def _make_engine(self, delay=0.0, stopped=False):
        from types import MethodType
        from src.tts_engine import TTSEngine
        import queue

        engine = TTSEngine.__new__(TTSEngine)
        engine.engine_type = 'fake'
        engine.engine = None
        engine.edge_tts = None
        engine.lock = threading.Lock()
        engine._stop_flag = False
        engine._current_thread = None
        engine._current_process = None
        engine._mci_alias = 'test'
        engine._mci_open = False
        engine._queue = queue.Queue()
        engine._queue_lock = threading.Lock()
        engine._worker_thread = None
        engine._generation = 0
        engine._generation_lock = threading.Lock()
        engine.last_result = None
        engine.calls = []
        engine.call_lock = threading.Lock()

        def fake_speak(self, text, voice=None, reset_stop=True):
            if delay:
                time.sleep(delay)
            with self.call_lock:
                self.calls.append(text)
            return {'status': 'stopped' if stopped else 'ok', 'error': ''}

        engine.speak = MethodType(fake_speak, engine)
        return engine

    def test_2000_async_calls_use_one_worker(self):
        engine = self._make_engine()
        done = threading.Event()
        completed = []

        def callback(result):
            completed.append(result)
            if len(completed) == 2000:
                done.set()

        for index in range(2000):
            engine.speak_async(f'text-{index}', on_done=callback)
        self.assertTrue(done.wait(10.0))
        self.assertEqual(len(completed), 2000)
        self.assertEqual(len(engine.calls), 2000)
        self.assertEqual(len(set(engine.calls)), 2000)
        self.assertEqual(engine._worker_thread.name, 'tts-queue')

    def test_stop_cancels_queued_items(self):
        engine = self._make_engine(delay=0.002)
        cancelled = []

        def callback(result):
            if result.get('status') == 'stopped':
                cancelled.append(result)

        for index in range(50):
            engine.speak_async(f'text-{index}', on_done=callback)
        time.sleep(0.01)
        engine.stop()
        deadline = time.time() + 3.0
        while len(cancelled) < 49 and time.time() < deadline:
            time.sleep(0.01)
        self.assertGreater(len(cancelled), 0)
        self.assertLessEqual(len(engine.calls), 50)
        # Новый вызов после «Стоп» снова работает.
        completed = []
        event = threading.Event()
        engine.speak_async('after-stop', on_done=lambda r: (completed.append(r), event.set()))
        self.assertTrue(event.wait(2.0))
        self.assertEqual(completed[0]['status'], 'ok')
        self.assertEqual(engine.calls[-1], 'after-stop')


if __name__ == '__main__':
    unittest.main()
