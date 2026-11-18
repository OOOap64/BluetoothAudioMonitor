# -*- coding: utf-8 -*-
"""Тесты журнала сеансов чтения вслух."""

import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.session_log import (
    SessionStore, build_session_record, default_sessions_path,
)


class SessionRecordTests(unittest.TestCase):
    def test_build_record_schema(self):
        record = build_session_record(
            text='Привет, мир!',
            recognized_text='Привет мир',
            voice={'id': 'piper::ru', 'name': 'Ирина', 'backend': 'piper'},
            language='ru',
            location={'lat': 55.7558, 'lon': 37.6173, 'source': 'manual'},
            microphone='Микрофон',
            speaker='Наушники',
            audio_status='spoken',
            recording={'path': '/tmp/a.wav', 'status': 'ok',
                       'duration_seconds': 1.25},
            tts_audio={'path': '/tmp/b.wav', 'status': 'saved'},
        )
        self.assertEqual(record['text'], 'Привет, мир!')
        self.assertEqual(record['voice']['name'], 'Ирина')
        self.assertEqual(record['location']['lat'], 55.7558)
        self.assertEqual(record['devices']['speaker'], 'Наушники')
        self.assertEqual(record['audio']['status'], 'spoken')
        self.assertEqual(record['recording']['duration_seconds'], 1.25)

    def test_invalid_location_is_dropped(self):
        record = build_session_record(
            text='x', location={'lat': 999, 'lon': 0, 'source': 'bad'})
        self.assertIsNone(record['location'])
        record = build_session_record(
            text='x', location={'lat': '55.1', 'lon': '37.2'})
        self.assertEqual(record['location']['lat'], 55.1)

    def test_default_path_uses_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = os.environ.get('BLUETOOTH_MONITOR_DATA_DIR')
            os.environ['BLUETOOTH_MONITOR_DATA_DIR'] = tmp
            try:
                path = default_sessions_path()
                self.assertEqual(Path(path), Path(tmp) / 'read_aloud' / 'sessions.jsonl')
            finally:
                if old is None:
                    os.environ.pop('BLUETOOTH_MONITOR_DATA_DIR', None)
                else:
                    os.environ['BLUETOOTH_MONITOR_DATA_DIR'] = old


class SessionStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'sessions.jsonl'
        self.store = SessionStore(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def test_append_load_and_last(self):
        self.assertEqual(self.store.count(), 0)
        first = self.store.append({'text': 'один'})
        second = self.store.append(
            build_session_record(text='два', language='en'))
        self.assertNotEqual(first['id'], second['id'])
        records = self.store.load()
        self.assertEqual([item['text'] for item in records], ['один', 'два'])
        self.assertEqual(self.store.last()['text'], 'два')
        self.assertEqual(self.store.count(), 2)
        self.assertTrue(first['created_at'])

    def test_unicode_and_newlines_are_preserved(self):
        text = 'Строка 1\nСтрока 2 — «ёжик» и emoji 🎧'
        self.store.append(build_session_record(text=text))
        self.assertEqual(self.store.load()[0]['text'], text)

    def test_corrupted_line_is_skipped(self):
        self.store.append({'text': 'ok'})
        with open(self.path, 'a', encoding='utf-8') as handle:
            handle.write('{broken json}\n\n')
        self.store.append({'text': 'after'})
        records = self.store.load()
        self.assertEqual([item['text'] for item in records], ['ok', 'after'])

    def test_concurrent_appends_are_not_lost(self):
        errors = []

        def writer(worker):
            try:
                for index in range(25):
                    self.store.append({
                        'text': f'worker={worker} item={index}',
                    })
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertFalse(errors)
        records = self.store.load()
        self.assertEqual(len(records), 8 * 25)
        for line in self.path.read_text(encoding='utf-8').splitlines():
            self.assertIsInstance(json.loads(line), dict)

    def test_export_csv_and_json(self):
        self.store.append(build_session_record(
            text='экспорт', voice={'name': 'Голос', 'backend': 'piper'},
            location={'lat': 1.5, 'lon': 2.5}, microphone='mic',
            speaker='spk', audio_status='ok',
            recording={'path': '/tmp/r.wav'}))
        csv_path = Path(self.temp.name) / 'out.csv'
        json_path = Path(self.temp.name) / 'out.json'
        self.store.export_csv(csv_path)
        self.store.export_json(json_path)
        csv_text = csv_path.read_text(encoding='utf-8-sig')
        self.assertIn('voice_name', csv_text)
        self.assertIn('Голос', csv_text)
        payload = json.loads(json_path.read_text(encoding='utf-8'))
        self.assertEqual(payload['sessions'][0]['text'], 'экспорт')


if __name__ == '__main__':
    unittest.main()
