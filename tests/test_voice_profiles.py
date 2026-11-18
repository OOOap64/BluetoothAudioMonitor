# -*- coding: utf-8 -*-
"""Тесты хранилища пользовательских голосов и настроек TTS."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tts_engine import (
    TTSEngine, _pretty_piper_name, find_piper_executable, find_piper_models,
)
from src.voice_profiles import (
    VoiceProfileStore, default_store_path, normalize_profile,
)


class VoiceProfileStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.temp_dir.name, 'voices.json')
        self.store = VoiceProfileStore(self.path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_add_and_reload(self):
        saved = self.store.add({
            'name': 'Мой голос',
            'backend': 'edge',
            'voice': 'ru-RU-SvetlanaNeural',
            'rate': 180,
            'volume': 0.8,
        })
        self.assertTrue(os.path.isfile(self.path))
        self.assertTrue(saved['id'].startswith('custom-'))

        again = VoiceProfileStore(self.path)
        self.assertEqual(1, len(again))
        profile = again.get(saved['id'])
        self.assertEqual('Мой голос', profile['name'])
        self.assertEqual('edge', profile['backend'])
        self.assertEqual('ru-RU-SvetlanaNeural', profile['voice'])
        self.assertEqual(180, profile['rate'])
        self.assertEqual(0.8, profile['volume'])

    def test_json_is_utf8_and_indented(self):
        self.store.add({
            'name': 'Голос «Тест»',
            'backend': 'edge',
            'voice': 'ru-RU-DmitryNeural',
        })
        raw = Path(self.path).read_text(encoding='utf-8')
        self.assertIn('Голос «Тест»', raw)
        data = json.loads(raw)
        self.assertEqual(1, data['version'])
        self.assertEqual(1, len(data['voices']))

    def test_duplicate_name_replaces_profile(self):
        first = self.store.add({
            'name': 'Мой голос',
            'backend': 'edge',
            'voice': 'ru-RU-DmitryNeural',
        })
        second = self.store.add({
            'name': 'мой голос',
            'backend': 'edge',
            'voice': 'ru-RU-SvetlanaNeural',
        })
        self.assertNotEqual(first['id'], second['id'])
        self.assertEqual(1, len(self.store))
        self.assertEqual('ru-RU-SvetlanaNeural',
                         self.store.profiles[0]['voice'])

    def test_add_assigns_unique_ids(self):
        first = self.store.add({
            'name': 'Первый', 'backend': 'edge',
            'voice': 'ru-RU-DmitryNeural'})
        second = self.store.add({
            'name': 'Второй', 'id': first['id'], 'backend': 'edge',
            'voice': 'ru-RU-SvetlanaNeural'})
        self.assertNotEqual(first['id'], second['id'])

    def test_remove(self):
        saved = self.store.add({
            'name': 'Удаляемый', 'backend': 'edge',
            'voice': 'ru-RU-DmitryNeural'})
        self.assertTrue(self.store.remove(saved['id']))
        self.assertFalse(self.store.remove(saved['id']))
        self.assertEqual(0, len(VoiceProfileStore(self.path)))

    def test_find_by_name(self):
        self.store.add({
            'name': '  Мой Голос  ', 'backend': 'edge',
            'voice': 'ru-RU-DmitryNeural'})
        found = self.store.find_by_name('мой голос')
        self.assertIsNotNone(found)
        self.assertEqual('Мой Голос', found['name'])

    def test_corrupted_file_does_not_crash(self):
        Path(self.path).write_text('{not json', encoding='utf-8')
        store = VoiceProfileStore(self.path)
        self.assertEqual(0, len(store))
        store.add({'name': 'Новый', 'backend': 'edge',
                   'voice': 'ru-RU-DmitryNeural'})
        self.assertEqual(1, len(VoiceProfileStore(self.path)))

    def test_unknown_backend_is_rejected(self):
        with self.assertRaises(ValueError):
            self.store.add({'name': 'X', 'backend': 'unknown',
                            'voice': 'x'})

    def test_piper_requires_model(self):
        with self.assertRaises(ValueError):
            normalize_profile({'name': 'Piper', 'backend': 'piper'})
        profile = normalize_profile({
            'name': 'Piper', 'backend': 'piper',
            'model_path': 'voice.onnx'})
        self.assertEqual('voice.onnx', profile['model_path'])

    def test_pyttsx3_requires_voice(self):
        with self.assertRaises(ValueError):
            normalize_profile({'name': 'Системный', 'backend': 'pyttsx3'})

    def test_values_are_clamped_and_defaults_applied(self):
        profile = normalize_profile({
            'name': 'Тест', 'backend': 'edge',
            'voice': 'ru-RU-DmitryNeural',
            'rate': '9999', 'volume': -5})
        self.assertEqual(400, profile['rate'])
        self.assertEqual(0.0, profile['volume'])

        profile = normalize_profile({
            'name': 'Тест', 'backend': 'edge',
            'voice': 'ru-RU-DmitryNeural',
            'rate': 'not a number', 'volume': None})
        self.assertEqual(165, profile['rate'])
        self.assertEqual(1.0, profile['volume'])

    def test_backend_aliases(self):
        self.assertEqual(
            'pyttsx3',
            normalize_profile({
                'name': 'A', 'backend': 'sapi', 'voice': 'x'})['backend'])
        self.assertEqual(
            'edge',
            normalize_profile({
                'name': 'B', 'backend': 'edge-tts',
                'voice': 'x'})['backend'])
        self.assertEqual(
            'piper',
            normalize_profile({
                'name': 'C', 'backend': 'custom',
                'model_path': 'x.onnx'})['backend'])

    def test_default_store_path_honours_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(
                    os.environ,
                    {'BLUETOOTH_MONITOR_VOICE_DIR': directory}):
                self.assertEqual(
                    os.path.join(directory, 'voices.json'),
                    default_store_path())


class TTSEngineVoiceTests(unittest.TestCase):
    def make_engine(self):
        engine = TTSEngine.__new__(TTSEngine)
        engine.engine_type = 'pyttsx3'
        engine.edge_tts = None
        return engine

    def test_resolve_profile(self):
        engine = self.make_engine()
        backend, profile = engine._resolve_voice({
            'backend': 'piper',
            'voice': '',
            'model_path': 'voice.onnx',
        })
        self.assertEqual('piper', backend)
        self.assertEqual('voice.onnx', profile['model_path'])

    def test_resolve_string_voice(self):
        engine = self.make_engine()
        backend, profile = engine._resolve_voice('pyttsx3::voice-1')
        self.assertEqual('pyttsx3', backend)
        self.assertEqual('voice-1', profile['voice'])

        backend, profile = engine._resolve_voice('ru-RU-DmitryNeural')
        self.assertEqual('edge', backend)
        self.assertEqual('ru-RU-DmitryNeural', profile['voice'])

    def test_resolve_default_voice(self):
        engine = self.make_engine()
        backend, profile = engine._resolve_voice(None)
        self.assertEqual('pyttsx3', backend)
        self.assertEqual('', profile['voice'])

    def test_piper_command(self):
        command = TTSEngine._build_piper_command(
            r'C:\piper\piper.exe',
            r'C:\voices\my.onnx',
            config_path=r'C:\voices\my.onnx.json',
            output_path=r'C:\temp\out.wav',
        )
        self.assertEqual(r'C:\piper\piper.exe', command[0])
        self.assertIn('--model', command)
        self.assertIn(r'C:\voices\my.onnx', command)
        self.assertIn('--config', command)
        self.assertIn(r'C:\voices\my.onnx.json', command)
        self.assertIn('--output_file', command)
        self.assertIn(r'C:\temp\out.wav', command)

    def test_piper_python_script_command(self):
        command = TTSEngine._build_piper_command(
            'piper.py', 'voice.onnx', output_path='out.wav')
        self.assertEqual(sys.executable, command[0])
        self.assertEqual('piper.py', command[1])

    def test_get_piper_voices_builds_profiles(self):
        engine = self.make_engine()
        with mock.patch(
                'src.tts_engine.find_piper_executable',
                return_value=r'C:\piper\piper.exe'), mock.patch(
                'src.tts_engine.find_piper_models',
                return_value=[{
                    'model_path': r'C:\piper\voices\ru_RU-irina-medium.onnx',
                    'config_path': r'C:\piper\voices\ru_RU-irina-medium.onnx.json',
                    'name': 'Irina (Piper, ru-RU, medium)',
                }]):
            voices = engine.get_piper_voices()
        self.assertEqual(1, len(voices))
        voice = voices[0]
        self.assertEqual('piper', voice['backend'])
        self.assertEqual(
            r'C:\piper\voices\ru_RU-irina-medium.onnx', voice['model_path'])
        self.assertEqual(r'C:\piper\piper.exe', voice['executable_path'])
        self.assertTrue(voice['id'].startswith('piper::'))


class PiperDiscoveryTests(unittest.TestCase):
    def test_pretty_name(self):
        self.assertEqual(
            'Irina (Piper, ru-RU, medium)',
            _pretty_piper_name('ru_RU-irina-medium'))
        self.assertEqual('custom (Piper)', _pretty_piper_name('custom'))

    def test_find_piper_models_uses_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / 'ru_RU-irina-medium.onnx'
            model.write_bytes(b'onnx')
            Path(str(model) + '.json').write_text('{}', encoding='utf-8')
            (Path(directory) / 'notes.txt').write_text('x', encoding='utf-8')
            with mock.patch.dict(
                    os.environ,
                    {'BLUETOOTH_MONITOR_PIPER_VOICES': directory}), \
                    mock.patch('src.tts_engine.piper_search_dirs',
                               return_value=[]):
                models = find_piper_models()
            self.assertEqual(1, len(models))
            self.assertEqual(str(model), models[0]['model_path'])
            self.assertTrue(models[0]['config_path'].endswith('.onnx.json'))
            self.assertIn('Piper', models[0]['name'])

    def test_find_piper_executable_uses_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / 'piper.exe'
            executable.write_bytes(b'fake')
            with mock.patch.dict(
                    os.environ,
                    {'BLUETOOTH_MONITOR_PIPER_EXE': str(executable)}):
                self.assertEqual(str(executable), find_piper_executable())

    def test_missing_piper_executable_returns_empty(self):
        with mock.patch.dict(
                os.environ,
                {'BLUETOOTH_MONITOR_PIPER_EXE':
                 os.path.join(tempfile.gettempdir(), 'nope', 'piper.exe')}):
            with mock.patch('src.tts_engine.piper_search_dirs',
                            return_value=[]):
                with mock.patch('src.tts_engine.shutil.which',
                                return_value=None):
                    self.assertEqual('', find_piper_executable())


if __name__ == '__main__':
    unittest.main(verbosity=2)
