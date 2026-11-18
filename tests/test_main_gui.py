# -*- coding: utf-8 -*-
"""GUI-тест произвольного чтения и сохранения результата."""

import os
import sys
import tempfile
import threading
import time
import unittest
import wave
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from PyQt6.QtCore import QEventLoop, QThread, QTimer
    from PyQt6.QtWidgets import QApplication
    import main
except Exception as exc:  # pragma: no cover - зависит от окружения
    raise unittest.SkipTest(f'GUI недоступен: {exc}')


class FakeTTS:
    def __init__(self):
        self.preview_calls = []

    def synthesize(self, text, voice=None):
        fd, path = tempfile.mkstemp(suffix='.wav', prefix='gui-tts-')
        os.close(fd)
        with wave.open(path, 'wb') as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(b'\x00\x00' * 80)
        return {'status': 'ok', 'path': path, 'backend': 'fake'}

    def play_file(self, path):
        return True

    def speak(self, text, voice=None):
        return {'status': 'ok', 'error': ''}

    def speak_async(self, text, voice=None, on_done=None):
        self.preview_calls.append((text, voice))
        if on_done is not None:
            on_done({'status': 'ok', 'error': '', 'backend': 'fake'})
        return None

    def stop(self):
        return None

_APP = None


class MainWindowReadAloudTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global _APP
        cls.temp = tempfile.TemporaryDirectory()
        os.environ['BLUETOOTH_MONITOR_DATA_DIR'] = cls.temp.name
        _APP = QApplication.instance() or QApplication([])
        cls.window = main.MainWindow()
        cls.window.tts = FakeTTS()

    @classmethod
    def tearDownClass(cls):
        try:
            cls.window.close()
        except Exception:
            pass
        _APP.processEvents()
        cls.temp.cleanup()

    def _wait_for_read(self, timeout=10.0):
        deadline = time.time() + timeout
        while self.window._read_active and time.time() < deadline:
            _APP.processEvents()
            time.sleep(0.01)
        _APP.processEvents()
        self.assertFalse(self.window._read_active)

    def _select_voice(self, voice_id):
        combo = self.window.tts_voice_combo
        for index in range(combo.count()):
            voice = combo.itemData(index)
            if isinstance(voice, dict) and voice.get('id') == voice_id:
                combo.setCurrentIndex(index)
                return voice
        self.skipTest(f'голос {voice_id!r} отсутствует в комбобоксе')

    def test_voice_language_detection(self):
        self.assertTrue(self.window._voice_is_english({
            'id': 'edge::en-US-AriaNeural', 'name': 'Aria (Edge, English)'
        }))
        self.assertFalse(self.window._voice_is_english({
            'id': 'piper::ru_RU-irina-medium.onnx',
            'name': 'Irina (Piper, ru-RU, medium)'
        }))

    def test_preview_uses_english_sample_for_english_voice(self):
        self.window.custom_text_edit.setPlainText('')
        voice = self._select_voice('edge::en-US-AriaNeural')
        text = self.window._preview_text_for_voice(voice)
        self.assertIn('Hello', text)
        self.window.tts.preview_calls.clear()
        self.window.preview_selected_voice()
        self.assertEqual(len(self.window.tts.preview_calls), 1)
        self.assertEqual(self.window.tts.preview_calls[0][0], text)
        self.assertEqual(self.window.tts.preview_calls[0][1], voice)

    def test_preview_uses_custom_text_when_entered(self):
        self.window.custom_text_edit.setPlainText(
            '  Проверка [12:00:01] нового текста  ')
        voice = self._select_voice('piper::ru_RU-irina-medium.onnx')
        try:
            text = self.window._preview_text_for_voice(voice)
            self.assertNotIn('[12:00:01]', text)
            self.assertEqual(text, 'Проверка нового текста')
            self.window.tts.preview_calls.clear()
            self.window.preview_selected_voice()
            self.assertEqual(self.window.tts.preview_calls[0][0], text)
        finally:
            self.window.custom_text_edit.setPlainText('')

    def test_preview_falls_back_for_english_voice_and_russian_text(self):
        self._select_voice('edge::en-US-AriaNeural')
        self.window.custom_text_edit.setPlainText('Привет, это русский текст')
        try:
            self.window.tts.preview_calls.clear()
            self.window.preview_selected_voice()
            self.assertEqual(len(self.window.tts.preview_calls), 1)
            _text, used_voice = self.window.tts.preview_calls[0]
            self.assertFalse(self.window._voice_is_english(used_voice))
        finally:
            self.window.custom_text_edit.setPlainText('')

    def test_voice_for_text_falls_back_for_english_voice_and_russian(self):
        english = self._select_voice('edge::en-US-GuyNeural')
        fallback = self.window._voice_for_text('Привет, это русский текст')
        self.assertIsNot(fallback, english)
        self.assertFalse(self.window._voice_is_english(fallback))
        self.assertEqual(
            self.window._voice_for_text('Hello, this is English text'),
            english)

    def test_main_thread_dispatcher_runs_on_gui_thread(self):
        result = {}
        loop = QEventLoop()
        gui_thread = QThread.currentThread()
        dispatcher = main._MainThreadDispatcher()

        def task():
            result['thread'] = QThread.currentThread()
            loop.quit()

        def worker():
            dispatcher(task)

        threading.Thread(target=worker, daemon=True).start()
        QTimer.singleShot(2000, loop.quit)
        loop.exec()
        self.assertIn('thread', result)
        self.assertEqual(result['thread'], gui_thread)

    def test_phrase_hints_field_exists_and_is_saved(self):
        self.assertTrue(hasattr(self.window, 'phrases_edit'))
        self.window.phrases_edit.setPlainText('Фраза один\nФраза два')
        self.assertEqual(
            'Фраза один\nФраза два', self.window._current_phrases())
        self.window._save_phrases()
        self.assertEqual(
            'Фраза один\nФраза два',
            str(self.window.settings.value('asr/phrases', '')))

    def test_hotkeys_include_read_and_save(self):
        keys = {
            key
            for _group, items in main.HOTKEY_GROUPS
            for key, _title, _handler in items
        }
        self.assertIn('F9', keys)
        self.assertIn('F10', keys)
        self.assertIn('Ctrl+L', keys)
        self.assertNotIn('Ctrl+Shift+M', keys)

    def test_hotkeys_include_save_audio(self):
        keys = {
            key
            for _group, items in main.HOTKEY_GROUPS
            for key, _title, _handler in items
        }
        self.assertIn('F11', keys)

    def test_all_hotkey_handlers_exist(self):
        for _group, items in main.HOTKEY_GROUPS:
            for _key, title, handler_name in items:
                with self.subTest(handler=handler_name, title=title):
                    self.assertTrue(
                        hasattr(self.window, handler_name),
                        f'нет обработчика {handler_name} для «{title}»')

    def test_save_audio_button_present_and_disabled(self):
        self.assertTrue(hasattr(self.window, 'save_audio_btn'))
        self.window._monitor_active = False
        self.window._refresh_buttons()
        self.assertFalse(self.window.save_audio_btn.isEnabled())

    def test_save_audio_button_enabled_during_capture(self):
        self.window._monitor_active = True
        self.window._refresh_buttons()
        self.assertTrue(self.window.save_audio_btn.isEnabled())
        self.window._monitor_active = False
        self.window._refresh_buttons()

    def test_save_monitoring_audio_writes_file(self):
        from unittest import mock

        class _FakeMonitor:
            def save_recording(self, path, seconds=None):
                Path(path).write_bytes(b'RIFF0000WAVE')
                return {
                    'status': 'ok',
                    'path': str(path),
                    'duration_seconds': 0.5,
                    'size_bytes': 12,
                }

        class _FakeThread:
            monitor = _FakeMonitor()

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'capture.wav'
            self.window.monitor_thread = _FakeThread()
            self.window._monitor_active = True
            try:
                with mock.patch.object(
                        main.QFileDialog, 'getSaveFileName',
                        return_value=(str(target), 'WAV (*.wav)')):
                    self.window.save_monitoring_audio()
            finally:
                self.window.monitor_thread = None
                self.window._monitor_active = False
            self.assertTrue(target.is_file())

    def test_save_monitoring_audio_without_capture(self):
        self.window.monitor_thread = None
        self.window._monitor_active = False
        # Не должно бросать исключение, только запись в журнал.
        self.window.save_monitoring_audio()

    def test_training_button_is_removed(self):
        self.assertFalse(hasattr(self.window, 'train_sound_btn'))

    def test_custom_text_is_read_and_saved(self):
        self.window.auto_save_cb.setChecked(True)
        self.window.record_while_read_cb.setChecked(False)
        self.window.custom_text_edit.setPlainText(
            'Произвольный текст для проверки журнала.')
        before = self.window.session_store.count()
        self.window.read_custom_aloud()
        self._wait_for_read()
        self.assertEqual(self.window.session_store.count(), before + 1)
        record = self.window.session_store.last()
        self.assertEqual(
            record['text'], 'Произвольный текст для проверки журнала.')
        self.assertEqual(record['source'], 'custom_text')
        self.assertEqual(record['audio']['status'], 'ok')
        self.assertTrue(Path(record['tts_audio']['path']).is_file())
        self.assertTrue(self.window._last_read_saved)

    def test_manual_save_does_not_duplicate(self):
        self.window.auto_save_cb.setChecked(False)
        self.window.custom_text_edit.setPlainText('Ручное сохранение')
        before = self.window.session_store.count()
        self.window.read_custom_aloud()
        self._wait_for_read()
        self.assertEqual(self.window.session_store.count(), before)
        self.window.save_read_result()
        self.assertEqual(self.window.session_store.count(), before + 1)
        # Повторное нажатие не создаёт дубликат.
        self.window.save_read_result()
        self.assertEqual(self.window.session_store.count(), before + 1)

    def test_empty_text_is_rejected(self):
        self.window.custom_text_edit.setPlainText('   ')
        before = self.window.session_store.count()
        self.window.read_custom_aloud()
        self.assertFalse(self.window._read_active)
        self.assertEqual(self.window.session_store.count(), before)

    def test_manual_coordinates_are_saved(self):
        self.window.latitude_edit.setText('59.9343')
        self.window.longitude_edit.setText('30.3351')
        self.window.auto_save_cb.setChecked(True)
        self.window.custom_text_edit.setPlainText('Координаты в записи')
        before = self.window.session_store.count()
        self.window.read_custom_aloud()
        self._wait_for_read()
        self.assertEqual(self.window.session_store.count(), before + 1)
        location = self.window.session_store.last()['location']
        self.assertEqual(location['lat'], 59.9343)
        self.assertEqual(location['lon'], 30.3351)
        self.assertEqual(location['source'], 'manual')

    def test_history_dialog_can_be_created(self):
        dialog = main.SessionHistoryDialog(
            self.window, self.window.session_store)
        self.assertGreaterEqual(dialog.table.rowCount(), 0)
        self.assertIn('Записей:', dialog.summary.text())
        dialog.close()


if __name__ == '__main__':
    unittest.main()
