# -*- coding: utf-8 -*-
"""Тесты сохранения аудиозаписи после анализа (WAV)."""

import os
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audio_utils import save_wav
from src.audio_monitor import AudioMonitor
from src.session_log import default_recordings_dir


class SaveWavTests(unittest.TestCase):
    def test_writes_valid_mono_wav(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'nested' / 'clip.wav'
            t = np.linspace(0.0, 1.0, 8000, endpoint=False)
            audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
            info = save_wav(path, audio, 8000)
            self.assertEqual(info['status'], 'ok')
            self.assertEqual(info['channels'], 1)
            self.assertEqual(info['sample_rate'], 8000)
            self.assertTrue(path.is_file())
            self.assertGreater(info['size_bytes'], 44)
            self.assertAlmostEqual(info['duration_seconds'], 1.0, places=2)
            with wave.open(str(path), 'rb') as handle:
                self.assertEqual(handle.getnchannels(), 1)
                self.assertEqual(handle.getsampwidth(), 2)
                self.assertEqual(handle.getframerate(), 8000)
                self.assertEqual(handle.getnframes(), 8000)

    def test_loud_signal_is_clipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'loud.wav'
            audio = np.array([5.0, -5.0, 2.0, -2.0], dtype=np.float32)
            info = save_wav(path, audio, 16000)
            self.assertEqual(info['status'], 'ok')
            with wave.open(str(path), 'rb') as handle:
                frames = np.frombuffer(handle.readframes(4), dtype=np.int16)
            self.assertLessEqual(int(np.max(np.abs(frames))), 32767)
            self.assertGreater(int(np.max(np.abs(frames))), 30000)

    def test_empty_audio_creates_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'empty.wav'
            info = save_wav(path, np.zeros(0, dtype=np.float32), 16000)
            self.assertEqual(info['status'], 'ok')
            self.assertTrue(path.is_file())
            self.assertEqual(info['duration_seconds'], 0.0)

    def test_stereo_audio_keeps_two_channels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'stereo.wav'
            audio = np.zeros((1000, 2), dtype=np.float32)
            info = save_wav(path, audio, 8000, channels=2)
            self.assertEqual(info['status'], 'ok')
            self.assertEqual(info['channels'], 2)
            with wave.open(str(path), 'rb') as handle:
                self.assertEqual(handle.getnchannels(), 2)
                self.assertEqual(handle.getnframes(), 1000)


class AudioMonitorRecordingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old_env = os.environ.get('BLUETOOTH_MONITOR_DATA_DIR')
        os.environ['BLUETOOTH_MONITOR_DATA_DIR'] = self._tmp.name

    def tearDown(self):
        if self._old_env is None:
            os.environ.pop('BLUETOOTH_MONITOR_DATA_DIR', None)
        else:
            os.environ['BLUETOOTH_MONITOR_DATA_DIR'] = self._old_env
        self._tmp.cleanup()

    def _feed(self, monitor, seconds=1.0, rate=16000):
        monitor.rate = rate
        monitor.channels = 1
        rng = np.random.default_rng(0)
        samples = (rng.standard_normal(int(rate * seconds)) * 0.1).astype(
            np.float32)
        monitor._buffer.append(samples.copy())
        return samples

    def test_save_to_explicit_path(self):
        monitor = AudioMonitor(rate=16000)
        try:
            self._feed(monitor)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'out.wav'
                info = monitor.save_recording(path)
                self.assertEqual(info['status'], 'ok')
                self.assertTrue(path.is_file())
                with wave.open(str(path), 'rb') as handle:
                    self.assertEqual(handle.getframerate(), 16000)
                    self.assertEqual(handle.getnframes(), 16000)
        finally:
            monitor.stop()

    def test_save_without_path_uses_recordings_dir(self):
        monitor = AudioMonitor(rate=16000)
        try:
            self._feed(monitor)
            info = monitor.save_recording()
            self.assertEqual(info['status'], 'ok')
            saved = Path(info['path'])
            self.assertTrue(saved.is_file())
            self.assertEqual(saved.parent, default_recordings_dir())
            self.assertTrue(saved.name.startswith('capture-'))
        finally:
            monitor.stop()

    def test_save_empty_buffer_reports_empty(self):
        monitor = AudioMonitor(rate=16000)
        try:
            info = monitor.save_recording()
            self.assertEqual(info['status'], 'empty')
            self.assertIsNone(info['path'])
        finally:
            monitor.stop()

    def test_seconds_limits_snapshot(self):
        monitor = AudioMonitor(rate=16000)
        try:
            self._feed(monitor, seconds=3.0)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'tail.wav'
                info = monitor.save_recording(path, seconds=1.0)
                self.assertEqual(info['status'], 'ok')
                with wave.open(str(path), 'rb') as handle:
                    self.assertEqual(handle.getnframes(), 16000)
        finally:
            monitor.stop()


if __name__ == '__main__':
    unittest.main()
