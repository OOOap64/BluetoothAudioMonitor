# -*- coding: utf-8 -*-
"""Тесты преобразования записей Ogg/Opus в WAV."""

import os
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audio_utils import (
    convert_audio_to_wav, convert_opus_to_wav, opus_to_wav,
)


def _soundfile_available():
    try:
        import soundfile as sf
        return 'OPUS' in sf.available_subtypes('OGG')
    except Exception:
        return False


def _av_available():
    try:
        import av  # noqa: F401
        return True
    except Exception:
        return False


def _write_opus(path, rate=48000, seconds=1.0, channels=1, frequency=440.0):
    import soundfile as sf

    t = np.arange(int(rate * seconds), dtype=np.float32) / float(rate)
    tone = (0.4 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    if channels > 1:
        audio = np.column_stack([tone] * channels)
    else:
        audio = tone
    sf.write(str(path), audio, rate, format='OGG', subtype='OPUS')


@unittest.skipUnless(_soundfile_available(), 'нет кодировщика Ogg/Opus')
class OpusConversionTests(unittest.TestCase):
    def test_default_output_is_wav_next_to_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'recording.opus'
            _write_opus(source)

            info = opus_to_wav(source)

            self.assertEqual(info['status'], 'ok')
            self.assertEqual(info['format'], 'wav')
            self.assertEqual(info['source_format'], 'opus')
            output = Path(info['path'])
            self.assertEqual(output, source.with_suffix('.wav'))
            self.assertTrue(output.is_file())
            self.assertGreater(info['size_bytes'], 44)
            self.assertAlmostEqual(info['duration_seconds'], 1.0, places=2)
            with wave.open(str(output), 'rb') as handle:
                self.assertEqual(handle.getnchannels(), 1)
                self.assertEqual(handle.getsampwidth(), 2)
                self.assertEqual(handle.getframerate(), 48000)
                self.assertEqual(handle.getnframes(), 48000)

    def test_explicit_path_resamples_and_downmixes(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'stereo.opus'
            _write_opus(source, channels=2)
            destination = Path(tmp) / 'nested' / 'converted'

            info = convert_audio_to_wav(
                source, destination, sample_rate=16000, channels=1)

            self.assertEqual(info['status'], 'ok')
            output = Path(info['path'])
            self.assertEqual(output, destination.with_suffix('.wav'))
            self.assertTrue(output.is_file())
            with wave.open(str(output), 'rb') as handle:
                self.assertEqual(handle.getnchannels(), 1)
                self.assertEqual(handle.getframerate(), 16000)
                self.assertAlmostEqual(
                    handle.getnframes() / 16000.0, 1.0, places=2)

    def test_alias_is_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'clip.opus'
            _write_opus(source)
            info = convert_opus_to_wav(source)
            self.assertEqual(info['status'], 'ok')
            self.assertTrue(Path(info['path']).is_file())

    def test_missing_source_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                convert_audio_to_wav(Path(tmp) / 'missing.opus')

    def test_same_source_and_destination_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'clip.opus'
            _write_opus(source)
            with self.assertRaises(ValueError):
                convert_audio_to_wav(source, source)

    @unittest.skipUnless(_av_available(), 'PyAV не установлен')
    def test_pyav_fallback_when_soundfile_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'fallback.opus'
            _write_opus(source)
            with mock.patch(
                    'soundfile.read',
                    side_effect=RuntimeError('soundfile недоступен')):
                info = convert_audio_to_wav(source)
            self.assertEqual(info['status'], 'ok')
            output = Path(info['path'])
            self.assertTrue(output.is_file())
            with wave.open(str(output), 'rb') as handle:
                self.assertEqual(handle.getframerate(), 48000)
                self.assertGreater(handle.getnframes(), 0)


if __name__ == '__main__':
    unittest.main()
