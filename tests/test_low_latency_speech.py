# -*- coding: utf-8 -*-
"""Тесты быстрого режима распознавания речи."""

import os
import sys
import time
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audio_monitor import AudioMonitor, Utterance, _UtteranceDetector, merge_partial_text
import src.speech_transcriber as speech_transcriber
from src.speech_transcriber import SpeechTranscriber


class RecordingModel:
    def __init__(self):
        self.options = None

    def transcribe(self, audio, **kwargs):
        self.options = kwargs
        return [], {}


class LowLatencyTests(unittest.TestCase):
    def test_defaults_are_fast(self):
        with mock.patch.dict(os.environ, {'BLUETOOTH_MONITOR_WHISPER_BEAM': ''}):
            transcriber = SpeechTranscriber()
        self.assertEqual('tiny', transcriber.model_size)
        self.assertGreaterEqual(transcriber.cpu_threads, 1)
        self.assertLessEqual(transcriber.cpu_threads, 8)

        model = RecordingModel()
        transcriber.model = model
        transcriber.loaded = True
        with mock.patch.dict(os.environ, {'BLUETOOTH_MONITOR_WHISPER_BEAM': ''}):
            transcriber.transcribe_segments(
                np.ones(16000, dtype=np.float32), sample_rate=16000)
        self.assertEqual(1, model.options['beam_size'])
        self.assertEqual(1, model.options['best_of'])
        self.assertFalse(model.options['condition_on_previous_text'])

    def test_detector_emits_partial_before_pause(self):
        detector = _UtteranceDetector(16000)
        speech = np.full(16000 * 2, 0.5, dtype=np.float32)
        silence = np.zeros(16000, dtype=np.float32)
        items = detector.push(np.concatenate((speech, silence)))

        partials = [item for item in items if item.partial]
        finals = [item for item in items if not item.partial]
        self.assertTrue(partials)
        self.assertEqual(1, len(finals))
        self.assertLessEqual(len(partials[0].audio) / 16000.0, 1.05)
        self.assertEqual(partials[0].phrase_id, finals[0].phrase_id)
        self.assertLess(len(finals[0].audio) / 16000.0, 2.05)

    def test_merge_partial_text(self):
        self.assertEqual('привет как дела', merge_partial_text('привет как', 'как дела'))
        self.assertEqual('привет как дела', merge_partial_text('привет как дела', 'как дела'))
        self.assertEqual('привет', merge_partial_text('', 'привет'))
        self.assertEqual('привет дела', merge_partial_text('привет', 'дела'))

    def test_live_text_and_final_phrase(self):
        monitor = AudioMonitor()
        monitor.autocorrect = False
        monitor._last_recognition_ms = 220
        monitor._update_partial(1, 'привет как')
        self.assertEqual('привет как', monitor.get_analysis()['live_text'])
        self.assertEqual(220, monitor.get_analysis()['last_recognition_ms'])

        monitor._update_partial(1, 'как дела')
        self.assertEqual('привет как дела', monitor.get_analysis()['live_text'])

        monitor._finish_phrase(1, 'привет как дела!', 310)
        data = monitor.get_analysis()
        self.assertEqual('', data['live_text'])
        self.assertEqual(310, data['last_recognition_ms'])
        self.assertIn('привет как дела!', data['transcribed_text'])

    def test_final_has_priority_over_queued_partials(self):
        monitor = AudioMonitor()
        first = Utterance(np.zeros(4), True, 1, time.time())
        second = Utterance(np.zeros(4), True, 1, time.time())
        final = Utterance(np.zeros(4), False, 1, time.time())
        monitor._enqueue_utterance(first)
        monitor._enqueue_utterance(second)
        monitor._enqueue_utterance(final)
        self.assertEqual(final, monitor._transcribe_queue[0])
        self.assertEqual(1, len(monitor._transcribe_queue))

    def test_identical_partials_are_not_queued_twice(self):
        monitor = AudioMonitor()
        first = Utterance(np.ones(4), True, 1, time.time())
        duplicate = Utterance(np.ones(4), True, 1, time.time())
        monitor._enqueue_utterance(first)
        monitor._enqueue_utterance(duplicate)
        self.assertEqual(1, len(monitor._transcribe_queue))

    def test_failed_model_load_is_cached(self):
        speech_transcriber.clear_model_cache()
        attempts = []

        class Boom:
            def __init__(self, *args, **kwargs):
                attempts.append(kwargs)
                raise RuntimeError('offline')

        try:
            with mock.patch('faster_whisper.WhisperModel', Boom):
                for _ in range(3):
                    transcriber = SpeechTranscriber(
                        model_size='debug-failure', device='cpu',
                        compute_type='int8', cpu_threads=1,
                        autocorrect=False)
                    with self.assertRaises(RuntimeError):
                        transcriber.load()
            self.assertEqual(2, len(attempts))
        finally:
            speech_transcriber.clear_model_cache()

    def test_successful_model_is_reused(self):
        speech_transcriber.clear_model_cache()
        instances = []

        class FakeModel:
            def __init__(self, *args, **kwargs):
                instances.append(kwargs)

        try:
            with mock.patch('faster_whisper.WhisperModel', FakeModel):
                for _ in range(2):
                    SpeechTranscriber(
                        model_size='debug-reuse', device='cpu',
                        compute_type='int8', cpu_threads=1,
                        autocorrect=False).load()
            self.assertEqual(1, len(instances))
            self.assertTrue(instances[0]['compute_type'] == 'int8')
        finally:
            speech_transcriber.clear_model_cache()

    def test_correction_runs_without_inference_lock(self):
        class Segment:
            text = 'превет'
            start = 0.0
            end = 1.0
            no_speech_prob = 0.0
            avg_logprob = 0.0

        class Model:
            def transcribe(self, audio, **kwargs):
                return [Segment()], {}

        transcriber = SpeechTranscriber(
            device='cpu', compute_type='int8', cpu_threads=1)
        transcriber.model = Model()
        transcriber.loaded = True
        calls = []

        def correct(text):
            calls.append(text)
            self.assertFalse(speech_transcriber._INFERENCE_LOCK.locked())
            return text

        with mock.patch.object(speech_transcriber, 'correct_speech_text', correct):
            result = transcriber.transcribe_segments(
                np.ones(16000, dtype=np.float32), sample_rate=16000)
        self.assertEqual(['превет'], calls)
        self.assertEqual('превет', result[0]['text'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
