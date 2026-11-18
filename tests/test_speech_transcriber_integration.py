# -*- coding: utf-8 -*-
"""Интеграционные тесты автокоррекции в конвейере распознавания речи.

Whisper-модель не загружается: вместо неё используется маленькая заглушка,
которая возвращает заранее заданные сегменты. Так проверяется именно то,
что текст проходит через автокорректор во всех ветках.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.speech_transcriber import SpeechTranscriber


class FakeSegment:
    def __init__(self, text, start=0.0, end=1.0):
        self.text = text
        self.start = start
        self.end = end


class FakeModel:
    def __init__(self, texts):
        self.texts = texts
        self.last_kwargs = None

    def transcribe(self, *args, **kwargs):
        self.last_kwargs = kwargs
        return (
            [FakeSegment(text, index, index + 1)
             for index, text in enumerate(self.texts)],
            {},
        )


class SpeechTranscriberAutocorrectTests(unittest.TestCase):
    def make_transcriber(self, texts, autocorrect=True, phrases=None):
        transcriber = SpeechTranscriber(
            language='ru', autocorrect=autocorrect, phrases=phrases)
        transcriber.model = FakeModel(texts)
        transcriber.loaded = True
        return transcriber

    def test_segments_are_corrected(self):
        transcriber = self.make_transcriber(['превет', 'канешно щас'])
        segments = transcriber.transcribe_segments(
            np.ones(16000, dtype=np.float32), sample_rate=16000)
        self.assertEqual(
            ['Привет.', 'Конечно, сейчас.'],
            [segment['text'] for segment in segments],
        )

    def test_transcribe_corrects_once_and_keeps_single_sentence(self):
        transcriber = self.make_transcriber(['превет', 'канешно щас'])
        text = transcriber.transcribe(
            np.ones(16000, dtype=np.float32), sample_rate=16000)
        self.assertEqual('Привет, конечно сейчас.', text)

    def test_autocorrect_can_be_disabled(self):
        transcriber = self.make_transcriber(
            ['превет', 'канешно щас'], autocorrect=False)
        text = transcriber.transcribe(
            np.ones(16000, dtype=np.float32), sample_rate=16000)
        self.assertEqual('превет канешно щас', text)

    def test_per_call_override(self):
        transcriber = self.make_transcriber(['превет'], autocorrect=False)
        text = transcriber.transcribe(
            np.ones(16000, dtype=np.float32),
            sample_rate=16000,
            autocorrect=True,
        )
        self.assertEqual('Привет.', text)

    def test_segments_can_disable_correction_for_one_call(self):
        transcriber = self.make_transcriber(['превет'], autocorrect=True)
        segments = transcriber.transcribe_segments(
            np.ones(16000, dtype=np.float32),
            sample_rate=16000,
            autocorrect=False,
        )
        self.assertEqual(['превет'], [s['text'] for s in segments])

    def test_user_phrases_are_passed_as_prompt_and_hotwords(self):
        phrases = ['Открой окно и включи свет', 'Привет, меня зовут']
        transcriber = self.make_transcriber(['привет'], phrases=phrases)
        transcriber.transcribe_segments(
            np.ones(16000, dtype=np.float32), sample_rate=16000)
        kwargs = transcriber.model.last_kwargs
        self.assertIn('Открой окно и включи свет', kwargs['initial_prompt'])
        self.assertEqual('; '.join(phrases), kwargs['hotwords'])

    def test_phrases_can_be_multiline_string(self):
        transcriber = self.make_transcriber(
            ['привет'], phrases='Первая фраза\n\nВторая фраза\n')
        self.assertEqual(
            ['Первая фраза', 'Вторая фраза'], transcriber.phrases)

    def test_explicit_prompt_overrides_phrase_prompt(self):
        transcriber = self.make_transcriber(
            ['привет'], phrases=['Открой окно'])
        transcriber.transcribe_segments(
            np.ones(16000, dtype=np.float32), sample_rate=16000,
            initial_prompt='Свой prompt', hotwords='Слово')
        kwargs = transcriber.model.last_kwargs
        self.assertEqual('Свой prompt', kwargs['initial_prompt'])
        self.assertEqual('Слово', kwargs['hotwords'])

    def test_empty_phrases_disable_hints(self):
        transcriber = self.make_transcriber(['привет'], phrases='')
        transcriber.transcribe_segments(
            np.ones(16000, dtype=np.float32), sample_rate=16000)
        kwargs = transcriber.model.last_kwargs
        self.assertNotIn('initial_prompt', kwargs)
        self.assertNotIn('hotwords', kwargs)

    def test_empty_audio_returns_empty(self):
        transcriber = self.make_transcriber(['превет'])
        self.assertEqual(
            [], transcriber.transcribe_segments(
                np.zeros(0, dtype=np.float32), sample_rate=16000))
        self.assertEqual(
            '', transcriber.transcribe(
                np.zeros(0, dtype=np.float32), sample_rate=16000))


if __name__ == '__main__':
    unittest.main(verbosity=2)
