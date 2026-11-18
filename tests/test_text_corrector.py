# -*- coding: utf-8 -*-
"""Тесты автокорректировки распознанной речи."""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.text_corrector as text_corrector
from src.text_corrector import SpeechTextCorrector, correct_speech_text


class SpeechTextCorrectorTests(unittest.TestCase):
    def setUp(self):
        self.corrector = SpeechTextCorrector()

    def assertCorrected(self, source, expected, **kwargs):
        actual = self.corrector.correct(source, **kwargs)
        self.assertEqual(expected, actual, msg=f'input={source!r}')

    def test_empty_values(self):
        self.assertEqual('', self.corrector.correct(None))
        self.assertEqual('', self.corrector.correct(''))
        self.assertEqual('   ', self.corrector.correct('   '))

    def test_typical_asr_errors(self):
        self.assertCorrected(
            'превет канешно щас всё будет',
            'Привет, конечно сейчас всё будет.')
        self.assertCorrected('ничо не понел', 'Ничего не понял.')
        self.assertCorrected('шо такоэ', 'Что такое.')
        self.assertCorrected('по русски', 'По-русски.')
        self.assertCorrected('впринципе нормально', 'В принципе нормально.')
        self.assertCorrected('досвидания', 'До свидания.')
        self.assertCorrected('незнаю', 'Не знаю.')
        self.assertCorrected('нехочу', 'Не хочу.')
        self.assertCorrected(
            'мне кажеться чтото не так',
            'Мне кажется что-то не так.')
        self.assertCorrected('привет как дела', 'Привет, как дела.')

    def test_timestamps_are_preserved(self):
        self.assertCorrected('[00:01:02] превет', '[00:01:02] Привет.')
        self.assertCorrected(
            '[00:01] превет\n[00:02] канешно',
            '[00:01] Привет.\n[00:02] Конечно.')
        self.assertCorrected('[01:02]', '[01:02]')

    def test_punctuation_and_spaces(self):
        self.assertCorrected('привет , как дела', 'Привет, как дела.')
        self.assertCorrected(
            'привет,мир!как дела?',
            'Привет, мир! Как дела?')
        self.assertCorrected('я   не   знаю', 'Я не знаю.')
        self.assertCorrected('это это очень в в важно', 'Это очень в важно.')

    def test_introductory_words(self):
        self.assertCorrected('наверно да', 'Наверное, да.')
        self.assertCorrected('короче я приду', 'Короче, я приду.')
        self.assertCorrected(
            'конечно это правда', 'Конечно, это правда.')
        self.assertCorrected('спасибо большое', 'Спасибо большое.')

    def test_names_and_acronyms_are_not_broken(self):
        self.assertCorrected('Мария Ивановна', 'Мария Ивановна.')
        self.assertCorrected('я Мария', 'Я Мария.')
        self.assertCorrected('ИИ анализирует звук', 'ИИ анализирует звук.')

    def test_abbreviations(self):
        self.assertCorrected('и т.д. и т.п.', 'И т.д. и т.п.')
        self.assertCorrected('т.е. это то же самое', 'Т.е. это то же самое.')
        self.assertCorrected('см. выше', 'См. выше.')

    def test_quotes(self):
        self.assertCorrected('"привет"', '"Привет."')
        self.assertCorrected('«привет»', '"Привет."')

    def test_urls_and_emails_are_not_broken(self):
        self.assertCorrected(
            'сайт https://example.com работает',
            'Сайт https://example.com работает.')
        self.assertCorrected(
            'почта test@example.com',
            'Почта test@example.com.')

    def test_partial_mode_has_no_final_period(self):
        self.assertCorrected(
            'привет как', 'Привет, как', add_final_period=False)
        self.assertCorrected(
            'ничо не понел', 'Ничего не понял', add_final_period=False)

    def test_idempotence(self):
        samples = [
            'превет канешно щас всё будет',
            '[00:01:02] превет',
            'и т.д. и т.п.',
            'привет,мир!как дела?',
            'наверно да',
            'т.е. это то же самое',
            '"привет"',
            'сайт https://example.com работает',
            'незнаю чтото',
            'мне кажеться чтото не так',
        ]
        for sample in samples:
            once = self.corrector.correct(sample)
            twice = self.corrector.correct(once)
            self.assertEqual(once, twice, msg=f'not idempotent: {sample!r}')

    def test_disabled_returns_source(self):
        source = 'превет   канешно'
        self.assertEqual(
            source, correct_speech_text(source, enabled=False))
        self.assertEqual('', correct_speech_text(None, enabled=False))

    def test_repeated_text_is_cached(self):
        text_corrector.clear_text_correction_cache()
        with mock.patch.object(
                text_corrector._DEFAULT_CORRECTOR, 'correct',
                wraps=text_corrector._DEFAULT_CORRECTOR.correct) as spy:
            first = correct_speech_text('превет канешно')
            second = correct_speech_text('превет канешно')
        self.assertEqual(first, second)
        self.assertEqual(1, spy.call_count)
        text_corrector.clear_text_correction_cache()

    def test_cached_result_matches_uncached(self):
        samples = [
            'превет канешно щас всё будет',
            '[00:01:02] превет',
            'привет,мир!как дела?',
            'сайт https://example.com работает',
        ]
        for sample in samples:
            text_corrector.clear_text_correction_cache()
            expected = correct_speech_text(sample)
            cached = correct_speech_text(sample)
            self.assertEqual(expected, cached)
            text_corrector.clear_text_correction_cache()


if __name__ == '__main__':
    unittest.main(verbosity=2)
