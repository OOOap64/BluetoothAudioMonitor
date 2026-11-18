# -*- coding: utf-8 -*-
"""Тесты признаков, обучения и инференса классификатора звуков."""

import os
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.custom_sound_model import (
    CustomSoundModel, ModelError, load_audio, scan_dataset, train_model,
)
from src.sound_classifier import SoundClassifier, category_for_label
from src.sound_features import extract_features, feature_size


def write_wav(path, audio, rate=16000):
    samples = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    samples = (samples * 32767.0).astype(np.int16)
    with wave.open(str(path), 'wb') as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(rate))
        handle.writeframes(samples.tobytes())


def make_dataset(root, per_class=12, seconds=1.0, rate=16000):
    root = Path(root)
    rng = np.random.default_rng(1234)
    length = int(seconds * rate)
    t = np.arange(length) / rate
    generators = {
        'tone': lambda: 0.5 * np.sin(2 * np.pi * 440 * t),
        'noise': lambda: 0.3 * rng.normal(0, 1, length).astype(np.float32),
        'chirp': lambda: 0.5 * np.sin(
            2 * np.pi * (200 + 1800 * t / max(seconds, 1e-6)) * t),
    }
    for name, generator in generators.items():
        folder = root / name
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(per_class):
            audio = generator() * (0.8 + 0.4 * rng.random())
            audio = audio + 0.005 * rng.normal(0, 1, length)
            write_wav(folder / f'{name}-{index:03d}.wav', audio, rate)
    return root


class FeatureTests(unittest.TestCase):
    def test_feature_shape_and_determinism(self):
        rng = np.random.default_rng(7)
        audio = rng.normal(0, 0.1, 16000).astype(np.float32)
        first = extract_features(audio, 16000)
        second = extract_features(audio, 16000)
        self.assertEqual(first.shape, (feature_size(),))
        self.assertEqual(first.dtype, np.float32)
        np.testing.assert_allclose(first, second, rtol=1e-5, atol=1e-5)
        self.assertTrue(np.all(np.isfinite(first)))

    def test_empty_audio_still_returns_vector(self):
        vector = extract_features(np.zeros(0, dtype=np.float32), 16000)
        self.assertEqual(vector.shape, (feature_size(),))
        self.assertTrue(np.all(np.isfinite(vector)))

    def test_different_sounds_have_different_features(self):
        rate = 16000
        t = np.arange(rate) / rate
        tone = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        noise = np.random.default_rng(3).normal(0, 0.3, rate).astype(np.float32)
        self.assertGreater(
            float(np.linalg.norm(
                extract_features(tone, rate) - extract_features(noise, rate))),
            1.0)


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.dataset = make_dataset(cls.root / 'dataset')
        cls.model_path = cls.root / 'sound_model.npz'
        cls.report = train_model(
            cls.dataset, cls.model_path, epochs=90,
            test_size=0.25, augment=True, seed=5)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_dataset_scan(self):
        classes, _warnings = scan_dataset(self.dataset)
        self.assertEqual(sorted(classes), ['chirp', 'noise', 'tone'])
        self.assertEqual(sum(len(items) for items in classes.values()), 36)

    def test_model_is_saved_and_loads(self):
        self.assertTrue(self.model_path.is_file())
        model = CustomSoundModel(str(self.model_path))
        self.assertTrue(model.loaded)
        self.assertEqual(sorted(model.classes), ['chirp', 'noise', 'tone'])
        info = model.info()
        self.assertEqual(info['path'], str(self.model_path))

    def test_accuracy_is_useful_on_synthetic_data(self):
        self.assertGreaterEqual(self.report['val_accuracy'], 0.90)
        self.assertGreaterEqual(self.report['train_accuracy'], 0.90)
        self.assertEqual(len(self.report['classes']), 3)

    def test_predicts_tone(self):
        model = CustomSoundModel(str(self.model_path))
        rate = 16000
        t = np.arange(rate) / rate
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        predictions = model.predict_proba(audio, rate)
        self.assertEqual(predictions[0]['label'], 'tone')
        self.assertGreater(predictions[0]['score'], 0.5)
        self.assertAlmostEqual(
            sum(item['score'] for item in predictions), 1.0, places=4)

    def test_load_audio(self):
        audio, rate = load_audio(self.dataset / 'tone' / 'tone-000.wav')
        self.assertEqual(rate, 16000)
        self.assertGreater(audio.size, 0)
        self.assertEqual(audio.dtype, np.float32)

    def test_missing_model_raises(self):
        with self.assertRaises(ModelError):
            CustomSoundModel(str(self.root / 'missing.npz'))


class SoundClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.dataset = make_dataset(cls.root / 'dataset', per_class=8)
        cls.model_path = cls.root / 'sound_model.npz'
        train_model(cls.dataset, cls.model_path, epochs=60,
                    test_size=0.25, augment=False, seed=11)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_custom_model_is_used_without_audioset(self):
        classifier = SoundClassifier(
            custom_model_path=str(self.model_path), use_audioset=False)
        self.assertTrue(classifier.load())
        rate = 16000
        t = np.arange(rate) / rate
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        results = classifier.classify(audio, rate)
        self.assertTrue(results)
        self.assertEqual(results[0]['source'], 'custom')
        self.assertEqual(results[0]['label'], 'tone')

    def test_no_models_raises(self):
        classifier = SoundClassifier(
            custom_model_path=str(self.root / 'missing.npz'),
            use_audioset=False)
        with self.assertRaises(RuntimeError):
            classifier.load()

    def test_category_mapping(self):
        self.assertEqual(category_for_label('Speech'), 'speech')
        self.assertEqual(category_for_label('Screaming'), 'yelling')
        self.assertEqual(category_for_label('Bird vocalization, chirp'), 'bird_singing')
        self.assertEqual(category_for_label('Door slam'), 'door')
        self.assertEqual(category_for_label('Crackling fire'), 'fire_burning')
        self.assertEqual(category_for_label('unknown_object'), 'other')


if __name__ == '__main__':
    unittest.main()
