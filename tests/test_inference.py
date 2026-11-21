"""Regression tests for NutriDish FoodPredictor contracts (lightweight)."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from model.inference import FOOD101_CLASSES, FoodPredictor


class _FakeTensor:
    """Minimal TF-tensor-like object supporting the ops used in _preprocess."""

    def __init__(self, arr):
        self.arr = np.asarray(arr, dtype=np.float32)

    def numpy(self):
        return self.arr

    def __truediv__(self, other):
        return _FakeTensor(self.arr / other)

    def __sub__(self, other):
        return _FakeTensor(self.arr - other)

    def __mul__(self, other):
        return _FakeTensor(self.arr * other)


def make_predictor() -> FoodPredictor:
    """Create a FoodPredictor without running heavy __init__ (no TF model load)."""
    predictor = FoodPredictor.__new__(FoodPredictor)
    predictor.tf = MagicMock()
    predictor.tf.image.resize.return_value = _FakeTensor(
        np.zeros((224, 224, 3), dtype=np.uint8)
    )
    predictor.tf.cast.return_value = _FakeTensor(
        np.zeros((224, 224, 3), dtype=np.float32)
    )
    predictor.tf.float32 = MagicMock()
    predictor.model = MagicMock()
    predictor.classes = FOOD101_CLASSES
    return predictor


class TestFoodPredictorContracts:
    def test_load_image_from_pil_returns_numpy_rgb(self):
        from PIL import Image

        predictor = make_predictor()
        img = Image.new("RGB", (300, 200), (255, 0, 0))
        arr = predictor._load_image(img)
        assert arr.shape == (200, 300, 3)
        assert arr.dtype == np.uint8

    def test_load_image_from_numpy_returns_array(self):
        predictor = make_predictor()
        arr = np.zeros((100, 200, 3), dtype=np.uint8)
        out = predictor._load_image(arr)
        assert out.shape == (100, 200, 3)

    def test_load_image_unsupported_type_raises(self):
        predictor = make_predictor()
        with pytest.raises(TypeError):
            predictor._load_image(12345)

    def test_preprocess_returns_batch_of_224(self):
        predictor = make_predictor()
        out = predictor._preprocess(np.zeros((224, 224, 3), dtype=np.uint8))
        assert out.shape == (1, 224, 224, 3)

    def test_predict_returns_expected_structure(self):
        predictor = make_predictor()
        probs = np.zeros(101, dtype=np.float32)
        probs[0] = 0.93
        predictor.model.predict.return_value = [probs]

        result = predictor.predict(
            np.zeros((224, 224, 3), dtype=np.uint8), top_k=3
        )
        assert set(result.keys()) == {"label", "confidence", "top_k"}
        assert result["label"] == FOOD101_CLASSES[0]
        assert result["confidence"] == pytest.approx(0.93)
        assert len(result["top_k"]) == 3
        for r in result["top_k"]:
            assert set(r.keys()) == {"label", "confidence"}

    def test_predict_top_k_respected(self):
        predictor = make_predictor()
        probs = np.zeros(101, dtype=np.float32)
        probs[0] = 0.93
        predictor.model.predict.return_value = [probs]

        result = predictor.predict(
            np.zeros((224, 224, 3), dtype=np.uint8), top_k=2
        )
        assert len(result["top_k"]) == 2
