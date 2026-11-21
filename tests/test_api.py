"""Regression tests for NutriDish FastAPI endpoints (mocked deps)."""

import pytest
from fastapi.testclient import TestClient

from api.main import app


EXPECTED_ROOT = {"status": "ok", "service": "NutriDish API", "version": "1.0.0"}


class TestHealthEndpoints:
    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert r.json() == EXPECTED_ROOT

    def test_health(self, client, mock_predictor):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True


class TestAnalyzeImage:
    def test_analyze_image_success(self, client, mock_predictor, sample_image_bytes):
        r = client.post(
            "/analyze-image",
            files={"file": ("pizza.jpg", sample_image_bytes, "image/jpeg")},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["food"] == "pizza"
        assert data["confidence"] == pytest.approx(0.93)
        assert data["calories"] == pytest.approx(266.0, abs=0.5)
        assert data["protein"] == pytest.approx(11.0, abs=0.5)
        assert data["carbs"] == pytest.approx(33.0, abs=0.5)
        assert data["fat"] == pytest.approx(10.0, abs=0.5)
        assert len(data["top_predictions"]) == 3

    def test_analyze_image_rejects_non_image(self, client, mock_predictor):
        r = client.post(
            "/analyze-image",
            files={"file": ("notes.txt", b"hello world", "text/plain")},
        )
        assert r.status_code == 400

    def test_analyze_image_rejects_invalid_image(self, client, mock_predictor):
        r = client.post(
            "/analyze-image",
            files={"file": ("bad.jpg", b"not-an-image", "image/jpeg")},
        )
        assert r.status_code == 422


class TestFoodLogging:
    def test_log_food_creates_entry_and_summary(self, client):
        payload = {
            "food": "pizza",
            "confidence": 0.93,
            "calories": 266.0,
            "protein": 11.0,
            "carbs": 33.0,
            "fat": 10.0,
        }
        r = client.post("/log-food", json=payload)
        assert r.status_code == 201
        body = r.json()
        assert body["food"] == "pizza"
        assert body["calories"] == pytest.approx(266.0, abs=0.5)

        r = client.get("/daily-summary")
        data = r.json()
        assert data["meals"] == 1
        assert data["total_calories"] == pytest.approx(266.0, abs=0.5)

        r = client.get("/logs")
        data = r.json()
        assert data["count"] == 1

    def test_delete_log_not_found(self, client):
        r = client.delete("/logs/9999")
        assert r.status_code == 404
