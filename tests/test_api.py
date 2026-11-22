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

    def test_analyze_image_defaults_to_100g(self, client, mock_predictor, sample_image_bytes):
        r = client.post(
            "/analyze-image",
            files={"file": ("pizza.jpg", sample_image_bytes, "image/jpeg")},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["serving_g"] == 100.0
        assert data["food"] == "pizza"
        assert data["confidence"] == pytest.approx(0.93)
        assert data["calories"] == pytest.approx(266.0, abs=0.5)
        assert len(data["top_predictions"]) == 3

    def test_analyze_image_explicit_100g_matches_existing(self, client, mock_predictor, sample_image_bytes):
        r = client.post(
            "/analyze-image",
            files={"file": ("pizza.jpg", sample_image_bytes, "image/jpeg")},
            data={"serving_g": "100"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["serving_g"] == 100.0
        assert data["calories"] == pytest.approx(266.0, abs=0.5)

    def test_analyze_image_50g_produces_half_values(self, client, mock_predictor, sample_image_bytes):
        r = client.post(
            "/analyze-image",
            files={"file": ("pizza.jpg", sample_image_bytes, "image/jpeg")},
            data={"serving_g": "50"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["serving_g"] == 50.0
        assert data["calories"] == pytest.approx(133.0, abs=0.5)
        assert data["protein"] == pytest.approx(5.5, abs=0.5)
        assert data["carbs"] == pytest.approx(16.5, abs=0.5)
        assert data["fat"] == pytest.approx(5.0, abs=0.5)

    def test_analyze_image_150g_produces_1_5x_values(self, client, mock_predictor, sample_image_bytes):
        r = client.post(
            "/analyze-image",
            files={"file": ("pizza.jpg", sample_image_bytes, "image/jpeg")},
            data={"serving_g": "150"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["serving_g"] == 150.0
        assert data["calories"] == pytest.approx(399.0, abs=0.5)
        assert data["protein"] == pytest.approx(16.5, abs=0.5)

    def test_analyze_image_fractional_serving(self, client, mock_predictor, sample_image_bytes):
        r = client.post(
            "/analyze-image",
            files={"file": ("pizza.jpg", sample_image_bytes, "image/jpeg")},
            data={"serving_g": "175.5"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["serving_g"] == 175.5
        assert data["calories"] == pytest.approx(466.83, abs=0.5)

    def test_analyze_image_zero_serving_returns_422(self, client, mock_predictor, sample_image_bytes):
        r = client.post(
            "/analyze-image",
            files={"file": ("pizza.jpg", sample_image_bytes, "image/jpeg")},
            data={"serving_g": "0"},
        )
        assert r.status_code == 422

    def test_analyze_image_negative_serving_returns_422(self, client, mock_predictor, sample_image_bytes):
        r = client.post(
            "/analyze-image",
            files={"file": ("pizza.jpg", sample_image_bytes, "image/jpeg")},
            data={"serving_g": "-10"},
        )
        assert r.status_code == 422

    def test_analyze_image_nan_serving_returns_422(self, client, mock_predictor, sample_image_bytes):
        r = client.post(
            "/analyze-image",
            files={"file": ("pizza.jpg", sample_image_bytes, "image/jpeg")},
            data={"serving_g": "nan"},
        )
        assert r.status_code == 422

    def test_analyze_image_infinity_serving_returns_422(self, client, mock_predictor, sample_image_bytes):
        r = client.post(
            "/analyze-image",
            files={"file": ("pizza.jpg", sample_image_bytes, "image/jpeg")},
            data={"serving_g": "inf"},
        )
        assert r.status_code == 422

    def test_analyze_image_non_numeric_serving_returns_422(self, client, mock_predictor, sample_image_bytes):
        r = client.post(
            "/analyze-image",
            files={"file": ("pizza.jpg", sample_image_bytes, "image/jpeg")},
            data={"serving_g": "abc"},
        )
        assert r.status_code == 422

    def test_analyze_image_empty_serving_returns_422(self, client, mock_predictor, sample_image_bytes):
        boundary = "xBoundary123"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="pizza.jpg"\r\n'
            f"Content-Type: image/jpeg\r\n\r\n"
        ).encode() + sample_image_bytes + b"\r\n"
        body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"serving_g\"\r\n\r\n\r\n".encode() + b"\r\n"
        body += f"--{boundary}--\r\n".encode()
        r = client.post(
            "/analyze-image",
            content=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        assert r.status_code == 422

    def test_analyze_image_boolean_like_serving_returns_422(self, client, mock_predictor, sample_image_bytes):
        r = client.post(
            "/analyze-image",
            files={"file": ("pizza.jpg", sample_image_bytes, "image/jpeg")},
            data={"serving_g": "true"},
        )
        assert r.status_code == 422

    def test_analyze_image_missing_image_returns_422(self, client, mock_predictor):
        r = client.post("/analyze-image")
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

    def test_log_food_stores_scaled_values_once(self, client):
        # Clear prior entries in this isolated in-memory database.
        existing = client.get("/logs")
        for log in existing.json()["logs"]:
            client.delete(f"/logs/{log['id']}")

        payload = {
            "food": "pizza",
            "confidence": 0.93,
            "calories": 133.0,
            "protein": 66.5,
            "carbs": 16.5,
            "fat": 5.0,
        }
        r = client.post("/log-food", json=payload)
        assert r.status_code == 201
        assert r.json()["calories"] == pytest.approx(133.0, abs=0.5)

        r = client.get("/daily-summary")
        data = r.json()
        assert data["meals"] == 1
        assert data["total_calories"] == pytest.approx(133.0, abs=0.5)

        r = client.get("/logs")
        data = r.json()
        assert data["count"] == 1

    def test_delete_log_not_found(self, client):
        r = client.delete("/logs/9999")
        assert r.status_code == 404
