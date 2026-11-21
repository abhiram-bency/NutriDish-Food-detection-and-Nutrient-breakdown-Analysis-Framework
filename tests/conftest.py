"""Shared pytest fixtures for NutriDish regression tests."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.database import Base, get_db
from api.main import app
from model.inference import FoodPredictor


# ── Temporary in-memory SQLite database for tests ────────────────────────────

SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

test_engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=test_engine,
)
Base.metadata.create_all(bind=test_engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture
def client():
    """FastAPI TestClient without running lifespan/startup (no real model/DB)."""
    return TestClient(app)


@pytest.fixture
def mock_predictor():
    """Replace FoodPredictor with a lightweight mock (no TF model load)."""
    from api import main as api_main

    mock = MagicMock()
    mock.predict.return_value = {
        "label": "pizza",
        "confidence": 0.93,
        "top_k": [
            {"label": "pizza", "confidence": 0.93},
            {"label": "bruschetta", "confidence": 0.04},
            {"label": "garlic_bread", "confidence": 0.02},
        ],
    }
    api_main._predictor = mock
    yield mock
    api_main._predictor = None


@pytest.fixture
def sample_image_bytes():
    """Read the existing sample pizza image from the project tree."""
    path = Path(__file__).resolve().parent.parent / "test_images" / "pizza.jpg"
    assert path.exists(), f"Sample image not found: {path}"
    return path.read_bytes()
