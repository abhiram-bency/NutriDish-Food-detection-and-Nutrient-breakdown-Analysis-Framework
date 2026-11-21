"""Regression tests for the existing NutriDish nutrition lookup engine."""

import pytest

from data.nutrition_db import (
    DEFAULT_NUTRITION,
    FOOD101_CLASSES,
    NUTRITION_DB,
    get_nutrition,
)


class TestNutritionDatabaseBasics:
    """Verify the shape and size of the existing nutrition database."""

    def test_nutrition_db_contains_101_foods(self):
        """The DB is expected to contain 101 Food-101 entries."""
        assert len(NUTRITION_DB) == 101

    def test_food101_classes_sorted_and_complete(self):
        """Class list should be sorted and contain all 101 labels."""
        assert len(FOOD101_CLASSES) == 101
        assert FOOD101_CLASSES == sorted(FOOD101_CLASSES)


class TestNutritionLookup:
    """Regression tests for get_nutrition() against current values."""

    def test_known_food_pizza_values(self):
        n = get_nutrition("pizza")
        assert n["calories"] == pytest.approx(266.0, abs=0.5)
        assert n["protein"] == pytest.approx(11.0, abs=0.5)
        assert n["carbs"] == pytest.approx(33.0, abs=0.5)
        assert n["fat"] == pytest.approx(10.0, abs=0.5)

    def test_known_food_hamburger_values(self):
        n = get_nutrition("hamburger")
        assert n["calories"] == pytest.approx(295.0, abs=0.5)
        assert n["protein"] == pytest.approx(17.0, abs=0.5)
        assert n["carbs"] == pytest.approx(30.0, abs=0.5)
        assert n["fat"] == pytest.approx(14.0, abs=0.5)

    def test_known_food_chicken_curry_values(self):
        n = get_nutrition("chicken_curry")
        assert n["calories"] == pytest.approx(175.0, abs=0.5)
        assert n["protein"] == pytest.approx(16.0, abs=0.5)
        assert n["carbs"] == pytest.approx(8.0, abs=0.5)
        assert n["fat"] == pytest.approx(9.0, abs=0.5)

    def test_get_nutrition_returns_required_fields(self):
        n = get_nutrition("pizza")
        assert set(n.keys()) == {"calories", "protein", "carbs", "fat"}

    def test_label_normalization_uppercase(self):
        n = get_nutrition("Pizza")
        assert n["calories"] == pytest.approx(266.0, abs=0.5)

    def test_label_normalization_spaces_to_underscores(self):
        n = get_nutrition("chicken curry")
        assert n["calories"] == pytest.approx(175.0, abs=0.5)

    def test_unknown_food_fallback(self):
        n = get_nutrition("definitely_not_a_food_xyz")
        assert n == DEFAULT_NUTRITION

    def test_per_100g_semantics(self):
        n = get_nutrition("pizza")
        assert n["calories"] > 0
        assert n["protein"] > 0
        assert n["carbs"] > 0
        assert n["fat"] > 0
