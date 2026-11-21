"""Regression tests for the existing NutriDish nutrition lookup engine."""

import pytest

from data.nutrition_db import (
    DEFAULT_NUTRITION,
    FOOD101_CLASSES,
    NUTRITION_DB,
    get_nutrition,
    scale_nutrition,
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


class TestServingScaling:
    """Regression tests for the serving-aware nutrition scaling function."""

    def test_100g_serving_returns_baseline_values(self):
        scaled = scale_nutrition(get_nutrition("pizza"), 100)
        assert scaled["calories"] == pytest.approx(266.0, abs=0.5)
        assert scaled["protein"] == pytest.approx(11.0, abs=0.5)
        assert scaled["carbs"] == pytest.approx(33.0, abs=0.5)
        assert scaled["fat"] == pytest.approx(10.0, abs=0.5)

    def test_50g_serving_scales_to_half(self):
        scaled = scale_nutrition(get_nutrition("pizza"), 50)
        assert scaled["calories"] == pytest.approx(133.0, abs=0.5)
        assert scaled["protein"] == pytest.approx(5.5, abs=0.5)
        assert scaled["carbs"] == pytest.approx(16.5, abs=0.5)
        assert scaled["fat"] == pytest.approx(5.0, abs=0.5)

    def test_200g_serving_scales_to_double(self):
        scaled = scale_nutrition(get_nutrition("pizza"), 200)
        assert scaled["calories"] == pytest.approx(532.0, abs=0.5)
        assert scaled["protein"] == pytest.approx(22.0, abs=0.5)
        assert scaled["carbs"] == pytest.approx(66.0, abs=0.5)
        assert scaled["fat"] == pytest.approx(20.0, abs=0.5)

    def test_fractional_serving_weights(self):
        scaled = scale_nutrition(get_nutrition("pizza"), 150)
        assert scaled["calories"] == pytest.approx(399.0, abs=0.5)
        assert scaled["protein"] == pytest.approx(16.5, abs=0.5)

    def test_invalid_zero_serving_raises(self):
        with pytest.raises(ValueError):
            scale_nutrition(get_nutrition("pizza"), 0)

    def test_negative_serving_raises(self):
        with pytest.raises(ValueError):
            scale_nutrition(get_nutrition("pizza"), -50)

    def test_nan_and_infinity_serving_raise(self):
        with pytest.raises(ValueError):
            scale_nutrition(get_nutrition("pizza"), float("nan"))
        with pytest.raises(ValueError):
            scale_nutrition(get_nutrition("pizza"), float("inf"))

    def test_boolean_serving_raises(self):
        with pytest.raises(ValueError):
            scale_nutrition(get_nutrition("pizza"), True)

    def test_non_numeric_serving_raises(self):
        with pytest.raises(ValueError):
            scale_nutrition(get_nutrition("pizza"), "150")

    def test_original_nutrition_dict_not_mutated(self):
        original = get_nutrition("pizza")
        snapshot = dict(original)
        scaled = scale_nutrition(original, 200)
        assert original == snapshot
        assert scaled is not original

    def test_unknown_food_behavior_remains_unchanged(self):
        assert get_nutrition("definitely_not_a_food_xyz") == DEFAULT_NUTRITION
        scaled_default = scale_nutrition(DEFAULT_NUTRITION, 100)
        assert scaled_default["calories"] == pytest.approx(200.0, abs=0.5)

    def test_missing_optional_nutrients_remain_none(self):
        scaled = scale_nutrition(get_nutrition("pizza"), 100)
        assert scaled.get("fiber") is None
        assert scaled.get("sugar") is None
        assert scaled.get("sodium") is None

    def test_sodium_units_remain_milligrams(self):
        nutrition = {"calories": 100, "protein": 1, "carbs": 2, "fat": 3, "sodium": 400.0}
        scaled_100g = scale_nutrition(nutrition, 100)
        scaled_200g = scale_nutrition(nutrition, 200)
        assert scaled_100g["sodium"] == pytest.approx(400.0)  # mg
        assert scaled_200g["sodium"] == pytest.approx(800.0)  # mg

    def test_existing_nutrition_lookup_still_works(self):
        n = get_nutrition("pizza")
        assert n["calories"] == pytest.approx(266.0, abs=0.5)
