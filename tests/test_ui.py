"""Static regression tests for the NutriDish serving-aware web UI contract."""

import re
from pathlib import Path

UI_PATH = Path(__file__).resolve().parent.parent / "ui" / "index.html"

SERVING_INPUT_RE = re.compile(r'<input[^>]*id="serving-input"[^>]*>')


def _load_ui() -> str:
    return UI_PATH.read_text(encoding="utf-8")


class TestServingWeightInput:
    def test_serving_weight_input_exists(self):
        html = _load_ui()
        assert 'id="serving-input"' in html
        assert 'type="number"' in html
        assert 'Serving weight (g)' in html

    def test_serving_weight_input_defaults_to_100g(self):
        html = _load_ui()
        m = SERVING_INPUT_RE.search(html)
        assert m is not None, 'expected <input id="serving-input"> in ui/index.html'
        tag = m.group(0)
        assert 'type="number"' in tag
        assert 'value="100"' in tag
        assert 'step="0.1"' in tag

    def test_serving_weight_input_allows_fractional_values(self):
        html = _load_ui()
        m = SERVING_INPUT_RE.search(html)
        assert m is not None
        tag = m.group(0)
        assert 'step="0.1"' in tag or 'step="any"' in tag


class TestServingWeightSubmission:
    def test_analyze_appends_serving_g_to_formdata(self):
        html = _load_ui()
        assert "formData.append('file', currentFile);" in html
        assert "formData.append('serving_g', servingValue);" in html

    def test_analyze_validates_serving_client_side(self):
        html = _load_ui()
        assert 'const servingNum = parseFloat(servingRaw);' in html
        assert 'servingNum > 0' in html
        assert 'Number.isNaN' in html
        assert 'Please enter a valid serving weight' in html


class TestNutritionDisplay:
    def test_show_results_uses_backend_serving_and_values(self):
        html = _load_ui()
        assert 'data.serving_g != null ? data.serving_g : 100' in html
        assert "document.getElementById('result-serving').textContent" in html
        assert "document.getElementById('val-cal').textContent = data.calories;" in html

    def test_backend_nutrition_not_rescaled_in_js(self):
        html = _load_ui()
        assert 'textContent = data.calories' in html
        assert 'textContent = data.protein' in html
        assert 'textContent = data.carbs' in html
        assert 'textContent = data.fat' in html
        assert '* servingWeight' not in html
        assert '* servingValue' not in html


class TestLoggingUsesAnalyzedValues:
    def test_log_uses_analyzed_nutrition_values(self):
        html = _load_ui()
        assert 'currentResult.food' in html
        assert 'currentResult.confidence' in html
        assert 'currentResult.calories' in html
        assert 'currentResult.protein' in html
        assert 'currentResult.carbs' in html
        assert 'currentResult.fat' in html


class TestStaleResultPrevention:
    def test_image_change_invalidates_result(self):
        html = _load_ui()
        assert 'currentResult = null;' in html

    def test_serving_change_invalidates_result(self):
        html = _load_ui()
        assert 'if (currentResult) currentResult = null;' in html


class TestEstimateLabel:
    def test_results_labeled_as_estimate(self):
        html = _load_ui()
        assert 'AI-generated estimate' in html
