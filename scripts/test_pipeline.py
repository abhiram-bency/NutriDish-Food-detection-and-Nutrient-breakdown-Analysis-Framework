"""
NutriDish — Test Script
Tests the full pipeline:
  1. Nutrition DB lookup
  2. Inference (if model exists)
  3. API request examples (printed as curl commands)

Run:
    python scripts/test_pipeline.py [--image path/to/food.jpg]
"""

import sys
import os
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.nutrition_db import get_nutrition, NUTRITION_DB


def test_nutrition_db():
    print("=" * 55)
    print("  1. Nutrition Database")
    print("=" * 55)
    samples = ["pizza", "hamburger", "sushi", "ramen", "tiramisu"]
    for food in samples:
        n = get_nutrition(food)
        print(f"  {food:<25} cal={n['calories']:>4}  prot={n['protein']:>4}g  "
              f"carbs={n['carbs']:>4}g  fat={n['fat']:>4}g")
    print(f"\n  Total foods in DB: {len(NUTRITION_DB)}")


def test_inference(image_path: str = None, model_path: str = "nutridish_model.keras"):
    print("\n" + "=" * 55)
    print("  2. Model Inference")
    print("=" * 55)

    if not os.path.exists(model_path):
        print(f"  ⚠️  Model not found at '{model_path}'.")
        print("  Run `python model/train.py` to train the model first.")
        return

    from model.inference import FoodPredictor
    predictor = FoodPredictor(model_path)

    if image_path:
        print(f"\n  Testing with: {image_path}")
        result = predictor.predict(image_path, top_k=5)
        print(f"\n  🍽️  Predicted : {result['label']}")
        print(f"      Confidence : {result['confidence']:.2%}\n")
        print("  Top 5 predictions:")
        for p in result["top_k"]:
            bar = "█" * int(p["confidence"] * 40)
            print(f"    {p['label']:<30} {p['confidence']:.2%}  {bar}")
    else:
        print("  No image path provided. Pass --image path/to/food.jpg")


def print_api_examples():
    print("\n" + "=" * 55)
    print("  3. Example API Calls (curl)")
    print("=" * 55)

    examples = [
        {
            "desc": "Analyze a food image",
            "cmd": 'curl -X POST "http://localhost:8000/analyze-image" \\\n'
                   '  -H "accept: application/json" \\\n'
                   '  -F "file=@pizza.jpg"',
            "response": json.dumps({
                "food": "pizza",
                "confidence": 0.9312,
                "calories": 266,
                "protein": 11,
                "carbs": 33,
                "fat": 10,
                "top_predictions": [
                    {"label": "pizza", "confidence": 0.9312},
                    {"label": "bruschetta", "confidence": 0.0421},
                    {"label": "garlic_bread", "confidence": 0.0198}
                ]
            }, indent=2),
        },
        {
            "desc": "Log a food entry",
            "cmd": 'curl -X POST "http://localhost:8000/log-food" \\\n'
                   '  -H "Content-Type: application/json" \\\n'
                   '  -d \'{"food":"pizza","calories":266,"protein":11,"carbs":33,"fat":10}\'',
            "response": json.dumps({
                "id": 1,
                "food": "pizza",
                "calories": 266,
                "protein": 11,
                "carbs": 33,
                "fat": 10,
                "log_date": "2025-07-15",
                "logged_at": "2025-07-15T13:42:00",
                "notes": None
            }, indent=2),
        },
        {
            "desc": "Get today's summary",
            "cmd": 'curl "http://localhost:8000/daily-summary"',
            "response": json.dumps({
                "date": "2025-07-15",
                "meals": 3,
                "total_calories": 712.0,
                "total_protein": 42.0,
                "total_carbs": 89.0,
                "total_fat": 28.0
            }, indent=2),
        },
        {
            "desc": "List all logs for a date",
            "cmd": 'curl "http://localhost:8000/logs?target_date=2025-07-15"',
            "response": "(returns array of log entries)",
        },
    ]

    for ex in examples:
        print(f"\n  ── {ex['desc']} ──")
        print(f"  $ {ex['cmd']}")
        print(f"\n  Response:")
        for line in ex["response"].splitlines():
            print(f"    {line}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NutriDish pipeline test")
    parser.add_argument("--image", help="Path to a food image for inference test")
    parser.add_argument("--model", default="nutridish_model.keras")
    args = parser.parse_args()

    test_nutrition_db()
    test_inference(args.image, args.model)
    print_api_examples()

    print("\n✅  Pipeline test complete.\n")
