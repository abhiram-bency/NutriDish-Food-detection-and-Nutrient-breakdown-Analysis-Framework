"""
NutriDish — Inference Pipeline

Usage:
    from model.inference import FoodPredictor
    predictor = FoodPredictor("nutridish_model.keras")
    result = predictor.predict("path/to/image.jpg")
    # or
    result = predictor.predict(pil_image)
    # or
    result = predictor.predict(np_array)
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Union

# ─── Constants ────────────────────────────────────────────────────────────────

IMG_SIZE = 224

FOOD101_CLASSES = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
    "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
    "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
    "ceviche", "cheese_plate", "cheesecake", "chicken_curry", "chicken_quesadilla",
    "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
    "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
    "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict",
    "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras",
    "french_fries", "french_onion_soup", "french_toast", "fried_calamari", "fried_rice",
    "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich",
    "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup",
    "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna",
    "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup",
    "mussels", "nachos", "omelette", "onion_rings", "oysters",
    "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck",
    "pho", "pizza", "pork_chop", "poutine", "prime_rib",
    "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto",
    "samosa", "sashimi", "scallops", "seaweed_salad", "shrimp_and_grits",
    "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake",
    "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare", "waffles",
]


# ─── Predictor ────────────────────────────────────────────────────────────────

class FoodPredictor:
    """Loads the trained Keras model and runs inference on food images."""

    def __init__(self, model_path: str = "nutridish_model.keras"):
        import tensorflow as tf
        self.tf = tf

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model not found at '{model_path}'.\n"
                "Run `python model/train.py` first to train and save the model."
            )

        print(f"Loading model from {model_path} …")
        self.model = tf.keras.models.load_model(model_path)
        self.classes = FOOD101_CLASSES
        print("Model loaded ✓")

    # ── Image loading ──────────────────────────────────────────────────────────

    def _load_image(self, source: Union[str, "PIL.Image.Image", np.ndarray]) -> np.ndarray:
        """Accept file path, PIL image, or numpy array. Return (H,W,3) uint8 array."""
        from PIL import Image

        if isinstance(source, str) or isinstance(source, Path):
            img = Image.open(source).convert("RGB")
            return np.array(img, dtype=np.uint8)

        if hasattr(source, "convert"):           # PIL Image
            return np.array(source.convert("RGB"), dtype=np.uint8)

        if isinstance(source, np.ndarray):
            if source.ndim == 2:                 # Grayscale
                source = np.stack([source] * 3, axis=-1)
            return source[:, :, :3].astype(np.uint8)

        raise TypeError(f"Unsupported image type: {type(source)}")

    # ── Preprocessing ──────────────────────────────────────────────────────────

    def _preprocess(self, img_np: np.ndarray) -> np.ndarray:
        """Resize → normalize to [-1, 1] → add batch dim."""
        tf = self.tf
        img = tf.image.resize(img_np, [IMG_SIZE, IMG_SIZE])
        img = tf.cast(img, tf.float32) / 255.0
        img = (img - 0.5) * 2.0          # MobileNetV2 expected range
        return img.numpy()[np.newaxis, ...]  # (1, 224, 224, 3)

    # ── Predict ────────────────────────────────────────────────────────────────

    def predict(
        self,
        source: Union[str, "PIL.Image.Image", np.ndarray],
        top_k: int = 3,
    ) -> dict:
        """
        Predict food label from an image.

        Returns:
            {
                "label":      "pizza",
                "confidence": 0.93,
                "top_k": [
                    {"label": "pizza",       "confidence": 0.93},
                    {"label": "bruschetta",  "confidence": 0.04},
                    {"label": "garlic_bread","confidence": 0.02},
                ]
            }
        """
        img_np = self._load_image(source)
        tensor = self._preprocess(img_np)

        probs = self.model.predict(tensor, verbose=0)[0]   # (101,)

        top_indices = np.argsort(probs)[::-1][:top_k]
        top_results = [
            {"label": self.classes[i], "confidence": float(probs[i])}
            for i in top_indices
        ]

        return {
            "label": top_results[0]["label"],
            "confidence": top_results[0]["confidence"],
            "top_k": top_results,
        }


# ─── Quick CLI test ───────────────────────────────────────────────────────────

def _cli_test():
    import argparse

    parser = argparse.ArgumentParser(description="NutriDish — quick inference test")
    parser.add_argument("image", help="Path to a food image")
    parser.add_argument("--model", default="nutridish_model.keras", help="Path to .keras model")
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    predictor = FoodPredictor(args.model)
    result = predictor.predict(args.image, top_k=args.top_k)

    print(f"\n🍽️  Predicted food : {result['label']}")
    print(f"   Confidence      : {result['confidence']:.2%}")
    print("\n   Top predictions:")
    for r in result["top_k"]:
        bar = "█" * int(r["confidence"] * 40)
        print(f"   {r['label']:30s}  {r['confidence']:.2%}  {bar}")


if __name__ == "__main__":
    _cli_test()
