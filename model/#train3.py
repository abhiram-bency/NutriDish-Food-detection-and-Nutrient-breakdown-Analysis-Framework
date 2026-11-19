"""
NutriDish — ML Training Pipeline
Uses Food-101 via Hugging Face streaming + MobileNetV2 transfer learning.
successful tarining but validation accuracy -low

problem -Phase 1 (epochs 1–8): Training accuracy goes 62% → 46% → 37% → 29% → 35% → 61% → 51% → 56% — wildly oscillating, never converging. This is not normal.
Phase 2 (epochs 9–20): Accuracy collapses to near 0% and stays there. The model has essentially catastrophically forgotten everything.

Run:
    python model/train.py
"""

import os
import gc
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, mixed_precision

# ─── GPU / Memory setup ───────────────────────────────────────────────────────

mixed_precision.set_global_policy("mixed_float16")

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU Enabled:", gpus)
    except RuntimeError as e:
        print(e)

# ─── Config ───────────────────────────────────────────────────────────────────

IMG_SIZE          = 224
BATCH_SIZE        = 32       # keep 16 for RTX 3050 4 GB VRAM
EPOCHS            = 20        # increase to 20 for full training
NUM_CLASSES       = 101
LEARNING_RATE     = 1e-3
FINE_TUNE_LR      = 1e-5
FINE_TUNE_AT      = 100      # unfreeze MobileNetV2 from this layer onward
STEPS_PER_EPOCH   = 500       # 500 × 32 = 16,000 samples/epoch
VAL_STEPS         = 200      # 200 × 32 = 6,400 samples, ~63 per class — stable metrics
MODEL_SAVE_PATH   = "nutridish_model.keras"

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

# ─── Dataset ──────────────────────────────────────────────────────────────────

def preprocess_image(image, label):
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    image = (image - 0.5) * 2.0          # MobileNetV2 expects [-1, 1]
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label


def augment_image(image, label):
    image, label = preprocess_image(image, label)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image, label


def build_tf_dataset(split: str, augment: bool = False, batch_size: int = BATCH_SIZE):
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install: pip install datasets Pillow")
        sys.exit(1)

    print(f"  Loading Food-101 [{split}] in streaming mode …")
    hf_ds = load_dataset("food101", split=split, streaming=True)

    # Shuffle both splits: validation data arrives class-sorted from HuggingFace.
    # Without shuffling, VAL_STEPS only samples the first N classes → accuracy = 0.
    # Use a larger buffer for validation since we don't need randomness for generalisation,
    # just class diversity within the steps we evaluate.
    shuffle_buf = 1000 if split == "train" else 3000
    hf_ds = hf_ds.shuffle(seed=42, buffer_size=shuffle_buf)

    def generator():
        for sample in hf_ds:
            try:
                img_np = np.array(sample["image"].convert("RGB"), dtype=np.uint8)
                yield img_np, sample["label"]
            except Exception:
                continue  # skip corrupted / truncated images

    tf_ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
            tf.TensorSpec(shape=(), dtype=tf.int64),
        ),
    )

    transform = augment_image if augment else preprocess_image
    tf_ds = tf_ds.map(transform, num_parallel_calls=2)
    tf_ds = tf_ds.batch(batch_size)
    tf_ds = tf_ds.repeat()   # required because steps_per_epoch < full dataset size
    tf_ds = tf_ds.prefetch(2)
    return tf_ds


# ─── Model ────────────────────────────────────────────────────────────────────

def build_model(num_classes: int = NUM_CLASSES):
    base = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    # dtype="float32" is critical with mixed_precision — softmax must stay in fp32
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = keras.Model(inputs, outputs, name="NutriDish_MobileNetV2")
    return model, base


def unfreeze_for_fine_tuning(base_model, fine_tune_at: int = FINE_TUNE_AT):
    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    print(f"  Fine-tuning from layer {fine_tune_at} / {len(base_model.layers)}")


# ─── Training ─────────────────────────────────────────────────────────────────

def train():
    print("=" * 60)
    print("  NutriDish Training Pipeline")
    print("=" * 60)

    # ── Phase 1: head only ────────────────────────────────────────────────────
    print("\n[Phase 1] Building model …")
    model, base_model = build_model()
    model.summary(line_length=80)

    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_acc")],
    )

    print("\n[Phase 1] Loading streaming datasets …")
    train_ds = build_tf_dataset("train", augment=True)
    val_ds   = build_tf_dataset("validation", augment=False)

    callbacks_p1 = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
        keras.callbacks.ModelCheckpoint("best_phase1.keras", save_best_only=True, verbose=1),
    ]

    print("\n[Phase 1] Training classification head …")
    history1 = model.fit(
        train_ds,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=val_ds,
        validation_steps=VAL_STEPS,   # 64 × 16 = 1024 val samples
        epochs=EPOCHS // 2,
        callbacks=callbacks_p1,
    )

    # ── Phase 2: fine-tune top layers ─────────────────────────────────────────
    print("\n[Phase 2] Unfreezing top layers for fine-tuning …")
    unfreeze_for_fine_tuning(base_model, FINE_TUNE_AT)

    model.compile(
        optimizer=keras.optimizers.Adam(FINE_TUNE_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_acc")],
    )

    # Streaming datasets are consumed — must rebuild for phase 2
    print("\n[Phase 2] Reloading streaming datasets …")
    train_ds = build_tf_dataset("train", augment=True)
    val_ds   = build_tf_dataset("validation", augment=False)

    callbacks_p2 = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, verbose=1),
        keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, verbose=1),
    ]

    print("\n[Phase 2] Fine-tuning …")
    model.fit(
        train_ds,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=val_ds,
        validation_steps=VAL_STEPS,
        epochs=EPOCHS,
        callbacks=callbacks_p2,
        initial_epoch=len(history1.history["loss"]),
    )

    print(f"\n✅ Model saved to: {MODEL_SAVE_PATH}")
    return model


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
    finally:
        print("Cleaning up …")
        tf.keras.backend.clear_session()
        gc.collect()
        # os._exit(0) skips Python's GIL teardown, which is what causes the
        # "PyGILState_Release" crash when HuggingFace streaming threads are
        # still alive at interpreter shutdown.
        os._exit(0)