"""
NutriDish — ML Training Pipeline
Uses Food-101 via Hugging Face streaming + MobileNetV2 transfer learning.

Run:
    python model/train.py
"""

import os
import gc
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ─── GPU Optimization ────────────────────────────────────────────────────────

from tensorflow.keras import mixed_precision

# Enable mixed precision (faster on RTX GPUs)
mixed_precision.set_global_policy("mixed_float16")

# Prevent TensorFlow from allocating all VRAM at once
gpus = tf.config.experimental.list_physical_devices("GPU")

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        print("✅ GPU Enabled:", gpus)

    except RuntimeError as e:
        print(e)


# ─── Config ────────────────────────────────────────────────────────────────────

IMG_SIZE = 224
BATCH_SIZE = 16   #actual -> 32
EPOCHS = 4     #actual -> 20
NUM_CLASSES = 101
LEARNING_RATE = 1e-3
FINE_TUNE_LR = 1e-5
FINE_TUNE_AT = 100        # Unfreeze MobileNetV2 layers from this index onward
STEPS_PER_EPOCH = 50     # Streamed dataset: define steps manually #actual -> 500
VAL_STEPS = 64           #actual -> 100
MODEL_SAVE_PATH = "nutridish_model.keras"

# Food-101 class names (alphabetical — matches HuggingFace label IDs)
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


# ─── Dataset (Streaming) ──────────────────────────────────────────────────────

def preprocess_image(image, label):
    """Resize, normalize, and optionally augment a single example."""
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    # MobileNetV2 expects [-1, 1]
    image = (image - 0.5) * 2.0
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label


def augment_image(image, label):
    """Light augmentation safe for food images."""
    image, label = preprocess_image(image, label)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image, label


def build_tf_dataset(split: str, augment: bool = False, batch_size: int = BATCH_SIZE):
    """
    Load Food-101 via HuggingFace streaming and convert to a tf.data pipeline.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install: pip install datasets Pillow")
        sys.exit(1)

    print(f"  Loading Food-101 [{split}] in streaming mode …")
    hf_dataset = load_dataset("food101", split=split, streaming=True)
    
    if split == "train": #actual -> remove only "if" split == "train": keep -hf_dataset = hf...
        hf_dataset = hf_dataset.shuffle(seed=42, buffer_size=1000)

    def generator():
        for sample in hf_dataset:
            img = sample["image"]
            # PIL → numpy
            img_np = np.array(img.convert("RGB"), dtype=np.uint8)
            label = sample["label"]
            yield img_np, label

    tf_ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
            tf.TensorSpec(shape=(), dtype=tf.int64),
        ),
    )

    transform = augment_image if augment else preprocess_image
    tf_ds = tf_ds.map(transform, num_parallel_calls=2) # actual -> num_parallel_calls=tf.data.AUTOTUNE
    tf_ds = tf_ds.batch(batch_size).repeat()  #actual -> remove the .repeat()
    tf_ds = tf_ds.prefetch(2) # actual -> replace 2 with tf.data.AUTOTUNE
    return tf_ds


# ─── Model ────────────────────────────────────────────────────────────────────

def build_model(num_classes: int = NUM_CLASSES) -> keras.Model:
    """MobileNetV2 backbone + custom classification head."""
    base = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False   # Freeze during Phase 1

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="NutriDish_MobileNetV2")
    return model, base


def unfreeze_for_fine_tuning(model: keras.Model, base_model: keras.Model, fine_tune_at: int = FINE_TUNE_AT):
    """Unfreeze top layers of the base model for fine-tuning."""
    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    print(f"  Fine-tuning from layer {fine_tune_at} / {len(base_model.layers)}")


# ─── Training ─────────────────────────────────────────────────────────────────

def train():
    print("=" * 60)
    print("  NutriDish Training Pipeline")
    print("=" * 60)

    # ── Phase 1: Train head only ────────────────────────────────────────────
    print("\n[Phase 1] Building model …")
    model, base_model = build_model()
    model.summary(line_length=80)

    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_acc")],
    )

    callbacks_phase1 = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
        keras.callbacks.ModelCheckpoint("best_phase1.keras", save_best_only=True, verbose=1),
    ]

    print("\n[Phase 1] Loading streaming datasets …")
    train_ds = build_tf_dataset("train", augment=True)
    val_ds = build_tf_dataset("validation", augment=False)
    print(train_ds)
    print(val_ds)

    print("\n[Phase 1] Training classification head …")
    history1 = model.fit(
        train_ds,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=val_ds,
        validation_steps=VAL_STEPS,
        epochs=EPOCHS // 2,
        callbacks=callbacks_phase1,
    )

    # ── Phase 2: Fine-tune ───────────────────────────────────────────────────
    print("\n[Phase 2] Unfreezing top layers for fine-tuning …")
    unfreeze_for_fine_tuning(model, base_model, FINE_TUNE_AT)

    model.compile(
        optimizer=keras.optimizers.Adam(FINE_TUNE_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_acc")],
    )

    callbacks_phase2 = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, verbose=1),
        keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, verbose=1),
    ]

    print("\n[Phase 2] Fine-tuning …")
    # Reload streaming datasets (they're consumed after one pass)
    train_ds = build_tf_dataset("train", augment=True)
    val_ds = build_tf_dataset("validation", augment=False)
    print(train_ds)
    print(val_ds)

    history2 = model.fit(
        train_ds,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=val_ds,
        validation_steps=VAL_STEPS,
        epochs=EPOCHS,
        callbacks=callbacks_phase2,
        initial_epoch=len(history1.history["loss"]),
    )

    print(f"\n✅ Model saved to: {MODEL_SAVE_PATH}")
    return model


"""if __name__ == "__main__":
    train()
"""#this is replaced by the below code block

if __name__ == "__main__":
    try:
        train()

    finally:
        print("\nCleaning up resources...")

        tf.keras.backend.clear_session()
        gc.collect()

        os._exit(0)