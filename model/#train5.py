"""
NutriDish — ML Training Pipeline
Loads cached .npz files via generator (not from_tensor_slices) to avoid
pushing the full dataset into GPU memory at once.
look at dataset_cache folder before running this pipeline

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
from pathlib import Path

# ─── Must be set BEFORE TF initialises the GPU allocator ─────────────────────
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

# ─── GPU setup ────────────────────────────────────────────────────────────────

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

IMG_SIZE            = 224
BATCH_SIZE          = 16      # dropped from 32 — safer for 1.76 GB VRAM
EPOCHS              = 20
NUM_CLASSES         = 101
LEARNING_RATE       = 1e-3
FINE_TUNE_LR        = 1e-5
FINE_TUNE_AT        = 100
MODEL_SAVE_PATH     = "nutridish_model.keras"

CACHE_TRAIN_SAMPLES = 10000
CACHE_VAL_SAMPLES   = 3000
CACHE_DIR           = Path("dataset_cache")
CACHE_DIR.mkdir(exist_ok=True)

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

# ─── Cache dataset to disk (runs once) ───────────────────────────────────────

def cache_split(split: str, n_samples: int) -> Path:
    cache_path = CACHE_DIR / f"{split}_{n_samples}.npz"
    if cache_path.exists():
        print(f"  ✅ Cache exists: {cache_path}")
        return cache_path

    try:
        from datasets import load_dataset
    except ImportError:
        print("Install: pip install datasets Pillow")
        sys.exit(1)

    print(f"  Streaming {n_samples} samples from Food-101 [{split}] …")
    hf_ds = load_dataset("food101", split=split, streaming=True)
    hf_ds = hf_ds.shuffle(seed=42, buffer_size=5000)

    images, labels = [], []
    for i, sample in enumerate(hf_ds):
        if i >= n_samples:
            break
        try:
            img = np.array(sample["image"].convert("RGB"), dtype=np.uint8)
            img_t = tf.image.resize(img, [IMG_SIZE, IMG_SIZE]).numpy().astype(np.uint8)
            images.append(img_t)
            labels.append(sample["label"])
        except Exception:
            continue
        if (i + 1) % 1000 == 0:
            print(f"    {i+1}/{n_samples} …")

    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels, dtype=np.int32)
    np.savez_compressed(cache_path, images=images, labels=labels)
    print(f"  ✅ Saved {len(images)} samples → {cache_path}")
    return cache_path


# ─── Build tf.data via generator (CPU RAM only, not GPU) ─────────────────────

def make_generator(images: np.ndarray, labels: np.ndarray, shuffle: bool):
    """Returns a generator function that yields one sample at a time."""
    indices = np.arange(len(images))

    def gen():
        idx = indices.copy()
        if shuffle:
            np.random.shuffle(idx)
        for i in idx:
            yield images[i], labels[i]

    return gen


def normalize(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = (image - 0.5) * 2.0
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label


def augment(image, label):
    image, label = normalize(image, label)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
    image = tf.image.random_saturation(image, lower=0.85, upper=1.15)
    image = tf.clip_by_value(image, -1.0, 1.0)
    return image, label


def build_dataset(cache_path: Path, do_augment: bool, batch_size: int = BATCH_SIZE):
    data   = np.load(cache_path)
    images = data["images"]   # stays in CPU RAM
    labels = data["labels"]

    gen = make_generator(images, labels, shuffle=do_augment)

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.uint8),
            tf.TensorSpec(shape=(),                      dtype=tf.int32),
        ),
    )
    ds = ds.map(augment if do_augment else normalize, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.repeat()          # repeat so Keras steps_per_epoch works
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, len(images)


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
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    return keras.Model(inputs, outputs, name="NutriDish_MobileNetV2"), base


def unfreeze_for_fine_tuning(base_model, fine_tune_at: int = FINE_TUNE_AT):
    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    print(f"  Fine-tuning from layer {fine_tune_at} / {len(base_model.layers)}")


# ─── Train ────────────────────────────────────────────────────────────────────

def train():
    print("=" * 60)
    print("  NutriDish Training Pipeline")
    print("=" * 60)

    print("\n[Setup] Preparing dataset cache …")
    train_cache = cache_split("train",      CACHE_TRAIN_SAMPLES)
    val_cache   = cache_split("validation", CACHE_VAL_SAMPLES)

    train_ds, n_train = build_dataset(train_cache, do_augment=True)
    val_ds,   n_val   = build_dataset(val_cache,   do_augment=False)

    steps_per_epoch = n_train // BATCH_SIZE
    val_steps       = n_val   // BATCH_SIZE
    print(f"\n  Train: {n_train} samples → {steps_per_epoch} steps/epoch")
    print(f"  Val:   {n_val}   samples → {val_steps} steps/epoch")

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    print("\n[Phase 1] Building model …")
    model, base_model = build_model()
    model.summary(line_length=80)

    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_acc")],
    )

    cb_p1 = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5,
            restore_best_weights=True, mode="max"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=3, verbose=1, mode="max"
        ),
        keras.callbacks.ModelCheckpoint(
            "best_phase1.keras", monitor="val_accuracy",
            save_best_only=True, mode="max", verbose=1
        ),
    ]

    print("\n[Phase 1] Training classification head …")
    history1 = model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=val_steps,
        epochs=EPOCHS // 2,
        callbacks=cb_p1,
    )

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    print("\n[Phase 2] Unfreezing top layers …")
    unfreeze_for_fine_tuning(base_model, FINE_TUNE_AT)

    model.compile(
        optimizer=keras.optimizers.SGD(FINE_TUNE_LR, momentum=0.9),
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_acc")],
    )

    cb_p2 = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=6,
            restore_best_weights=True, mode="max"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=3, verbose=1, mode="max"
        ),
        keras.callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH, monitor="val_accuracy",
            save_best_only=True, mode="max", verbose=1
        ),
    ]

    print("\n[Phase 2] Fine-tuning …")
    model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=val_steps,
        epochs=EPOCHS,
        callbacks=cb_p2,
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
        os._exit(0)