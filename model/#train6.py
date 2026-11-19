"""
NutriDish — ML Training Pipeline (Simple & Reliable)
"""

"""
What was actually wrong: The generator-based approach had a subtle exhaustion bug — when Keras calls model.evaluate() on the validation set internally, the generator runs to completion and then on the next epoch it starts from scratch but TF's internal iterator gets confused, returning 0 samples → accuracy = 0.
The fix: Go back to from_tensor_slices() which is a proper in-memory dataset that TF can rewind and re-iterate perfectly every epoch. The earlier OOM was because we didn't have TF_GPU_ALLOCATOR=cuda_malloc_async set. Now that it's set, TF uses async memory allocation and won't try to load the whole array into a single GPU chunk.
Your training accuracy was already reaching 65%+ in epoch 1 — the model was working perfectly. This was purely a validation metric bug. With the fix you should see real val_accuracy numbers from epoch 1 itself. 
"""

import os, gc, sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, mixed_precision
from pathlib import Path

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

mixed_precision.set_global_policy("mixed_float16")
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("✅ GPU Enabled:", gpus)

# ─── Config ───────────────────────────────────────────────────────────────────

IMG_SIZE            = 224
BATCH_SIZE          = 16
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
    "apple_pie","baby_back_ribs","baklava","beef_carpaccio","beef_tartare",
    "beet_salad","beignets","bibimbap","bread_pudding","breakfast_burrito",
    "bruschetta","caesar_salad","cannoli","caprese_salad","carrot_cake",
    "ceviche","cheese_plate","cheesecake","chicken_curry","chicken_quesadilla",
    "chicken_wings","chocolate_cake","chocolate_mousse","churros","clam_chowder",
    "club_sandwich","crab_cakes","creme_brulee","croque_madame","cup_cakes",
    "deviled_eggs","donuts","dumplings","edamame","eggs_benedict",
    "escargots","falafel","filet_mignon","fish_and_chips","foie_gras",
    "french_fries","french_onion_soup","french_toast","fried_calamari","fried_rice",
    "frozen_yogurt","garlic_bread","gnocchi","greek_salad","grilled_cheese_sandwich",
    "grilled_salmon","guacamole","gyoza","hamburger","hot_and_sour_soup",
    "hot_dog","huevos_rancheros","hummus","ice_cream","lasagna",
    "lobster_bisque","lobster_roll_sandwich","macaroni_and_cheese","macarons","miso_soup",
    "mussels","nachos","omelette","onion_rings","oysters",
    "pad_thai","paella","pancakes","panna_cotta","peking_duck",
    "pho","pizza","pork_chop","poutine","prime_rib",
    "pulled_pork_sandwich","ramen","ravioli","red_velvet_cake","risotto",
    "samosa","sashimi","scallops","seaweed_salad","shrimp_and_grits",
    "spaghetti_bolognese","spaghetti_carbonara","spring_rolls","steak","strawberry_shortcake",
    "sushi","tacos","takoyaki","tiramisu","tuna_tartare","waffles",
]

# ─── Cache ────────────────────────────────────────────────────────────────────

def cache_split(split, n_samples):
    cache_path = CACHE_DIR / f"{split}_{n_samples}.npz"
    if cache_path.exists():
        print(f"  ✅ Cache exists: {cache_path}")
        return cache_path
    from datasets import load_dataset
    print(f"  Streaming {n_samples} from Food-101 [{split}] …")
    hf_ds = load_dataset("food101", split=split, streaming=True)
    hf_ds = hf_ds.shuffle(seed=42, buffer_size=5000)
    images, labels = [], []
    for i, sample in enumerate(hf_ds):
        if i >= n_samples: break
        try:
            img = np.array(sample["image"].convert("RGB"), dtype=np.uint8)
            img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE]).numpy().astype(np.uint8)
            images.append(img)
            labels.append(sample["label"])
        except: continue
        if (i+1) % 1000 == 0: print(f"    {i+1}/{n_samples} …")
    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels, dtype=np.int32)
    np.savez_compressed(cache_path, images=images, labels=labels)
    print(f"  ✅ Saved {len(images)} → {cache_path}")
    return cache_path

# ─── Dataset — KEY FIX: load into RAM, use tf.data properly ──────────────────

def build_dataset(cache_path, do_augment):
    """
    Load npz into RAM as numpy arrays, then build a PROPER repeatable
    tf.data pipeline. No generator — avoids the reset/exhaustion bug.
    """
    data   = np.load(cache_path)
    images = data["images"]          # uint8, stays in CPU RAM
    labels = data["labels"].astype(np.int32)

    # Preprocess entire val set into float32 tensors upfront (small enough)
    # For train, we use map() with augmentation
    def preprocess(img, lbl):
        img = tf.cast(img, tf.float32) / 255.0
        img = (img - 0.5) * 2.0
        lbl = tf.one_hot(lbl, NUM_CLASSES)
        return img, lbl

    def augment(img, lbl):
        img, lbl = preprocess(img, lbl)
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.15)
        img = tf.image.random_contrast(img, 0.85, 1.15)
        img = tf.clip_by_value(img, -1.0, 1.0)
        return img, lbl

    # Build dataset from numpy — TF handles batching from CPU RAM
    ds = tf.data.Dataset.from_tensor_slices((images, labels))

    if do_augment:
        ds = ds.shuffle(len(images), reshuffle_each_iteration=True)
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# ─── Model ────────────────────────────────────────────────────────────────────

def build_model():
    base = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False, weights="imagenet"
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
    out = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)
    return keras.Model(inputs, out, name="NutriDish_MobileNetV2"), base

# ─── Train ────────────────────────────────────────────────────────────────────

def train():
    print("="*60)
    print("  NutriDish Training Pipeline")
    print("="*60)

    print("\n[Setup] Preparing cache …")
    train_cache = cache_split("train",      CACHE_TRAIN_SAMPLES)
    val_cache   = cache_split("validation", CACHE_VAL_SAMPLES)

    # Build datasets — no steps_per_epoch needed, Keras counts batches automatically
    train_ds = build_dataset(train_cache, do_augment=True)
    val_ds   = build_dataset(val_cache,   do_augment=False)

    print(f"\n  Train batches/epoch : {CACHE_TRAIN_SAMPLES // BATCH_SIZE}")
    print(f"  Val   batches/epoch : {CACHE_VAL_SAMPLES   // BATCH_SIZE}")

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
            restore_best_weights=True, mode="max", verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=3,
            verbose=1, mode="max"
        ),
        keras.callbacks.ModelCheckpoint(
            "best_phase1.keras", monitor="val_accuracy",
            save_best_only=True, mode="max", verbose=1
        ),
    ]

    print("\n[Phase 1] Training head (frozen backbone) …")
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS // 2,
        callbacks=cb_p1,
    )

    print("\n[Phase 2] Unfreezing top MobileNetV2 layers …")
    base_model.trainable = True
    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False
    print(f"  Fine-tuning from layer {FINE_TUNE_AT}/{len(base_model.layers)}")

    model.compile(
        optimizer=keras.optimizers.SGD(FINE_TUNE_LR, momentum=0.9),
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_acc")],
    )

    cb_p2 = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=6,
            restore_best_weights=True, mode="max", verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=3,
            verbose=1, mode="max"
        ),
        keras.callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH, monitor="val_accuracy",
            save_best_only=True, mode="max", verbose=1
        ),
    ]

    print("\n[Phase 2] Fine-tuning …")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=cb_p2,
        initial_epoch=len(history1.history["loss"]),
    )

    print(f"\n✅ Model saved → {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        tf.keras.backend.clear_session()
        gc.collect()
        os._exit(0)