"""
NutriDish — ML Training Pipeline (Stratified Cache Fix)
"""

import os, gc, sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

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

# Stratified: N samples PER CLASS → total = N × 101
TRAIN_PER_CLASS     = 80    # 80 × 101 = 8080 train samples
VAL_PER_CLASS       = 20    # 20 × 101 = 2020 val samples

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

# ─── Stratified cache — guaranteed all 101 classes ───────────────────────────

def build_stratified_cache(split: str, per_class: int) -> Path:
    """
    Stream Food-101 and collect exactly `per_class` images per class.
    This guarantees all 101 classes are represented equally.
    """
    cache_path = CACHE_DIR / f"{split}_stratified_{per_class}pc.npz"
    if cache_path.exists():
        data = np.load(cache_path)
        n_classes = len(np.unique(data["labels"]))
        if n_classes == NUM_CLASSES:
            print(f"  ✅ Cache exists ({n_classes} classes): {cache_path}")
            return cache_path
        else:
            print(f"  ⚠️  Cache only has {n_classes} classes — rebuilding …")
            cache_path.unlink()

    from datasets import load_dataset
    print(f"  Building stratified cache for [{split}], {per_class} per class …")

    hf_ds = load_dataset("food101", split=split, streaming=True)

    # Bucket per class
    buckets = {i: [] for i in range(NUM_CLASSES)}
    done    = set()
    total   = NUM_CLASSES * per_class

    for sample in hf_ds:
        lbl = sample["label"]
        if lbl in done:
            continue
        if len(buckets[lbl]) < per_class:
            try:
                img = np.array(sample["image"].convert("RGB"), dtype=np.uint8)
                img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE]).numpy().astype(np.uint8)
                buckets[lbl].append(img)
            except:
                continue
            if len(buckets[lbl]) == per_class:
                done.add(lbl)
                collected = sum(len(v) for v in buckets.values())
                print(f"    {collected}/{total} — class {lbl} ({FOOD101_CLASSES[lbl]}) done")
        if len(done) == NUM_CLASSES:
            break

    images, labels = [], []
    for lbl, imgs in buckets.items():
        for img in imgs:
            images.append(img)
            labels.append(lbl)

    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels, dtype=np.int32)

    # Shuffle so classes aren't in blocks
    idx = np.random.permutation(len(images))
    images, labels = images[idx], labels[idx]

    np.savez_compressed(cache_path, images=images, labels=labels)
    print(f"  ✅ Saved {len(images)} samples ({len(np.unique(labels))} classes) → {cache_path}")
    return cache_path

# ─── Dataset ──────────────────────────────────────────────────────────────────

def preprocess(img, lbl):
    img = tf.cast(img, tf.float32) / 127.5 - 1.0
    lbl = tf.one_hot(lbl, NUM_CLASSES)
    return img, lbl

def augment_and_preprocess(img, lbl):
    img, lbl = preprocess(img, lbl)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.1)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    img = tf.clip_by_value(img, -1.0, 1.0)
    return img, lbl

def build_dataset(cache_path, do_augment):
    data   = np.load(cache_path)
    images = data["images"]
    labels = data["labels"].astype(np.int32)
    print(f"  Loaded {len(images)} samples, {len(np.unique(labels))} classes")
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if do_augment:
        ds = ds.shuffle(len(images), reshuffle_each_iteration=True)
        ds = ds.map(augment_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
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
    out = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return keras.Model(inputs, out, name="NutriDish_MobileNetV2"), base

# ─── Train ────────────────────────────────────────────────────────────────────

def train():
    print("="*60)
    print("  NutriDish Training Pipeline")
    print("="*60)

    print("\n[Setup] Building stratified cache …")
    train_cache = build_stratified_cache("train",      TRAIN_PER_CLASS)
    val_cache   = build_stratified_cache("validation", VAL_PER_CLASS)

    print("\n[Setup] Loading datasets …")
    train_ds = build_dataset(train_cache, do_augment=True)
    val_ds   = build_dataset(val_cache,   do_augment=False)

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
            restore_best_weights=True, mode="max", verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=3,
            verbose=1, mode="max"),
        keras.callbacks.ModelCheckpoint(
            "best_phase1.keras", monitor="val_accuracy",
            save_best_only=True, mode="max", verbose=1),
    ]

    print("\n[Phase 1] Training head (frozen backbone) …")
    history1 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS // 2, callbacks=cb_p1,
    )

    print("\n[Phase 2] Unfreezing top MobileNetV2 layers …")
    base_model.trainable = True
    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False
    print(f"  Fine-tuning from layer {FINE_TUNE_AT}/{len(base_model.layers)}")

    model.compile(
        optimizer=keras.optimizers.Adam(FINE_TUNE_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_acc")],
    )

    cb_p2 = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=6,
            restore_best_weights=True, mode="max", verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=3,
            verbose=1, mode="max"),
        keras.callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH, monitor="val_accuracy",
            save_best_only=True, mode="max", verbose=1),
    ]

    print("\n[Phase 2] Fine-tuning …")
    model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS, callbacks=cb_p2,
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