import os
from pathlib import Path

# -----------------------------------------
# Environment: CPU-only baseline (optional)
# -----------------------------------------
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# -----------------------------
# Basic configuration
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "CIFAR-10-Dataset"
DATA_DIR.mkdir(exist_ok=True)

CIFAR_PATH = DATA_DIR / "cifar10.npz"

if CIFAR_PATH.exists():
    print(f"[INFO] {CIFAR_PATH} already exists, nothing to do.")
else:
    print("[INFO] Downloading CIFAR-10 via tf.keras.datasets...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    print("[INFO] Saving to cifar10.npz...")
    np.savez_compressed(
        CIFAR_PATH,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )
    print(f"[INFO] Saved: {CIFAR_PATH}")

NUM_CLASSES   = 10
TEST_SIZE     = 0.2        # we'll just use the default CIFAR split
BATCH_SIZE    = 64
EPOCHS        = 20         # rely on early stopping
LEARNING_RATE = 1e-3

# CIFAR-10 images are 32x32x3
INPUT_SHAPE   = (32, 32, 3)


# -----------------------------
# Data loading & prep
# -----------------------------

def load_cifar10():
    """
    Load CIFAR-10 from tf.keras.datasets and preprocess for ResNet.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Scale to [0,1]; optionally you could use resnet50.preprocess_input
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32") / 255.0

    # One-hot labels
    y_train = to_categorical(y_train, NUM_CLASSES)
    y_test  = to_categorical(y_test, NUM_CLASSES)

    return x_train, x_test, y_train, y_test


# -----------------------------
# Model definition
# -----------------------------

def build_resnet50_model(
    input_shape=INPUT_SHAPE,
    num_classes=NUM_CLASSES,
    learning_rate=LEARNING_RATE,
) -> tf.keras.Model:
    """
    ResNet-50 classifier for CIFAR-10.

    """
    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights=None,          # random init; we only care about size, not accuracy
        input_shape=input_shape,
    )

    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(base.output)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=base.input, outputs=outputs, name="ResNet50_CIFAR10")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# -----------------------------
# Train and evaluate helpers
# -----------------------------

def train(
    model: tf.keras.Model,
    x_train,
    y_train,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
):
    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    print("\n[TRAIN] Starting training...")
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )
    return history


def evaluate(
    model: tf.keras.Model,
    x_test,
    y_test,
):
    print("\n[EVAL] Evaluating on test set...")
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"[EVAL] Test Loss: {loss:.4f}")
    print(f"[EVAL] Test Accuracy: {acc:.4f}")
    return loss, acc


# -----------------------------
# Script entry point
# -----------------------------

if __name__ == "__main__":
    print("[INFO] Loading CIFAR-10 dataset...")
    x_train, x_test, y_train, y_test = load_cifar10()

    print("\n[INFO] Data shapes:")
    print(f"  x_train: {x_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  x_test:  {x_test.shape}")
    print(f"  y_test:  {y_test.shape}")

    print("\n[INFO] Building ResNet-50 model...")
    model = build_resnet50_model()

    print("\n[INFO] Model summary:")
    model.summary()

    history = train(model, x_train, y_train)
    evaluate(model, x_test, y_test)
