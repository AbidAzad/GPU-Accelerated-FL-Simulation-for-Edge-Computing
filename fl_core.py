# fl_core.py
from __future__ import annotations
import os, io, base64, re
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import tensorflow as tf


# ============================================================
# GLOBAL CONFIG
# ============================================================

# Switches (server & clients all see the same import)
MODEL_TYPE   = "lstm"                       # options: "lstm", "cifar_resnet"
DATASET_TYPE = "cyber_threat"                # options: "cyber_threat", "cifar10"

# Text model / dataset parameters (LSTM + cyber-threat)
MAX_NB_WORDS   = 50000
EMBEDDING_DIM  = 100
MAX_SEQ_LENGTH = 450
TEST_SIZE      = 0.2

# Base directory and dataset paths
BASE_DIR      = Path(__file__).resolve().parent
CYBER_DF_PATH = BASE_DIR / "CyberThreatModel" / "CyberThreatDataset" / "cyber-threat-intelligence_all.csv"

# CIFAR-10 local archive (created separately by Centralized_CIFAR.py)
CIFAR_DIR           = BASE_DIR / "CIFAR-10" / "CIFAR-10-Dataset"
CIFAR_PATH          = CIFAR_DIR / "cifar10.npz"
CIFAR_INPUT_SHAPE   = (32, 32, 3)
CIFAR_LEARNING_RATE = 1e-3


# ============================================================
# MODEL SWITCHER
# ============================================================

def build_model(num_classes: int):
    """
    Universal model builder, controlled by MODEL_TYPE.
    """
    global MODEL_TYPE

    if MODEL_TYPE == "lstm":
        return _build_lstm(num_classes)
    if MODEL_TYPE == "cifar_resnet":
        return _build_cifar_resnet(num_classes)

    # For future:
    # elif MODEL_TYPE == "othermodel":
    #     return _build_othermodel(num_classes)

    raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")


def _build_lstm(num_classes: int):
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQ_LENGTH))
    model.add(LSTM(150, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    return model


def _build_cifar_resnet(num_classes: int):
    """
    ResNet-50 classifier for CIFAR-10 used in FL mode.
    """
    inputs = tf.keras.Input(shape=CIFAR_INPUT_SHAPE)
    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights=None,           
        input_tensor=inputs,
    )

    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(base.output)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs,
                           outputs=outputs,
                           name="ResNet50_CIFAR10")

    optimizer = tf.keras.optimizers.Adam(learning_rate=CIFAR_LEARNING_RATE)
    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# ============================================================
# DATASET SWITCHER
# ============================================================

def load_data():
    """
    Returns: X_train, X_test, y_train, y_test, num_classes
    """
    global DATASET_TYPE

    if DATASET_TYPE == "cyber_threat":
        df = _load_cti_dataset(CYBER_DF_PATH)
        return _prepare_text_dataset(df)

    if DATASET_TYPE == "cifar10":
        return _load_cifar10_dataset(CIFAR_PATH)

    # Future dataset:
    # elif DATASET_TYPE == "my_other_dataset":
    #     df = pd.read_csv(OTHER_PATH)
    #     return _prepare_text_dataset(df)

    raise ValueError(f"Unknown DATASET_TYPE: {DATASET_TYPE}")


# ---------- cyber-threat text dataset ----------

def _load_cti_dataset(path: Path):
    df = pd.read_csv(path)
    df["text"]  = df["text"].fillna("")
    df["label"] = df["label"].fillna("benign")
    return df[["text", "label"]]


def _clean(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _prepare_text_dataset(df: pd.DataFrame):
    df["text_clean"] = df["text"].apply(_clean)

    # Encode labels
    enc = LabelEncoder()
    y_int = enc.fit_transform(df["label"])
    num_classes = len(enc.classes_)
    y = to_categorical(y_int, num_classes)

    # Tokenize
    tok = Tokenizer(num_words=MAX_NB_WORDS)
    tok.fit_on_texts(df["text_clean"])

    seqs = tok.texts_to_sequences(df["text_clean"])
    X = pad_sequences(seqs, maxlen=MAX_SEQ_LENGTH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y_int
    )

    return X_train, X_test, y_train, y_test, num_classes


# ---------- CIFAR-10 image dataset ----------

def _load_cifar10_dataset(path: Path):
    """
    Load CIFAR-10 from a local npz (created by Centralized_CIFAR.py).

    The archive is expected to contain x_train, y_train, x_test, y_test
    arrays in the original keras.datasets.cifar10 format.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"CIFAR-10 archive not found at {path}. "
            "Run Centralized_CIFAR.py once to download and save cifar10.npz, "
            "or update CIFAR_PATH if it's stored elsewhere."
        )

    data = np.load(path)
    x_train = data["x_train"].astype("float32") / 255.0
    x_test  = data["x_test"].astype("float32") / 255.0

    # y_* are shape (N, 1) in the original dataset; flatten them
    y_train_int = data["y_train"].reshape(-1)
    y_test_int  = data["y_test"].reshape(-1)

    num_classes = int(max(y_train_int.max(), y_test_int.max()) + 1)
    y_train = to_categorical(y_train_int, num_classes)
    y_test  = to_categorical(y_test_int, num_classes)

    return x_train, x_test, y_train, y_test, num_classes


# ============================================================
# SERIALIZATION HELPERS
# ============================================================

def weights_to_b64(weights):
    buf = io.BytesIO()
    np.savez_compressed(buf, *weights)
    return base64.b64encode(buf.getvalue()).decode()


def weights_from_b64(s):
    data = base64.b64decode(s.encode())
    buf = io.BytesIO(data)
    npz = np.load(buf)
    return [npz[k] for k in npz]


def arrays_to_b64(*arrays):
    buf = io.BytesIO()
    np.savez_compressed(buf, *arrays)
    return base64.b64encode(buf.getvalue()).decode()


def arrays_from_b64(s):
    data = base64.b64decode(s.encode())
    buf = io.BytesIO(data)
    npz = np.load(buf)
    return [npz[k] for k in npz]


def fedavg_weighted_average(client_updates):
    weight_lists = [w for (w, n) in client_updates]
    counts = np.array([n for (w, n) in client_updates])
    total = np.sum(counts)

    out = []
    num_layers = len(weight_lists[0])

    for i in range(num_layers):
        stack = np.stack([w[i] for w in weight_lists], axis=0)  # (K, ...)
        flat  = stack.reshape(stack.shape[0], -1)
        avg   = np.dot(counts, flat) / total
        out.append(avg.reshape(stack.shape[1:]))

    return out


def prime_model(model, seq_len: int = MAX_SEQ_LENGTH):
    """
    Run one dummy forward pass so Keras creates all weights.

    This makes set_weights/get_weights safe immediately after build_model().
    """

    if MODEL_TYPE == "lstm":
        # Dummy batch of one text sequence with the correct length
        dummy_input = np.zeros((1, seq_len), dtype="int32")
        _ = model(dummy_input, training=False)
        return model

    if MODEL_TYPE == "cifar_resnet":
        # Dummy batch of one CIFAR-10 image
        dummy_input = np.zeros((1,) + CIFAR_INPUT_SHAPE, dtype="float32")
        _ = model(dummy_input, training=False)
        return model

    # Fallback: just return the model without priming
    return model
