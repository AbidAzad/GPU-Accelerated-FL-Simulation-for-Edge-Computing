from __future__ import annotations

import os, io, base64, re, ctypes, sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
MODEL_TYPE   = "network_mlp"                       # options: "lstm", "cifar_resnet", "network_mlp"
DATASET_TYPE = "network_traffic"              # options: "cyber_threat", "cifar10", "network_traffic"

# Text model / dataset parameters (LSTM + cyber-threat)
MAX_NB_WORDS   = 50000
EMBEDDING_DIM  = 100
MAX_SEQ_LENGTH = 450
TEST_SIZE      = 0.2

# Base directory and dataset paths
BASE_DIR      = Path(__file__).resolve().parent
CYBER_DF_PATH = BASE_DIR / "CyberThreatModel" / "CyberThreatDataset" / "cyber-threat-intelligence_all.csv"

# CIFAR-10 local archive
CIFAR_DIR           = BASE_DIR / "CIFAR-10" / "CIFAR-10-Dataset"
CIFAR_PATH          = CIFAR_DIR / "cifar10.npz"
CIFAR_INPUT_SHAPE   = (32, 32, 3)
CIFAR_LEARNING_RATE = 1e-3

# Network-traffic CSV (TABULAR features)
# TODO: point this to your actual network-traffic CSV file.
NETWORK_FLOW_PATH = BASE_DIR / "NetworkTraffic" / "network_traffic.csv"

# Path to CUDA FedAvg shared library
if sys.platform.startswith("win"):
    FEDAVG_LIB_PATH = BASE_DIR / "fedavg_gpu" / "fedavg_gpu.dll"
else:
    FEDAVG_LIB_PATH = BASE_DIR / "fedavg_gpu" / "libfedavg_gpu.so"

_FEDAVG_LIB: ctypes.CDLL | None = None

# Saved label names for the cyber-threat dataset (filled in _prepare_text_dataset)
CYBER_CLASS_NAMES: List[str] | None = None

# Saved label names + feature dim for the network-traffic dataset
NETWORK_CLASS_NAMES: List[str] | None = None
NETWORK_INPUT_DIM: int | None = None


def _load_fedavg_lib() -> ctypes.CDLL:
    global _FEDAVG_LIB
    if _FEDAVG_LIB is not None:
        return _FEDAVG_LIB

    if not FEDAVG_LIB_PATH.is_file():
        raise RuntimeError(
            f"{FEDAVG_LIB_PATH.name} not found at: {FEDAVG_LIB_PATH}. "
            "Build it and place it under fedavg_gpu/ or turn off USE_GPU_AGG."
        )

    lib = ctypes.cdll.LoadLibrary(str(FEDAVG_LIB_PATH))

    # extern "C" void fedavg_weighted_average_gpu(
    #     const float* h_client_weights,
    #     const int*   h_counts,
    #     float*       h_out,
    #     int          num_clients,
    #     int          vec_len);
    lib.fedavg_weighted_average_gpu.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # h_client_weights
        ctypes.POINTER(ctypes.c_int),    # h_counts
        ctypes.POINTER(ctypes.c_float),  # h_out
        ctypes.c_int,                    # num_clients
        ctypes.c_int,                    # vec_len
    ]
    lib.fedavg_weighted_average_gpu.restype = None

    _FEDAVG_LIB = lib
    return lib


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
    if MODEL_TYPE == "network_mlp":
        return _build_network_mlp(num_classes)

    raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")


def _build_lstm(num_classes: int):
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQ_LENGTH))
    model.add(LSTM(150, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model


def _build_cifar_resnet(num_classes: int):
    """
    ResNet-50 classifier for CIFAR-10 used in FL mode.
    """
    inputs = tf.keras.Input(shape=CIFAR_INPUT_SHAPE)
    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights=None,   # train from scratch to avoid large external downloads
        input_tensor=inputs,
    )

    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(base.output)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="ResNet50_CIFAR10",
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=CIFAR_LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _build_network_mlp(num_classes: int):
    """
    Simple MLP for tabular network-traffic data.

    Expects NETWORK_INPUT_DIM to be set by _prepare_network_dataset()
    via load_data() before this is called.
    """
    global NETWORK_INPUT_DIM
    if NETWORK_INPUT_DIM is None:
        raise RuntimeError(
            "NETWORK_INPUT_DIM is not set. "
            "Call load_data() with DATASET_TYPE='network_traffic' before build_model()."
        )

    inputs = tf.keras.Input(shape=(NETWORK_INPUT_DIM,), name="features")
    x = tf.keras.layers.Dense(256, activation="relu")(inputs)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="NetworkTrafficMLP")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ============================================================
# DATASET SWITCHER
# ============================================================

def load_data(nrows: int | None = None):
    """
    Returns: X_train, X_test, y_train, y_test, num_classes

    nrows: optional row limit for quick dev (only used for CSV-based datasets).
    """
    global DATASET_TYPE

    if DATASET_TYPE == "cyber_threat":
        df = _load_cti_dataset(CYBER_DF_PATH, nrows=nrows)
        return _prepare_text_dataset(df)

    if DATASET_TYPE == "cifar10":
        return _load_cifar10_dataset(CIFAR_PATH)

    if DATASET_TYPE == "network_traffic":
        return _load_network_traffic_dataset(NETWORK_FLOW_PATH, nrows=nrows)

    raise ValueError(f"Unknown DATASET_TYPE: {DATASET_TYPE}")


# ---------- cyber-threat text dataset ----------

def _load_cti_dataset(path: Path, nrows: int | None = None):
    df = pd.read_csv(path, nrows=nrows)
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

    # Save class names globally so the server can print nice reports later
    global CYBER_CLASS_NAMES
    CYBER_CLASS_NAMES = enc.classes_.tolist()

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
    Load CIFAR-10 from a local npz.

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


# ---------- Network-traffic tabular dataset ----------

def _load_network_traffic_dataset(path: Path, nrows: int | None = None):
    if not path.exists():
        raise FileNotFoundError(
            f"Network-traffic CSV not found at {path}. "
            "Update NETWORK_FLOW_PATH to point to your dataset."
        )

    df = pd.read_csv(path, nrows=nrows)
    return _prepare_network_dataset(df)


def _prepare_network_dataset(df: pd.DataFrame):
    """
    Prepare a generic tabular network-traffic dataset:

      * Auto-detect label column by name (label/attack/class/etc.)
      * Keep numeric feature columns, try to coerce non-numeric
      * Scale features with StandardScaler
      * Stratified train/test split
    """
    if df.empty:
        raise ValueError("Network-traffic DataFrame is empty.")

    # --- Label column detection ---
    # Try common name patterns
    candidates = []
    for col in df.columns:
        col_l = col.lower()
        if any(key in col_l for key in ["label", "attack", "class", "category", "result"]):
            candidates.append(col)

    if not candidates:
        raise ValueError(
            "Could not automatically find a label column in network-traffic dataset. "
            "Expected a column containing one of: label, attack, class, category, result."
        )

    label_col = candidates[0]
    y_raw = df[label_col].astype(str)

    # Features: drop label and obviously non-feature columns like timestamps
    feature_df = df.drop(columns=[label_col])

    # Drop columns that are all NaN or constant
    feature_df = feature_df.dropna(axis=1, how="all")
    nunique = feature_df.nunique(dropna=False)
    feature_df = feature_df.loc[:, nunique > 1]

    # Try to coerce non-numeric columns to numeric where possible
    for col in feature_df.columns:
        if not np.issubdtype(feature_df[col].dtype, np.number):
            try:
                feature_df[col] = pd.to_numeric(feature_df[col], errors="raise")
            except Exception:
                # If still non-numeric, drop the column
                feature_df = feature_df.drop(columns=[col])

    if feature_df.empty:
        raise ValueError("No numeric feature columns remain in network-traffic dataset.")

    X = feature_df.fillna(0.0).astype("float32")

    # Encode labels
    enc = LabelEncoder()
    y_int = enc.fit_transform(y_raw)
    num_classes = len(enc.classes_)
    y = to_categorical(y_int, num_classes)

    # Save globals for later usage (metrics + model building)
    global NETWORK_CLASS_NAMES, NETWORK_INPUT_DIM
    NETWORK_CLASS_NAMES = enc.classes_.tolist()
    NETWORK_INPUT_DIM   = X.shape[1]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype("float32")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, stratify=y_int
    )

    return X_train, X_test, y_train, y_test, num_classes


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


# ============================================================
# FEDERATED AVERAGING (CPU + GPU)
# ============================================================

def fedavg_weighted_average(
    client_updates: List[Tuple[List[np.ndarray], int]]
) -> List[np.ndarray]:
    """
    Pure NumPy (CPU) FedAvg.
    """
    if not client_updates:
        raise ValueError("No client updates provided to fedavg_weighted_average().")
    print("\n[AGG] CPU is being used for aggregation.")
    weight_lists = [w for (w, n) in client_updates]
    counts = np.array([n for (w, n) in client_updates], dtype=np.float64)
    total = np.sum(counts)

    if total <= 0:
        raise ValueError("Total sample count is non-positive in fedavg_weighted_average().")

    out: List[np.ndarray] = []
    num_layers = len(weight_lists[0])

    for i in range(num_layers):
        stack = np.stack([w[i] for w in weight_lists], axis=0)  # (K, ...)
        flat  = stack.reshape(stack.shape[0], -1)
        avg   = np.dot(counts, flat) / total
        out.append(avg.reshape(stack.shape[1:]))

    return out


def fedavg_weighted_average_gpu(
    client_updates: List[Tuple[List[np.ndarray], int]]
) -> List[np.ndarray]:
    """
    GPU-oriented FedAvg using fedavg_gpu.dll (.so on Linux).
    Flattens all layers into one big vector per client, calls the CUDA kernel,
    then reshapes back to the original layer shapes.
    """
    print("\n[AGG] GPU is being used for aggregation.")
    if not client_updates:
        raise ValueError("No client updates provided to fedavg_weighted_average_gpu().")

    # Unpack weights and counts
    weight_lists: List[List[np.ndarray]] = [w for (w, n) in client_updates]
    counts = np.asarray([n for (w, n) in client_updates], dtype=np.int32)
    total_samples = int(np.sum(counts))

    if total_samples == 0:
        raise ValueError("Total number of samples is zero in fedavg_weighted_average_gpu().")

    # Use the first client's weights as a reference for shapes/dtypes
    ref_weights: List[np.ndarray] = weight_lists[0]
    layer_shapes = [w.shape for w in ref_weights]
    layer_sizes  = [int(np.prod(shape)) for shape in layer_shapes]
    total_params = sum(layer_sizes)

    # Flatten all layers for each client into one 1D vector
    flat_list: List[np.ndarray] = []
    for idx, weights in enumerate(weight_lists):
        flat_client = np.concatenate(
            [w.reshape(-1) for w in weights],
            axis=0
        ).astype(np.float32)

        if flat_client.size != total_params:
            raise ValueError(
                f"All clients must share the same total parameter count. "
                f"client {idx} has {flat_client.size}, expected {total_params}"
            )

        flat_list.append(flat_client)

    # Shape (num_clients, total_params)
    flat_stack = np.stack(flat_list, axis=0)
    flat_stack = np.ascontiguousarray(flat_stack, dtype=np.float32)
    counts     = np.ascontiguousarray(counts, dtype=np.int32)

    num_clients, total_params_check = flat_stack.shape
    assert total_params_check == total_params

    out = np.empty(total_params, dtype=np.float32)

    lib = _load_fedavg_lib()

    ptr_weights = flat_stack.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    ptr_counts  = counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    ptr_out     = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    lib.fedavg_weighted_average_gpu(
        ptr_weights,
        ptr_counts,
        ptr_out,
        ctypes.c_int(num_clients),
        ctypes.c_int(total_params),
    )

    flat_avg = out  # shape: (total_params,)

    # Reshape averaged vector back into per-layer tensors
    averaged_weights: List[np.ndarray] = []
    offset = 0
    for shape, size, ref_w in zip(layer_shapes, layer_sizes, ref_weights):
        chunk = flat_avg[offset:offset + size]
        averaged_weights.append(chunk.reshape(shape).astype(ref_w.dtype))
        offset += size

    return averaged_weights


# ============================================================
# MODEL PRIMING
# ============================================================

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

    if MODEL_TYPE == "network_mlp":
        # Use NETWORK_INPUT_DIM determined by _prepare_network_dataset
        global NETWORK_INPUT_DIM
        if NETWORK_INPUT_DIM is None:
            raise RuntimeError(
                "NETWORK_INPUT_DIM is not set. "
                "Call load_data() for DATASET_TYPE='network_traffic' before prime_model()."
            )
        dummy_input = np.zeros((1, NETWORK_INPUT_DIM), dtype="float32")
        _ = model(dummy_input, training=False)
        return model

    # Fallback: just return the model without priming
    return model


# ============================================================
# METRICS HELPERS (SERVER USES THESE)
# ============================================================

def print_cyber_threat_metrics(model, X_test, y_test, round_num: int) -> None:
    """
    Print detailed classification metrics for the cyber-threat text dataset.

    Assumes y_test is one-hot encoded. Uses CYBER_CLASS_NAMES (from the
    LabelEncoder) when available, otherwise falls back to a hard-coded list.
    Also prints a benign vs malicious summary, where benign is identified
    by name if possible.
    """
    if DATASET_TYPE != "cyber_threat":
        print(f"[SRV][R{round_num}] print_cyber_threat_metrics() skipped (DATASET_TYPE={DATASET_TYPE!r})")
        return

    from sklearn.metrics import classification_report

    # Get predictions
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Class names for full report
    if CYBER_CLASS_NAMES is not None:
        class_names = CYBER_CLASS_NAMES
    else:
        # Fallback based on your dataset
        class_names = [
            "benign",
            "malware",
            "attack_pattern",
            "software_attack",
            "threat_actor",
            "identity",
        ]

    print(f"\n[SRV][R{round_num}] Detailed Classification Report:")
    try:
        print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    except Exception as e:
        print(f"[SRV][R{round_num}] classification_report failed: {e}")

    # Benign vs malicious summary
    # Try to locate the 'benign' class by name; fall back to index 0.
    if CYBER_CLASS_NAMES is not None and "benign" in CYBER_CLASS_NAMES:
        benign_index = CYBER_CLASS_NAMES.index("benign")
    else:
        benign_index = 0

    benign_mask = (y_true == benign_index)
    malicious_mask = ~benign_mask

    benign_acc = float(np.mean(y_pred[benign_mask] == y_true[benign_mask])) if benign_mask.any() else 0.0
    malicious_acc = float(np.mean(y_pred[malicious_mask] == y_true[malicious_mask])) if malicious_mask.any() else 0.0

    num_samples = len(y_test)
    benign_pct = 100.0 * float(benign_mask.sum()) / num_samples if num_samples else 0.0
    malicious_pct = 100.0 * float(malicious_mask.sum()) / num_samples if num_samples else 0.0

    print(f"\n[SRV][R{round_num}] Threat Detection Summary:")
    print(f"  Benign samples:     {benign_mask.sum()} ({benign_pct:.1f}%)")
    print(f"  Malicious samples:  {malicious_mask.sum()} ({malicious_pct:.1f}%)")
    print(f"  Benign accuracy:    {benign_acc * 100:.2f}%")
    print(f"  Malicious accuracy: {malicious_acc * 100:.2f}%")


def print_network_traffic_metrics(model, X_test, y_test, round_num: int) -> None:
    """
    Print detailed classification metrics for the network-traffic tabular dataset.

    Assumes y_test is one-hot encoded. Uses NETWORK_CLASS_NAMES when available.
    """
    if DATASET_TYPE != "network_traffic":
        print(f"[SRV][R{round_num}] print_network_traffic_metrics() skipped (DATASET_TYPE={DATASET_TYPE!r})")
        return

    from sklearn.metrics import classification_report, confusion_matrix

    # Get predictions
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Class names
    if NETWORK_CLASS_NAMES is not None:
        class_names = NETWORK_CLASS_NAMES
    else:
        class_names = sorted(list({int(c) for c in np.unique(y_true)}))
        class_names = [str(c) for c in class_names]

    print(f"\n[SRV][R{round_num}] Network-Traffic Classification Report:")
    try:
        print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    except Exception as e:
        print(f"[SRV][R{round_num}] classification_report failed: {e}")

    try:
        cm = confusion_matrix(y_true, y_pred)
        print(f"\n[SRV][R{round_num}] Confusion Matrix (rows=true, cols=pred):")
        print(cm)
    except Exception as e:
        print(f"[SRV][R{round_num}] confusion_matrix failed: {e}")
