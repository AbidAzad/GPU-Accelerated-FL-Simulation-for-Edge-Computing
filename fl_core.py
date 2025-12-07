# fl_core.py
# from __future__ import annotations
import os, io, base64, re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout


import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "fedavg_gpu"))

import fedavg_gpu as agg_ext 

import ctypes
import os
import numpy as np

_fedavg_lib = None

def _load_fedavg_lib():
    global _fedavg_lib
    if _fedavg_lib is not None:
        return _fedavg_lib

    #   fl_core.py
    #   fedavg_gpu/libfedavg_gpu.so
    here = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(here, "fedavg_gpu", "libfedavg_gpu.so")

    if not os.path.isfile(lib_path):
        raise RuntimeError("libfedavg_gpu.so not found at: {}".format(lib_path))

    lib = ctypes.cdll.LoadLibrary(lib_path)


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

    _fedavg_lib = lib
    return lib



# ============================================================
# GLOBAL CONFIG
# ============================================================

MODEL_TYPE   = "lstm"           # options: "lstm" only
DATASET_TYPE = "cyber_threat"   # options: "cyber_threat" only

# Global parameters (client/server can override)
MAX_NB_WORDS   = 50000
EMBEDDING_DIM  = 100
MAX_SEQ_LENGTH = 450
TEST_SIZE      = 0.2

# Path to dataset
BASE_DIR = Path(__file__).resolve().parent
CYBER_DF_PATH = BASE_DIR / "CyberThreatModel" / "CyberThreatDataset" / "cyber-threat-intelligence_all.csv"


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
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
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

    # Future dataset:
    # elif DATASET_TYPE == "my_other_dataset":
    #     df = pd.read_csv(OTHER_PATH)
    #     return _prepare_text_dataset(df)

    raise ValueError(f"Unknown DATASET_TYPE: {DATASET_TYPE}")


def _load_cti_dataset(path: Path):
    df = pd.read_csv(path)
    df["text"]  = df["text"].fillna("")
    df["label"] = df["label"].fillna("benign")
    return df[["text", "label"]]


def _clean(s: str) -> str:
    if not isinstance(s, str): return ""
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


def fedavg_weighted_average_gpu(
    client_updates: List[Tuple[List[np.ndarray], int]]
) -> List[np.ndarray]:
    """
    GPU-oriented FedAvg wrapper.
    """
    if not client_updates:
        raise ValueError("No client updates provided to fedavg_weighted_average_gpu().")

    
    # unpack weights and counts
    weight_lists: List[List[np.ndarray]] = [w for (w, n) in client_updates]
    counts = np.asarray([n for (w, n) in client_updates], dtype=np.int32)
    total_samples = int(np.sum(counts))

    if total_samples == 0:
        raise ValueError("Total number of samples is zero in fedavg_weighted_average_gpu().")

    # Use the first client's weights as a reference for layer shapes and dtypes.
    ref_weights: List[np.ndarray] = weight_lists[0]
    layer_shapes = [w.shape for w in ref_weights]
    layer_sizes = [int(np.prod(shape)) for shape in layer_shapes]
    total_params = sum(layer_sizes)

    # flatten all layers for each client into a single 1D vector
    flat_list: List[np.ndarray] = []
    for weights in weight_lists:
        # Flatten each layer and concatenate: [layer1_flat, layer2_flat, ...]
        flat_client = np.concatenate(
            [w.reshape(-1) for w in weights],
            axis=0
        ).astype(np.float32)

        if flat_client.size != total_params:
            raise ValueError("All clients must share the same total parameter count.")

        flat_list.append(flat_client)

    # Shape: (num_clients, total_params)
    flat_stack = np.stack(flat_list, axis=0)  
    counts_f = counts.astype(np.float32)    

    # compute weighted average along the client dimension
    lib = _load_fedavg_lib()

    
    flat_stack = np.ascontiguousarray(flat_stack, dtype=np.float32)
    counts     = np.ascontiguousarray(counts, dtype=np.int32)

    num_clients, total_params = flat_stack.shape


    out = np.empty(total_params, dtype=np.float32)

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


    # reshape the averaged 1D vector back into per-layer tensors
    averaged_weights: List[np.ndarray] = []
    offset = 0
    for shape, size, ref_w in zip(layer_shapes, layer_sizes, ref_weights):
        chunk = flat_avg[offset:offset + size]
        averaged_weights.append(chunk.reshape(shape).astype(ref_w.dtype))
        offset += size

    return averaged_weights


def prime_model(model, seq_len: int = MAX_SEQ_LENGTH):
    """
    Run a single dummy forward pass so Keras creates all weights.
    This makes set_weights/get_weights safe immediately after build_model().
    """

    if MODEL_TYPE == "lstm":
        # Create a dummy batch of one sequence with the correct length
        dummy_input = np.zeros((1, seq_len), dtype="int32")
        # Call the model once (no training) to initialize weights
        _ = model(dummy_input, training=False)
        return model
    else:
        return model