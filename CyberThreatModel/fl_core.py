# fl_core.py — shared core utilities
# data prep, model build, (de)serialization, and FedAvg.
# The server and client both import from here.

from __future__ import annotations

import os, io, base64, re
# Force CPU and keep TensorFlow chatty logs down
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense


# ---------- Paths / constants (match centralized LSTM script) ----------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "CyberThreatDataset" / "cyber-threat-intelligence_all.csv"

MAX_NB_WORDS   = 50000
MAX_SEQ_LENGTH = 450
TEST_SIZE      = 0.2
RANDOM_STATE   = 42
EMBEDDING_DIM  = 100
BATCH_SIZE     = 64
EPOCHS         = 20  # roles can override


# ---------- Model builders ----------
def build_lstm_model(max_words: int, embedding_dim: int, seq_len: int, num_classes: int) -> Sequential:
    """
    Exact same model as centralized:
      Embedding -> LSTM(150, dropout=0.2, recurrent_dropout=0.2) -> Dropout(0.2) -> Dense(softmax)
    """
    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=seq_len))
    model.add(LSTM(150, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation="softmax"))

    # steps_per_execution helps amortize Python overhead on CPU
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"], steps_per_execution=64)
    return model


def prime_model(model: Sequential, seq_len: int = MAX_SEQ_LENGTH) -> Sequential:
    """
    We run one dummy forward pass so Keras creates the weights immediately.
    This lets us call set_weights/get_weights safely right after build.
    """
    _ = model(np.zeros((1, seq_len), dtype="int32"), training=False)
    return model


# ---------- Text cleaning / data prep ----------
def _clean_text(s: str) -> str:
    if not isinstance(s, str): 
        return ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_text_label_data(path: Path) -> pd.DataFrame:
    """
    Load the CSV and make sure we have the expected 'text' and 'label' columns.
    """
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Expected 'text' and 'label' columns.")
    df["label"] = df["label"].fillna("benign")
    df["text"]  = df["text"].fillna("")
    return df[["text", "label"]]


def prepare_encoded_padded_data(df: pd.DataFrame):
    """
    End-to-end prep for our LSTM:
    - Clean text
    - Label-encode to ints, then one-hot
    - Tokenize and pad to MAX_SEQ_LENGTH
    - Train/test split (stratified)
    """
    df = df.copy()
    df["text_clean"] = df["text"].apply(_clean_text)

    label_enc = LabelEncoder()
    y_int = label_enc.fit_transform(df["label"])
    num_classes = len(label_enc.classes_)
    y = to_categorical(y_int, num_classes=num_classes)

    tok = Tokenizer(num_words=MAX_NB_WORDS, oov_token="<OOV>")
    tok.fit_on_texts(df["text_clean"].values)

    seqs = tok.texts_to_sequences(df["text_clean"].values)
    X = pad_sequences(seqs, maxlen=MAX_SEQ_LENGTH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_int
    )
    return X_train, X_test, y_train, y_test, num_classes, tok, label_enc


# ---------- (De)serialization helpers ----------
def arrays_to_b64(*arrays: np.ndarray) -> str:
    """
    Convert one or more numpy arrays into a compressed base64 string.
    We use this to ship shards over HTTP.
    """
    buf = io.BytesIO()
    np.savez_compressed(buf, *arrays)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def arrays_from_b64(s: str):
    """
    Convert our compressed base64 back into numpy arrays.
    Returns a list in the order they were saved.
    """
    data = base64.b64decode(s.encode("utf-8"))
    buf = io.BytesIO(data)
    npz = np.load(buf, allow_pickle=True)
    return [npz[k] for k in sorted(npz.files, key=lambda x: int(x.split("_")[-1]))]


def weights_to_b64(weights: list[np.ndarray]) -> str:
    return arrays_to_b64(*weights)


def weights_from_b64(s: str) -> list[np.ndarray]:
    return arrays_from_b64(s)


# ---------- FedAvg ----------
def fedavg_weighted_average(client_updates: list[tuple[list[np.ndarray], int]]) -> list[np.ndarray]:
    """
    Weighted FedAvg: sum_k (n_k * w_k) / sum_k n_k
    NOTE: This is the exact place we plan to swap for our CUDA C++ aggregator later.
    """
    if not client_updates:
        raise ValueError("No client updates.")
    weight_lists = [w for (w, n) in client_updates]
    counts       = np.array([n for (w, n) in client_updates], dtype=np.float32)
    total = float(np.sum(counts))
    if total <= 0:
        raise ValueError("Total samples is zero.")

    num_layers = len(weight_lists[0])
    out = []
    for i in range(num_layers):
        # Stack same layer across all clients
        stack = np.stack([w[i] for w in weight_lists], axis=0)  # (K, ...)
        # Flatten so the weighted sum is simple
        flat  = stack.reshape(stack.shape[0], -1)               # (K, P)
        # Weighted average per-parameter
        avg   = np.dot(counts, flat) / total                    # (P,)
        # Back to original shape for that layer
        out.append(avg.reshape(stack.shape[1:]))
    return out
