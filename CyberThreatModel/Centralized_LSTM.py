import re
from pathlib import Path
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense

# -----------------------------
# Basic configuration
# -----------------------------

# Base directory where this script lives
BASE_DIR = Path(__file__).resolve().parent

# Path to the dataset CSV
DATA_PATH = BASE_DIR / "CyberThreatDataset" / "cyber-threat-intelligence_all.csv"

# Model / data settings
MAX_NB_WORDS = 50000      # Max vocabulary size
MAX_SEQ_LENGTH = 450      # Max tokens per sample (matches notebook choice)
TEST_SIZE = 0.2           # 80% train, 20% test
RANDOM_STATE = 42         # For reproducibility
EMBEDDING_DIM = 100       # Embedding vector size
BATCH_SIZE = 64
EPOCHS = 10


# -----------------------------
# Data loading and cleaning
# -----------------------------

def load_dataset(path: Path) -> pd.DataFrame:
    """
    Load the dataset CSV and keep only 'text' and 'label' columns.
    Fills missing labels as 'benign' and missing text as empty string.
    """
    df = pd.read_csv(path)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Expected 'text' and 'label' columns in the dataset.")

    # Fill missing labels with a safe default
    df["label"] = df["label"].fillna("benign")

    # Replace NaN text with empty strings
    df["text"] = df["text"].fillna("")

    # Keep only what we need
    return df[["text", "label"]]


def clean_text(text: str) -> str:
    """
    Simple text cleaner.
    - Lowercase
    - Remove non-alphanumeric characters
    - Collapse extra spaces
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_data(df: pd.DataFrame):
    """
    Prepare data for the LSTM model.

    Steps:
    - Clean the raw text
    - Encode labels
    - Tokenize and pad text
    - Split into train and test sets
    """
    # Clean text
    df["text_clean"] = df["text"].apply(clean_text)

    # Encode labels as integers
    label_encoder = LabelEncoder()
    y_int = label_encoder.fit_transform(df["label"])
    num_classes = len(label_encoder.classes_)

    # One-hot encode labels for softmax output
    y = to_categorical(y_int, num_classes=num_classes)

    # Tokenizer for text -> integer sequences
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(df["text_clean"].values)

    # Convert cleaned text to integer sequences
    sequences = tokenizer.texts_to_sequences(df["text_clean"].values)

    # Pad all sequences to the same length
    X = pad_sequences(sequences, maxlen=MAX_SEQ_LENGTH)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_int
    )

    return X_train, X_test, y_train, y_test, num_classes, tokenizer, label_encoder


# -----------------------------
# Model definition
# -----------------------------

def build_lstm_model(
    max_words: int,
    embedding_dim: int,
    input_length: int,
    num_classes: int
) -> Sequential:
    """
    Baseline LSTM model for multi-class text classification.
    """
    model = Sequential()

    # Embedding layer: learns word vectors from scratch
    model.add(Embedding(
        input_dim=max_words,
        output_dim=embedding_dim,
        input_length=input_length
    ))

    # LSTM layer:
    # - 150 units (as in the notebook)
    # - dropout and recurrent_dropout for regularization
    model.add(LSTM(150, dropout=0.2, recurrent_dropout=0.2))

    # Extra dropout before final layer
    model.add(Dropout(0.2))

    # Output layer: one unit per class, softmax for probabilities
    model.add(Dense(num_classes, activation="softmax"))

    # Standard classification setup
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    return model


# -----------------------------
# Train and evaluate helpers
# -----------------------------

def train(
    model: Sequential,
    X_train,
    y_train,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS
):
    """
    Train the given model on the training data.
    """
    print("\n[TRAIN] Starting training...")
    train_history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,  # small validation slice from training data
        verbose=1
    )
    return train_history


def evaluate(model: Sequential, X_test, y_test):
    """
    Evaluate the trained model on the test set.
    """
    print("\n[EVAL] Evaluating on test set...")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[EVAL] Test Loss: {loss:.4f}")
    print(f"[EVAL] Test Accuracy: {acc:.4f}")
    return loss, acc


# -----------------------------
# Script entry point
# -----------------------------

if __name__ == "__main__":
    print("[INFO] Loading dataset...")
    df = load_dataset(DATA_PATH)

    print("[INFO] Preparing data...")
    X_train, X_test, y_train, y_test, num_classes, tokenizer, label_encoder = prepare_data(df)

    # Minimal sanity checks
    print("\n[INFO] Data Shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_test:  {y_test.shape}")
    print(f"\n[INFO] Number of classes: {num_classes}")
    print(f"[INFO] Vocab size (capped): {min(len(tokenizer.word_index) + 1, MAX_NB_WORDS)}")
    print(f"[INFO] Sequence length: {X_train.shape[1]} (expected {MAX_SEQ_LENGTH})")

    print("\n[INFO] Building LSTM model...")
    model = build_lstm_model(
        max_words=MAX_NB_WORDS,
        embedding_dim=EMBEDDING_DIM,
        input_length=MAX_SEQ_LENGTH,
        num_classes=num_classes
    )

    print("\n[INFO] Model Summary:")
    model.summary()

    # Train and evaluate baseline model
    train_history = train(model, X_train, y_train)
    evaluate(model, X_test, y_test)
