import numpy as np
import re
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "CyberThreatDataset" / "cyber-threat-intelligence_all.csv"
MAX_NB_WORDS = 50000     # max vocabulary size
MAX_SEQ_LENGTH = 450     # max tokens per sample (based on python notebook)
TEST_SIZE = 0.2          # 80% train, 20% test
RANDOM_STATE = 42        # for reproducibility (based on python notebook)
EMBEDDING_DIM = 100      # embedding vector size
BATCH_SIZE = 64
EPOCHS = 5

def load_dataset(path: str):
    df = pd.read_csv(path)

    # Making sure these columns exist
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Expected 'text' and 'label' columns in the dataset.")

    # Fill missing labels with a default benign label
    df["label"] = df["label"].fillna("benign")

    # Drop/Clean the NAN values
    df["text"] = df["text"].fillna("")
    df = df[["text", "label"]]

    return df

testDf = load_dataset(DATA_PATH)

'''
Checking out the Data...

print("\n[1] Data Shape (rows, columns):")
print(testDf.shape)

print("\n[2] Columns in Dataset:")
print(list(testDf.columns))

print("\n[3] Missing Value Count:")
print(testDf.isnull().sum())

print("\n[4] Label Counts:")
print(testDf["label"].value_counts())

print("\n[5] Unique Label Names:")
print(sorted(testDf["label"].unique()))

print("\n[6] Example Row):")
print(testDf.sample(5, random_state=42))

empty_text_count = (testDf["text"].str.strip() == "").sum()
print("\n[7] Number of Rows with Empty Text:")
print(empty_text_count)
'''

#Simple Text Cleaner
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    
    # Turn it lowercase.
    text = text.lower()

    # Get rid of the non-alphas
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Get rid of extra spaces.
    text = re.sub(r"\s+", " ", text).strip()
    return text

def prepare_data(df: pd.DataFrame):
    # Clean text
    df["text_clean"] = df["text"].apply(clean_text)

    # Encode labels to integers
    label_encoder = LabelEncoder()
    y_int = label_encoder.fit_transform(df["label"])
    num_classes = len(label_encoder.classes_)

    # One-hot encode labels for softmax
    y = to_categorical(y_int, num_classes=num_classes)

    # Tokenize text
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(df["text_clean"].values)

    # Convert text to sequences of integers
    sequences = tokenizer.texts_to_sequences(df["text_clean"].values)

    # Pad sequences to fixed length
    X = pad_sequences(sequences, maxlen=MAX_SEQ_LENGTH)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_int
    )

    return X_train, X_test, y_train, y_test, num_classes, tokenizer, label_encoder

X_train, X_test, y_train, y_test, num_classes, tokenizer, label_encoder = prepare_data(testDf)

'''
Prepping the data...

print("\n[2] Shapes:")
print(f"  X_train: {X_train.shape}")
print(f"  y_train: {y_train.shape}")
print(f"  X_test:  {X_test.shape}")
print(f"  y_test:  {y_test.shape}")

print("\n[3] Number of Classes:")
print(f"  {num_classes}")

print("\n[4] Tokenizer Vocab Size:")
print(f"  {min(len(tokenizer.word_index) + 1, MAX_NB_WORDS)}")

print("\n[5] Sequence Length Check (should equal MAX_SEQ_LENGTH):")
print(f"  X_train[0] length: {len(X_train[0])}")
print(f"  X_test[0]  length: {len(X_test[0])}")
'''

def build_lstm_model(max_words: int,
                     embedding_dim: int,
                     input_length: int,
                     num_classes: int) -> Sequential:

    model = Sequential()
    # Embedding: learns word vectors from scratch
    model.add(Embedding(
        input_dim=max_words,
        output_dim=embedding_dim,
        input_length=input_length
    ))

    # LSTM layer:
    #  - 150 units like the notebook
    #  - dropout: regularization on inputs
    #  - recurrent_dropout: regularization on recurrent connections
    model.add(LSTM(150, dropout=0.2, recurrent_dropout=0.2))

    # Dropout before final layer to reduce overfitting
    model.add(Dropout(0.2))

    # Output layer:
    #  - num_classes units
    #  - softmax for multi-class probabilities
    model.add(Dense(num_classes, activation="softmax"))

    # Compile with standard settings for classification
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    return model

model = build_lstm_model(
    max_words=MAX_NB_WORDS,
    embedding_dim=EMBEDDING_DIM,
    input_length=MAX_SEQ_LENGTH,
    num_classes=num_classes
)

'''
#Building the model...
print(model.summary())
'''

def train(model,
          X_train,
          y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS):
    
    #Training the Model, straightforward. 
    train_history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        verbose=1
    )
    return train_history

def evaluate(model, X_test, y_test):
    print("\nTesting out the model...")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {acc:.4f}")
    return loss, acc

train(model, X_train, y_train)
evaluate(model, X_test, y_test)