import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "CyberThreatDataset" / "cyber-threat-intelligence_all.csv"

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