import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_and_preprocess(path, sample_size=200000):
    print("Loading dataset...")
    df = pd.read_csv(path, low_memory=False)

    df.columns = df.columns.str.strip()

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    print(f"Original dataset shape: {df.shape}")

    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        print(f"Sampled dataset shape: {df.shape}")

    # Encode labels
    le = LabelEncoder()
    df['Label'] = le.fit_transform(df['Label'])

    # 🔥 SAVE LABEL ENCODER
    os.makedirs("../models", exist_ok=True)
    joblib.dump(le, "../models/label_encoder.pkl")

    X = df.drop('Label', axis=1)
    y = df['Label']

    return train_test_split(X, y, test_size=0.2, random_state=42)