import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

st.set_page_config(page_title="AI Network Intrusion Detection", layout="wide")

st.title("🔐 AI Network Intrusion Detection System (NIDS)")

# 🔥 Proper base directory handling
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "xgb_gpu_model.pkl")
encoder_path = os.path.join(BASE_DIR, "models", "label_encoder.pkl")

if not os.path.exists(model_path):
    st.error("Model not found. Train model first.")
    st.stop()

if not os.path.exists(encoder_path):
    st.error("Label encoder not found. Retrain model.")
    st.stop()

model = joblib.load(model_path)
le = joblib.load(encoder_path)

if not os.path.exists(model_path):
    st.error("Model not found. Train model first.")
    st.stop()

model = joblib.load(model_path)
le = joblib.load(encoder_path)

uploaded_file = st.file_uploader("📂 Upload Network Traffic CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, low_memory=False)
    df.columns = df.columns.str.strip()

    if 'Label' in df.columns:
        df = df.drop('Label', axis=1)

    st.subheader("📊 Raw Data Preview")
    st.dataframe(df.head())

    # Predict
    predictions = model.predict(df)

    decoded_predictions = le.inverse_transform(predictions.astype(int))
    df['Prediction'] = decoded_predictions

    st.subheader("🚨 Prediction Results")

    col1, col2, col3 = st.columns(3)

    total = len(df)
    attack_count = sum(decoded_predictions != "BENIGN")
    normal_count = sum(decoded_predictions == "BENIGN")

    col1.metric("Total Records", total)
    col2.metric("Attacks Detected", attack_count)
    col3.metric("Normal Traffic", normal_count)

    st.subheader("📈 Attack Distribution")
    st.bar_chart(df['Prediction'].value_counts())

    st.subheader("📋 Sample Predictions")
    st.dataframe(df.head())

    # Feature importance
    st.subheader("🧠 Top Feature Importance")

    import matplotlib.pyplot as plt

    importance = model.feature_importances_
    feature_names = df.drop("Prediction", axis=1).columns

    feat_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False).head(20)

    fig, ax = plt.subplots()
    ax.barh(feat_df["Feature"], feat_df["Importance"])
    ax.invert_yaxis()

    st.pyplot(fig)