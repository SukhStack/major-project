import os
import joblib
from xgboost import XGBClassifier
from preprocess import load_and_preprocess

X_train, X_test, y_train, y_test = load_and_preprocess("../data/CICIDS2017.csv")

print("Training XGBoost with CUDA GPU...")

model = XGBClassifier(
    tree_method="hist",
    device="cuda",
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="mlogloss"
)

model.fit(X_train, y_train)

os.makedirs("../models", exist_ok=True)
joblib.dump(model, "../models/xgb_gpu_model.pkl")

print("Model trained and saved successfully.")