import joblib
from sklearn.metrics import classification_report, confusion_matrix
from preprocess import load_and_preprocess

X_train, X_test, y_train, y_test = load_and_preprocess("../data/CICIDS2017.csv")

model = joblib.load("../models/xgb_gpu_model.pkl")

y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))