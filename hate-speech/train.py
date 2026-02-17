import pandas as pd
import re
import joblib
import nltk

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


nltk.download("stopwords")

STOP_WORDS = set(stopwords.words("english"))

DATA_PATH = "archive/HateSpeechDatasetBalanced.csv"



def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = [w for w in text.split() if w not in STOP_WORDS]
    return " ".join(words)



df = pd.read_csv(DATA_PATH)

df["clean"] = df["Content"].apply(clean_text)

X = df["clean"]
y = df["Label"]



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



vectorizer = TfidfVectorizer(max_features=5000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)



model = LogisticRegression(max_iter=300)
model.fit(X_train_vec, y_train)



y_pred = model.predict(X_test_vec)

print("\n📊 Model Evaluation")
print("--------------------")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))



joblib.dump(model, "hate_speech_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\n✅ Model saved as hate_speech_model.pkl")
