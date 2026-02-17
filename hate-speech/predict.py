import sys
import re
import joblib
import nltk
from nltk.corpus import stopwords


nltk.download("stopwords")
STOP_WORDS = set(stopwords.words("english"))

MODEL_PATH = "hate_speech_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = [w for w in text.split() if w not in STOP_WORDS]
    return " ".join(words)


def predict(text):
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    clean = clean_text(text)
    vec = vectorizer.transform([clean])

    pred = model.predict(vec)[0]

    label = "HATE SPEECH" if pred == 1 else "NOT HATE SPEECH"

    print("\n🧠 Prediction Result")
    print("---------------------")
    print(f"Input Text : {text}")
    print(f"Label      : {label}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Usage: python predict.py \"your sentence here\"")
        sys.exit(1)

    input_text = " ".join(sys.argv[1:])
    predict(input_text)
