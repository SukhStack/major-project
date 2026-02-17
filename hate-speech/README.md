# Hate Speech Detection Using Machine Learning

## 📌 Project Overview

This project is a **Natural Language Processing (NLP)** based machine learning system that detects whether a given text contains **hate speech or not**.

The goal of this assignment was to understand the **complete ML workflow** for text classification, including:

* Data preprocessing
* Feature extraction
* Model training
* Evaluation
* Real-time prediction

---

## 🎯 What This Project Does

* Takes a **sentence or text input**
* Processes and cleans the text
* Converts text into **numerical features using TF-IDF**
* Uses a **Logistic Regression classifier**
* Predicts:

```
HATE SPEECH
or
NOT HATE SPEECH
```

---

## 🧠 What I Learned From This Course

Through this machine learning course and assignment, I learned:

* Basics of **Natural Language Processing (NLP)**
* How **text cleaning and preprocessing** works
* Using **TF-IDF vectorization** to convert text into numbers
* Training a **supervised classification model**
* Evaluating model performance using **accuracy and classification report**
* Saving and loading trained ML models
* Running **real-time predictions from terminal**
* Setting up a **Python virtual environment on Linux**
* Solving **dependency and installation issues** in real ML projects

This project helped me move from **theoretical learning** to **practical implementation**.

---

## ⚠️ Challenges and Mistakes I Faced

During this project, I encountered several real-world issues:

* Missing Python libraries like **scikit-learn, nltk, pandas**
* Environment problems due to **Python version incompatibility**
* Learning how to properly use **virtual environments**
* Understanding **dataset structure and preprocessing**
* Debugging errors step-by-step instead of giving up

These mistakes were actually **important learning experiences** and helped me understand how real ML development works.

---

## 🗂️ Project Structure

```
hate-speech-detection/
│
├── HateSpeechDatasetBalanced.csv   # Dataset
├── train.py                        # Model training script
├── predict.py                      # Prediction script
├── hate_speech_model.pkl           # Saved ML model
├── vectorizer.pkl                  # TF-IDF vectorizer
└── README.md                       # Documentation
```

---

## ⚙️ Technologies Used

* **Python 3.10**
* **Scikit-learn**
* **Pandas**
* **NLTK**
* **TF-IDF Vectorization**
* **Logistic Regression**

---

## 📊 Model Performance

* Achieves **good accuracy (~85–95%)** depending on dataset split
* Works on **CPU without GPU**
* Provides **instant prediction from terminal input**

---

## 🚀 How to Run This Project

### 1️⃣ Activate virtual environment

```bash
source venv/bin/activate.fish
```

### 2️⃣ Install dependencies

```bash
pip install scikit-learn pandas nltk joblib
```

### 3️⃣ Train the model

```bash
python train.py
```

This creates:

```
hate_speech_model.pkl
vectorizer.pkl
```

---

### 4️⃣ Run prediction

```bash
python predict.py "your sentence here"
```

Example:

```bash
python predict.py "you are stupid"
```

Output:

```
HATE SPEECH
```

---

## 🎓 Conclusion

This assignment demonstrates a **complete end-to-end NLP machine learning pipeline**.

From this project, I gained:

* Practical ML development experience
* Confidence in handling **real errors and debugging**
* Understanding of **text classification using ML**

This forms a strong base for **advanced NLP, AI, and real-world ML applications** in the future.

---

## 👨‍💻 Student Author

**Sukhdeep Kaur**
Machine Learning Student
