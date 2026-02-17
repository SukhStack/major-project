# Pneumonia Detection Using Deep Learning

## 📌 Project Overview

This project is a **machine learning–based medical image classification system** that detects **pneumonia from chest X-ray images** using a **Convolutional Neural Network (CNN)** with **transfer learning (MobileNetV2)**.

The goal of this assignment was to understand the **complete machine learning pipeline**, from dataset handling to training, evaluation, and real-time prediction.

---

## 🎯 Learning Objectives

Through this course and project, I learned:

* Basics of **Machine Learning and Deep Learning**
* How **CNNs work for image classification**
* Use of **transfer learning** with pretrained models
* **Data preprocessing** and image augmentation
* Training and validating a neural network
* Saving and loading trained ML models
* Running **real predictions on new images**
* Setting up a proper **Python virtual environment** on Linux
* Debugging real-world dependency and compatibility issues

This project helped me move from **theory to practical implementation** of ML.

---

## 🧠 Course Contribution

The ML course provided:

* Fundamental understanding of **supervised learning**
* Knowledge of **neural networks and computer vision**
* Hands-on exposure to **TensorFlow/Keras**
* Guidance on **model evaluation and accuracy**
* Real-world problem solving using **medical datasets**

Because of the course structure, I was able to **build a complete end-to-end ML application**, not just run sample notebooks.

---

## 🗂️ Project Structure

```
pneumonia-detection/
│
├── archive/chest_xray/      # Dataset (train/val/test)
├── train.py                 # Model training script
├── predict.py               # Single image prediction script
├── pneumonia_model.h5       # Trained deep learning model
└── README.md                # Project documentation
```

---

## ⚙️ Technologies Used

* **Python 3.10**
* **TensorFlow / Keras**
* **NumPy, SciPy, Pillow**
* **ImageDataGenerator for preprocessing**
* **MobileNetV2 (Transfer Learning)**

---

## 🏥 Dataset

The model is trained on a **Chest X-ray Pneumonia dataset** containing:

* **NORMAL** lung images
* **PNEUMONIA** infected lung images

The dataset is split into:

* Training set
* Validation set
* Test set

---

## 🚀 How to Run the Project

### 1️⃣ Create and activate virtual environment

```bash
python3.10 -m venv venv
source venv/bin/activate.fish   # fish shell
```

### 2️⃣ Install dependencies

```bash
pip install tensorflow==2.15.0 numpy scipy pillow matplotlib
```

### 3️⃣ Train the model

```bash
python train.py
```

This will generate:

```
pneumonia_model.h5
```

---

### 4️⃣ Run prediction on an X-ray image

```bash
python predict.py archive/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg
```

Output example:

```
Prediction: PNEUMONIA
Confidence: 96%
```

---

## 📊 Model Performance

* Achieves **high validation accuracy (~90%+)**
* Uses **transfer learning** for efficient training
* Works on **CPU without GPU**

---

## 📚 Key Concepts Demonstrated

* Convolutional Neural Networks (CNN)
* Transfer Learning
* Binary Image Classification
* Data Augmentation
* Model Serialization & Inference
* Real-world ML debugging on Linux

---

## 🎓 Conclusion

This project demonstrates a **complete deep learning workflow** for medical image analysis.

Through this assignment, I gained:

* Practical ML implementation skills
* Confidence in building **end-to-end AI systems**
* Understanding of **real-world challenges in ML environments**

This experience forms a strong foundation for **advanced AI, computer vision, and healthcare ML applications** in the future.

---

## 👨‍💻 Author

**Sukhdeep Kaur**
Machine Learning Student & Developer
