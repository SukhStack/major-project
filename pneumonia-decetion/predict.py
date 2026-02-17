import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

IMG_SIZE = 224
MODEL_PATH = "pneumonia_model.h5"


def predict_image(img_path):
    
    model = tf.keras.models.load_model(MODEL_PATH)

   
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

   
    prediction = model.predict(img_array)[0][0]
    confidence = float(prediction if prediction > 0.5 else 1 - prediction) * 100

    if prediction > 0.5:
        label = "PNEUMONIA"
    else:
        label = "NORMAL"

    print("\n🩺 Prediction Result")
    print("---------------------")
    print(f"Label      : {label}")
    print(f"Confidence : {confidence:.2f}%\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Usage: python predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    predict_image(image_path)
