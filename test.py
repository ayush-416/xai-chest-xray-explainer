import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# CONFIG
IMG_SIZE = 128
DATASET_PATH = r"C:\Users\pri12\OneDrive\Desktop\xAI proj\test\test"

# LOAD DATA
def load_data(path):
    images = []
    labels = []

    for label, folder in enumerate(["false", "true"]):
        folder_path = os.path.join(path, folder)

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = img / 255.0
            img = np.reshape(img, (IMG_SIZE, IMG_SIZE, 1))

            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels)

X_test, y_test = load_data(DATASET_PATH)
y_test = to_categorical(y_test, 2)

# LOAD MODEL
model = load_model("heart_xray_cnn.h5")

# EVALUATE
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc * 100:.2f}%")
