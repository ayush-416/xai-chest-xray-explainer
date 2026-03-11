import cv2
import numpy as np
from tensorflow.keras.models import load_model


# CONFIG

MODEL_PATH = "heart_xray_cnn.h5"
IMG_PATH = r"C:\Users\pri12\OneDrive\Desktop\xAI proj\sample\1.png"   
IMG_SIZE = 128


# LOAD MODEL

model = load_model(MODEL_PATH)


# LOAD & PREPROCESS IMAGE

img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise ValueError("Image not found or path is incorrect")

# Normalize
img = img / 255.0

# Resize (safety, even if already 128x128)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

# Add batch & channel dimensions
img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)


# PREDICTION

prediction = model.predict(img)

# Softmax outputs
prob_false = prediction[0][0]
prob_true = prediction[0][1]


# OUTPUT

if prob_true > prob_false:
    print("Prediction: CARDIOMEGALY (True)")
    print(f"Confidence: {prob_true * 100:.2f}%")
else:
    print("Prediction: NORMAL (False)")
    print(f"Confidence: {prob_false * 100:.2f}%")
