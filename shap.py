import shap
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# CONFIG

MODEL_PATH = "heart_xray_cnn.h5"
IMG_PATH = r"C:\Users\pri12\OneDrive\Desktop\xAI proj\test\test\true\1.png"
IMG_SIZE = 128


# LOAD MODEL

model = load_model(MODEL_PATH)

_ = model(tf.zeros((1,128,128,1)))


img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise ValueError("Image not found")

img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img / 255.0

x = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)

# BACKGROUND DATA

background = np.zeros((1, IMG_SIZE, IMG_SIZE, 1))

# CREATING EXPLAINER

explainer = shap.GradientExplainer(model, background)


# COMPUTE SHAP VALUES

shap_values = explainer.shap_values(x)

# shap_values[0] → class 0
# shap_values[1] → class 1

# Using predicted class
pred = model.predict(x)
pred_class = np.argmax(pred)

shap_map = shap_values[pred_class][0,:,:,0]


# plotting

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(img, cmap="gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.title("SHAP Explanation")
plt.imshow(shap_map, cmap="seismic")
plt.axis("off")

plt.tight_layout()
plt.show()
