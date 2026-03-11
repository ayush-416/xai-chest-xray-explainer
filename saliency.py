import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# CONFIG

MODEL_PATH = "heart_xray_cnn.h5"
IMG_PATH = r"C:\Users\pri12\OneDrive\Desktop\xAI proj\test\test\true\1.png"
IMG_SIZE = 128

# LOAD MODEL

model = load_model(MODEL_PATH)

# Force build 
_ = model(tf.zeros((1, 128, 128, 1)))


# LOADING & PREPROCESSING IMAGE

img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise ValueError("Image not found")

img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img / 255.0

input_img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
input_tensor = tf.convert_to_tensor(input_img, dtype=tf.float32)


with tf.GradientTape() as tape:
    tape.watch(input_tensor)
    predictions = model(input_tensor)
    predicted_class = tf.argmax(predictions[0])
    class_score = predictions[:, predicted_class]

# Gradient of class score 
grads = tape.gradient(class_score, input_tensor)

saliency = tf.abs(grads)[0]

saliency = tf.reduce_max(saliency, axis=-1)

# Normalizing for visualization
saliency = saliency / (tf.reduce_max(saliency) + 1e-8)

saliency = saliency.numpy()

# VISUALIZATION
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(img, cmap='gray')
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Saliency Map")
plt.imshow(saliency, cmap='hot')
plt.axis("off")

plt.tight_layout()
plt.show()
