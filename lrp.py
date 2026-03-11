import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import innvestigate
import innvestigate.utils as iutils
from tensorflow.keras.models import load_model

# CONFIG
MODEL_PATH = "heart_xray_cnn.h5"
IMG_PATH = r"C:\Users\pri12\OneDrive\Desktop\xAI proj\test\test\true\1.png"
IMG_SIZE = 128

# LOAD MODEL
model = load_model(MODEL_PATH)

# Remove softmax
model.layers[-1].activation = tf.keras.activations.linear
model = iutils.keras.graph.model_wo_softmax(model)

# CREATE LRP ANALYZER
analyzer = innvestigate.create_analyzer(
    "lrp.z",   # LRP rule
    model
)

# LOAD IMAGE
img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise ValueError("Image not found")

img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img / 255.0

x = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)

# PREDICTION
pred = model.predict(x)
pred_class = np.argmax(pred)

print("Prediction:", "CARDIOMEGALY" if pred_class==1 else "NORMAL")
print("Confidence:", pred[0][pred_class])

# LRP ANALYSIS

analysis = analyzer.analyze(x)

# Remove batch + channel dims
heatmap = analysis[0,:,:,0]

# Normalize
heatmap = heatmap / (np.max(np.abs(heatmap)) + 1e-8)

# DISPLAY

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(img, cmap="gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.title("LRP Heatmap")
plt.imshow(heatmap, cmap="seismic")
plt.axis("off")

plt.tight_layout()
plt.show()
