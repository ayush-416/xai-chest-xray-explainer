import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Activation

# CONFIG

MODEL_PATH = "heart_xray_cnn.h5"
IMG_PATH = r"C:\Users\pri12\OneDrive\Desktop\xAI proj\test\test\true\1.png"  
IMG_SIZE = 128
LAST_CONV_LAYER = "conv2d_2"


# LOAD ORIGINAL MODEL

model = load_model(MODEL_PATH)
# Force model build
_ = model(tf.zeros((1, 128, 128, 1)))

# Get last conv layer
conv_layer = model.get_layer("conv2d_2")

# Build Grad-CAM model (NO model.output!)
grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[conv_layer.output, model.outputs[0]]
)


# LOAD & PREPROCESS IMAGE

img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise ValueError("Image not found")

img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img / 255.0
input_img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)


# PREDICTION

preds = model.predict(input_img)
predicted_class = np.argmax(preds[0])

print("Prediction:", "CARDIOMEGALY" if predicted_class == 1 else "NORMAL")
print("Confidence:", preds[0][predicted_class])


# GRAD-CAM

with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(input_img)
    loss = predictions[:, predicted_class]

grads = tape.gradient(loss, conv_outputs)

# This WILL NOT be None now
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

conv_outputs = conv_outputs[0]
heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

# Normalize
heatmap = tf.maximum(heatmap, 0)
heatmap /= tf.reduce_max(heatmap) + 1e-8


# VISUALIZATION

heatmap = cv2.resize(heatmap.numpy(), (IMG_SIZE, IMG_SIZE))
heatmap = np.uint8(255 * heatmap)
heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

img_rgb = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
overlay = cv2.addWeighted(img_rgb, 0.6, heatmap_color, 0.4, 0)

cv2.imwrite("gradcam_result.png", overlay)

plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Grad-CAM")
plt.show()
