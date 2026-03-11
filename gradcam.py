import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# CONFIG

MODEL_PATH = "heart_xray_cnn.h5"
IMG_PATH = r"C:\Users\pri12\OneDrive\Desktop\xAI proj\sample\1.png"   
IMG_SIZE = 128
LAST_CONV_LAYER = "conv2d_2"   

# LOAD MODEL

model = load_model(MODEL_PATH)


# LOADING & PREPROCESS IMAGE

img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img / 255.0

input_img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)

# PREDICTION

preds = model.predict(input_img)
predicted_class = np.argmax(preds[0])

print("Prediction:", "CARDIOMEGALY" if predicted_class == 1 else "NORMAL")
print("Confidence:", preds[0][predicted_class])


# GRAD-CAM MODEL


_ = model(input_img)

# Last conv layer
conv_layer = model.get_layer("conv2d_2")

# Pre-softmax output (logits)
logits = model.get_layer("dense_1").input

grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[conv_layer.output, logits]
)


# GRAD-CAM COMPUTATION

with tf.GradientTape() as tape:
    conv_outputs, logits = grad_model(input_img, training=False)
    class_channel = logits[:, predicted_class]

grads = tape.gradient(class_channel, conv_outputs)

if grads is None:
    raise RuntimeError("Gradients are None. Grad-CAM cannot be computed.")


# Global average pooling
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
conv_outputs = conv_outputs[0]

heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
heatmap = tf.maximum(heatmap, 0)
heatmap /= tf.reduce_max(heatmap) + 1e-8



# Resizing

heatmap = cv2.resize(heatmap.numpy(), (IMG_SIZE, IMG_SIZE))
heatmap = np.uint8(255 * heatmap)

# Applying color map
heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# Convert grayscale to RGB
img_rgb = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

# Overlay heatmap
overlay = cv2.addWeighted(img_rgb, 0.6, heatmap_color, 0.4, 0)

# SAVE 

cv2.imwrite("gradcam_result.png", overlay)

plt.figure(figsize=(6,6))
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Grad-CAM Heatmap")
plt.show()
