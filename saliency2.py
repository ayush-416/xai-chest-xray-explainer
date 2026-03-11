import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# CONFIG

MODEL_PATH = "heart_xray_cnn.h5"
IMG_PATH = r"C:\Users\pri12\OneDrive\Desktop\xAI proj\test\test\true\1.png"
IMG_SIZE = 128
# output_path = r"C:\Users\pri12\OneDrive\Desktop\xAI proj"
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

# ROI OVERLAP CALCULATION


# Step 1: Threshold saliency 
threshold = 0.4
saliency_binary = (saliency > threshold).astype(np.float32)

heart_roi = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

heart_roi[60:110, 35:95] = 1   

# Step 3: Compute Overlap
intersection = np.sum(saliency_binary * heart_roi)
total_saliency = np.sum(saliency_binary) + 1e-8

overlap_score = intersection / total_saliency

print("Heart Overlap Score:", overlap_score)

# For multi-method extension later
overlaps = [overlap_score]

# TRUST SCORING


average_overlap = np.mean(overlaps)

if average_overlap > 0.6:
    trust = "High"
elif average_overlap > 0.4:
    trust = "Moderate"
else:
    trust = "Low"

print("Trust Level:", trust)

# Agreement score placeholder 
agreement_score = 1.0  


import json

result = {
    "prediction": "Cardiomegaly" if predicted_class == 1 else "Normal",
    "confidence": float(predictions[0][predicted_class]),
    "heart_overlap_average": float(average_overlap),
    "agreement_score": float(agreement_score),
    "trust_level": trust
}
output_path = "explanation_output.json"

with open(output_path, "w") as f:
    json.dump(result, f, indent=2)

print(f"\nSaved explanation to {output_path}")

print("\nStructured Output:")
print(json.dumps(result, indent=2))
output_path = "explanation_output.json"



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
