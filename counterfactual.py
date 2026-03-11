import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 128

# load trained model
model = tf.keras.models.load_model("heart_xray_cnn.h5")

# load original image
img = cv2.imread("sample.png", cv2.IMREAD_GRAYSCALE)
print(img)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img / 255.0

# load saliency map
saliency_map = cv2.imread("saliency_map.png", cv2.IMREAD_GRAYSCALE)
saliency_map = cv2.resize(saliency_map, (IMG_SIZE, IMG_SIZE))
saliency_map = saliency_map / 255.0

# create mask of important regions
mask = saliency_map > 0.7

# create counterfactual image
counterfactual = img.copy()
counterfactual[mask] = np.mean(img)

# reshape for model
original_input = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
counterfactual_input = counterfactual.reshape(1, IMG_SIZE, IMG_SIZE, 1)

# predictions
pred_original = model.predict(original_input)
pred_cf = model.predict(counterfactual_input)

print("Original prediction:", pred_original)
print("Counterfactual prediction:", pred_cf)

# visualize
cv2.imshow("Original", img)
cv2.imshow("Counterfactual", counterfactual)
cv2.waitKey(0)