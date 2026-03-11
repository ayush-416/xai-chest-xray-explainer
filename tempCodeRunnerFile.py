import numpy as np
import cv2
from tensorflow.keras.models import load_model
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

# CONFIG

MODEL_PATH = "heart_xray_cnn.h5"
IMG_PATH = r"C:\Users\pri12\OneDrive\Desktop\xAI proj\test\test\true\1.png"
IMG_SIZE = 128

# LOADING MODEL

model = load_model(MODEL_PATH)


img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img / 255.0

img_rgb = np.stack((img,)*3, axis=-1)


# prediction function

def predict_fn(images):
    images = images[:,:,:,0]  # convert back to grayscale
    images = images.reshape(len(images), IMG_SIZE, IMG_SIZE, 1)
    return model.predict(images)


# Lime explainer

explainer = lime_image.LimeImageExplainer()

explanation = explainer.explain_instance(
    img_rgb,
    predict_fn,
    top_labels=1,
    hide_color=0,
    num_samples=1000
)


temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[0],
    positive_only=True,
    num_features=5,
    hide_rest=False
)


# DISPLAY

plt.imshow(mark_boundaries(temp, mask))
plt.title("LIME Explanation")
plt.axis("off")
plt.show()
