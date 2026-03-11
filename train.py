import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# CONFIG

IMG_SIZE = 128
DATASET_PATH = r"C:\Users\pri12\OneDrive\Desktop\xAI proj\train\train"
EPOCHS = 15
BATCH_SIZE = 32


# LOAD DATA

def load_data(path):
    images = []
    labels = []

    for label, folder in enumerate(["false", "true"]):
        folder_path = os.path.join(path, folder)

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = img / 255.0                    # NORMALIZATION
            img = np.reshape(img, (IMG_SIZE, IMG_SIZE, 1))

            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels)

X_train, y_train = load_data(DATASET_PATH)
y_train = to_categorical(y_train, 2)

# MODEL
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# TRAIN
model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1
)

# SAVE MODEL
model.save("heart_xray_cnn.h5")
print("Model saved as heart_xray_cnn.h5")
