import numpy as np
import cv2
import tensorflow as tf

print("NumPy version:", np.__version__)
print("OpenCV version:", cv2.__version__)
print("TensorFlow version:", tf.__version__)

print("GPU Available:", tf.config.list_physical_devices('GPU'))
