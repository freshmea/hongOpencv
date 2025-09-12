# 0202.py
import cv2
import matplotlib.pyplot as plt
import numpy as np

with np.load("data/0201_data50.npz") as X:
    x_train = X['x_train'].astype(np.float32)
    y_train = X['y_train'].astype(np.int32)
    height, width = X['size']

print("x_tain", x_train)
print("y_train", y_train)
print("height, width", height, width)
