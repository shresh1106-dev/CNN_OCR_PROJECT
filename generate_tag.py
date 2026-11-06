import numpy as np
import cv2
from tensorflow.keras.datasets import mnist

# Load MNIST dataset
(_, _), (x_test, y_test) = mnist.load_data()

# Select digits for "1107"
target_digits = [1, 1, 0, 7]
indices = []
for digit in target_digits:
    idx = np.where(y_test == digit)[0][0]
    indices.append(idx)

# Combine images horizontally
selected_images = [x_test[idx] for idx in indices]
asset_tag = np.hstack(selected_images)

# Add white paper border
asset_tag = cv2.copyMakeBorder(asset_tag, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=0)

# Invert colors (dark ink on white paper)
asset_tag_paper = cv2.bitwise_not(asset_tag)

# Save the image
cv2.imwrite("mnist_tag_1107.png", asset_tag_paper)
print("Saved 'mnist_tag_1107.png' successfully in your project folder!")