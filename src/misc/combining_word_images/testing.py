import cv2
import numpy as np

img = cv2.imread('5.jpg', cv2.IMREAD_GRAYSCALE)

# Threshold (any pixel above e.g. 200 can be considered "white")
_, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

# Find coordinates of all non-zero (white) pixels in the thresholded image
coords = np.column_stack(np.where(thresh > 0))
# Get bounding box: top-left (x_min, y_min) and bottom-right (x_max, y_max)
y_min, x_min = coords.min(axis=0)
y_max, x_max = coords.max(axis=0)

# Crop
cropped = img[y_min:y_max+1, x_min:x_max+1]

# save the cropped image
cv2.imwrite('cropped_image.jpg', cropped)