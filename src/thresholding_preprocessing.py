import cv2
import numpy as np
from PIL import Image
import glob
import os
import json

input_dir = "data/processed_IIT_HW_Hindi_v1/filtered_4/combined/images/"
output_dir = "data/processed_IIT_HW_Hindi_v1/filtered_4/combined/processed/images/"
labels_file = "data/processed_IIT_HW_Hindi_v1/filtered_4/combined/labels.json"
labels_output_dir =  "data/processed_IIT_HW_Hindi_v1/filtered_4/combined/processed/"

with open(labels_file, "r", encoding="utf-8") as f:
    labels = json.load(f)

processed_labels = {}

def process_image(input_path, output_path):
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    _, thresholded = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((2,2), np.uint8)
    thickened_text = cv2.dilate(thresholded, kernel, iterations=2)
    inverted = 255 - thickened_text
    cv2.imwrite(output_path, inverted)

for input_path in glob.glob(os.path.join(input_dir, "*.jpg")):
    filename = os.path.basename(input_path)
    output_path = os.path.join(output_dir, filename)
    process_image(input_path, output_path)
    print("Processed:", filename)

    original_image_path = os.path.join(input_dir, filename)
    processed_image_path = os.path.join(output_dir, filename)
    if original_image_path in labels:
        processed_labels[processed_image_path] = labels[original_image_path]

# save labels to the new directory
labels_json_path = os.path.join(labels_output_dir, "labels.json")
with open(labels_json_path, "w", encoding="utf-8") as f:
    json.dump(processed_labels, f, ensure_ascii=False, indent=4)

print("Processing complete! All images saved in:", output_dir)
print("Labels JSON saved at:", labels_json_path)
