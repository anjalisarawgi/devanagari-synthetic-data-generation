import cv2
import numpy as np
import os
import json

BASE_DIR = "data/IIT_HW_Hindi_v1/"
LABELS_JSON = "data/processed_IIT_HW_Hindi_v1/filtered/labels.json"


def draw_text_bounding_box(image_path, output_path, padding=15):
    full_image_path = os.path.join(BASE_DIR, image_path)
    image = cv2.imread(full_image_path)
    img_height, img_width = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    # finding contours (all)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"No text found in the image{full_image_path} to crop. Saving the original image.")
        cv2.imwrite(output_path, image)
        return output_path

    # merge all contours into one bounding box
    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = 0, 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if x < x_min:
            x_min = x
        if y < y_min:
            y_min = y
        if x + w > x_max:
            x_max = x + w
        if y + h > y_max:
            y_max = y + h

    # padding
    x_padded = max(0, x_min - padding)
    y_padded = max(0, y_min - padding)
    w_padded = (x_max - x_min) + 2 * padding
    h_padded = (y_max - y_min) + 2 * padding

    if x_padded + w_padded > img_width:
        w_padded = img_width - x_padded
    if y_padded + h_padded > img_height:
        h_padded = img_height - y_padded

    cropped_image = image[y_padded : y_padded + h_padded, x_padded : x_padded + w_padded]

    cv2.imwrite(output_path, cropped_image)
    print(f"Cropped bounding box saved to: {output_path}")
    return output_path

##############

input_file = "data/processed_IIT_HW_Hindi_v1/filtered/filtered_sorted_train.txt"
output_folder = "data/processed_IIT_HW_Hindi_v1/filtered/images/"

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

labels_dict = {}

for line in lines:
    image_path, label = line.strip().split(" ", 1) 
    
    parts = image_path.split("/")
    folder_number = parts[-2]  # This is '151' in "HindiSeg/train/1/151/1.jpg"
    filename = parts[-1]       # this is '1.jpg' in "HindiSeg/train/1/151/1.jpg"
    new_filename = f"{folder_number}_{filename}"

    cropped_path = os.path.join(output_folder, os.path.basename(new_filename))
    draw_text_bounding_box(image_path, cropped_path, padding=15)

    if cropped_path:
        labels_dict[cropped_path] = label

with open(LABELS_JSON, "w", encoding="utf-8") as f:
    json.dump(labels_dict, f, ensure_ascii=False, indent=4)
print(f"Labels saved to: {LABELS_JSON}")


