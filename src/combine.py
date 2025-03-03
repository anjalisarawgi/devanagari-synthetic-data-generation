import os
import json
from PIL import Image
import cv2

IMAGE_DIR = "data/processed_IIT_HW_Hindi_v1/filtered/images/"
LABELS_JSON = "data/processed_IIT_HW_Hindi_v1/filtered/labels.json"
COMBINED_IMAGE_DIR = "data/processed_IIT_HW_Hindi_v1/filtered/combined/images/"
COMBINED_LABELS_JSON = "data/processed_IIT_HW_Hindi_v1/filtered/combined/labels.json"

def remove_white_gaps(image_path):
    """Removes white gaps in an image using inpainting."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    filled_image = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return filled_image


os.makedirs(COMBINED_IMAGE_DIR, exist_ok=True)
with open(LABELS_JSON, "r", encoding="utf-8") as f:
    labels_dict = json.load(f)

image_label_pairs = list(labels_dict.items())
grouped_images = [image_label_pairs[i:i+6] for i in range(0, len(image_label_pairs), 6)]

combined_labels = {}

for idx, group in enumerate(grouped_images, start=1):
    images = []
    labels = []

    for img_path, label in group:
        img = Image.open(img_path)
        images.append(img)
        labels.append(label)


    if len(images) < 2:
        continue  

    max_height = max(img.height for img in images)
    total_width = sum(img.width for img in images)
    combined_image = Image.new("RGB", (total_width, max_height), "white")
    x_offset = 0

    for img in images:
        y_offset = (max_height - img.height) // 2
        combined_image.paste(img, (x_offset, y_offset))
        x_offset += img.width

    combined_image_name = f"{idx}.jpg"
    combined_image_path = os.path.join(COMBINED_IMAGE_DIR, combined_image_name)
    combined_image.save(combined_image_path)
    cleaned_image = remove_white_gaps(combined_image_path)
    if cleaned_image is not None:
        cv2.imwrite(combined_image_path, cleaned_image)

    combined_labels[combined_image_path] = " ".join(labels)
    print(f"Saved combined image: {combined_image_path}")


with open(COMBINED_LABELS_JSON, "w", encoding="utf-8") as json_file:
    json.dump(combined_labels, json_file, ensure_ascii=False, indent=4)

print(f"Labels saved to: {COMBINED_LABELS_JSON}")