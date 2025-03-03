import cv2
import numpy as np

def draw_text_bounding_box(image_path, output_path, padding=15):

    image = cv2.imread(image_path)
    img_height, img_width = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    # finding contours (all)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No text found in the image")
        return

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


if __name__ == "__main__":
    input_image = "4.jpg" 
    output_image = "text_with_box_4.jpg"
    draw_text_bounding_box(input_image, output_image, padding=15)


# import cv2
# import numpy as np

# def draw_text_bounding_box(image_path, output_path, padding):
#     image = cv2.imread(image_path)
#     img_height, img_width = image.shape[:2]

#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#     thresh = cv2.dilate(thresh, kernel, iterations=1)

#     contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     largest_contour = max(contours, key=cv2.contourArea)
#     x, y, w, h = cv2.boundingRect(largest_contour)

#     # padding for the bounding box
#     x_padded = x - padding
#     y_padded = y - padding
#     w_padded = w + 2 * padding
#     h_padded = h + 2 * padding

#     x_padded = max(0, x_padded)
#     y_padded = max(0, y_padded)

#     if x_padded + w_padded > img_width:
#         w_padded = img_width - x_padded
#     if y_padded + h_padded > img_height:
#         h_padded = img_height - y_padded

#     # cv2.rectangle(image,
#     #               (x_padded, y_padded),
#     #               (x_padded + w_padded, y_padded + h_padded),
#     #               (0, 255, 0),
#     #               2)

#     # cv2.imwrite(output_path, image)
#     # print(f"Bounding box drawn and output saved to: {output_path}")

#     ### Crop the image
#     cropped_image = image[y_padded:y_padded + h_padded, x_padded:x_padded + w_padded]
#     cv2.imwrite(output_path, cropped_image)
#     print(f"Bounding box cropped and output saved to: {output_path}")


# if __name__ == "__main__":
#     input_image = "test_2.png" 
#     output_image = "text_with_box_4.jpg"
#     draw_text_bounding_box(input_image, output_image, padding=15)

