import cv2
import numpy as np
import random

def random_perspective_transform(img, max_shift=0.02):
    """
    Applies a random perspective transform to the image.
    :param img: Input image (grayscale or color).
    :param max_shift: Maximum fraction of width/height to shift corners.
    :return: Warped image.
    """
    height, width = img.shape[:2]

    # Original corner points (top-left, top-right, bottom-left, bottom-right)
    src_points = np.float32([
        [0, 0],
        [width, 0],
        [0, height],
        [width, height]
    ])

    # Randomly shift each corner within max_shift * width/height
    shift_x = max_shift * width
    shift_y = max_shift * height

    dst_points = np.float32([
        [random.uniform(0, shift_x),              random.uniform(0, shift_y)],              # top-left
        [width - random.uniform(0, shift_x),      random.uniform(0, shift_y)],              # top-right
        [random.uniform(0, shift_x),              height - random.uniform(0, shift_y)],     # bottom-left
        [width - random.uniform(0, shift_x),      height - random.uniform(0, shift_y)]      # bottom-right
    ])

    # Compute perspective transform matrix
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    # Warp the image
    warped = cv2.warpPerspective(img, M, (width, height), 
                                 flags=cv2.INTER_LINEAR, 
                                 borderMode=cv2.BORDER_REPLICATE)
    return warped

def random_local_morphological_effect(img, kernel_size=3, iterations=1, patch_count=5):
    h, w = img.shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    for _ in range(patch_count):
        # Pick top-left corner
        x1 = random.randint(0, w - 2)
        y1 = random.randint(0, h - 2)
        
        # Ensure x2 > x1 and y2 > y1
        x2 = random.randint(x1 + 1, w - 1)
        y2 = random.randint(y1 + 1, h - 1)

        # Extract the patch
        patch = img[y1:y2, x1:x2]
        
        # Randomly choose dilation or erosion
        if random.choice([True, False]):
            patch = cv2.dilate(patch, kernel, iterations=iterations)
        else:
            patch = cv2.erode(patch, kernel, iterations=iterations)
        
        # Put the patch back
        img[y1:y2, x1:x2] = patch
    
    return img

def add_random_noise(img, noise_level=10):
    """
    Adds random speckle-like noise to the image.
    :param img: Grayscale image (uint8).
    :param noise_level: Controls the amount of noise.
    :return: Noisy image (uint8).
    """
    noise = np.random.randint(-noise_level, noise_level+1, img.shape, dtype='int16')
    temp = img.astype('int16') + noise
    temp = np.clip(temp, 0, 255)
    return temp.astype('uint8')

def add_random_scribbles(img, circle_count=10, line_count=5, max_radius=5):
    """
    Draws random scribbles (small circles and lines) to simulate stray pen marks.
    :param img: Grayscale image (uint8).
    :param circle_count: Number of random circles to draw.
    :param line_count: Number of random lines to draw.
    :param max_radius: Max radius for circles.
    :return: Image with scribbles.
    """
    h, w = img.shape[:2]
    
    # Random circles
    for _ in range(circle_count):
        cx = random.randint(0, w-1)
        cy = random.randint(0, h-1)
        r = random.randint(1, max_radius)
        color = random.choice([0, 255])  # black or white scribble
        cv2.circle(img, (cx, cy), r, color, -1)
    
    # Random lines
    for _ in range(line_count):
        x1 = random.randint(0, w-1)
        y1 = random.randint(0, h-1)
        x2 = random.randint(0, w-1)
        y2 = random.randint(0, h-1)
        color = random.choice([0, 255])
        thickness = random.randint(1, 2)
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    return img

def handwritten_effect_pipeline(input_path, output_path):
    # 1. Load image in grayscale
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {input_path}")

    # 2. Check average intensity to guess background color
    #    If background is lighter, invert it (we want text as white on black for morphological ops)
    if np.mean(img) > 127:
        img = 255 - img

    # 3. Threshold for a cleaner binary image (optional)
    _, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    # 4. Apply a random perspective transform
    warped = random_perspective_transform(img_bin, max_shift=0.03)

    # 5. Apply local morphological transformations to vary thickness in patches
    warped = random_local_morphological_effect(warped, kernel_size=3, iterations=1, patch_count=5)

    # 6. Add random scribbles (like stray pen marks)
    warped = add_random_scribbles(warped, circle_count=10, line_count=5, max_radius=4)

    # 7. Add random noise
    noisy = add_random_noise(warped, noise_level=30)

    # 8. Slight blur
    blurred = cv2.GaussianBlur(noisy, (3, 3), 0)

    # 9. Re-invert so final text is black on white
    final = 255 - blurred

    # 10. Optional final threshold to keep it more “binary”
    # _, final = cv2.threshold(final, 128, 255, cv2.THRESH_BINARY)

    # 11. Save result
    cv2.imwrite(output_path, final)
    print(f"Saved handwritten-style output to {output_path}")

if __name__ == "__main__":
    input_image_path = "10003.png"     # Replace with your input
    output_image_path = "handwritten_10003.png"   # Replace with your desired output
    handwritten_effect_pipeline(input_image_path, output_image_path)