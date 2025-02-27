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


def add_paper_texture(text_img, texture_img, alpha=0.2):
    """
    Blend text_img onto texture_img to simulate paper texture.
    :param text_img: (H x W) final text image (uint8).
    :param texture_img: (H x W) paper texture image (uint8).
    :param alpha: blending factor for text overlay.
    :return: Blended image (uint8).
    """
    # Ensure both images are same size
    h, w = text_img.shape[:2]
    texture_resized = cv2.resize(texture_img, (w, h))

    # Convert to float for blending
    text_float = text_img.astype(np.float32)
    texture_float = texture_resized.astype(np.float32)

    # Normalize texture if desired to keep background from overshadowing text
    texture_norm = cv2.normalize(texture_float, None, 0, 255, cv2.NORM_MINMAX)

    # Blend: final = (1 - alpha)*texture + alpha*text
    blended = (1 - alpha) * texture_norm + alpha * text_float
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return blended

def add_paper_texture(text_img, texture_img, alpha=0.2):
    """
    Blend text_img onto texture_img to simulate paper texture.
    :param text_img: (H x W) final text image (uint8).
    :param texture_img: (H x W) paper texture image (uint8).
    :param alpha: blending factor for text overlay.
    :return: Blended image (uint8).
    """
    # Ensure both images are same size
    h, w = text_img.shape[:2]
    texture_resized = cv2.resize(texture_img, (w, h))

    # Convert to float for blending
    text_float = text_img.astype(np.float32)
    texture_float = texture_resized.astype(np.float32)

    # Normalize texture if desired to keep background from overshadowing text
    texture_norm = cv2.normalize(texture_float, None, 0, 255, cv2.NORM_MINMAX)

    # Blend: final = (1 - alpha)*texture + alpha*text
    blended = (1 - alpha) * texture_norm + alpha * text_float
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return blended

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

def elastic_transform(image, alpha=36, sigma=4, random_state=None):
    """
    Performs an elastic distortion on a grayscale image.
    :param image: Grayscale (H x W) numpy array.
    :param alpha: Scaling factor that controls how far pixels are moved.
    :param sigma: Standard deviation for Gaussian smoothing of the displacement fields.
    :param random_state: Optional np.random.RandomState for reproducibility.
    :return: Distorted image (same shape as input).
    """
    if random_state is None:
        random_state = np.random.RandomState(None)
    
    # Store shape
    h, w = image.shape[:2]
    
    # Generate random displacement fields (dx, dy) in the range [-1, 1]
    dx = random_state.rand(h, w) * 2 - 1  # [-1..1]
    dy = random_state.rand(h, w) * 2 - 1  # [-1..1]
    
    # Smooth them with a Gaussian filter (cv2.GaussianBlur or manual)
    # This ensures the displacement is coherent (smooth) rather than pixel-by-pixel noise
    dx = cv2.GaussianBlur(dx, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT101)
    dy = cv2.GaussianBlur(dy, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT101)
    
    # Scale the displacement fields
    dx *= alpha
    dy *= alpha
    
    # Build a meshgrid of (x, y) coordinates
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    
    # Distort the coordinates
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    
    # Remap the original image to the new coordinates
    distorted = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    return distorted

def ink_bleed_effect(img, blur_ksize=3, intensity=0.5):
    """
    Simulates slight ink bleed by merging a blurred version with the original.
    :param img: Binary or near-binary text image (uint8).
    :param blur_ksize: Kernel size for blurring.
    :param intensity: How strongly to blend the blurred edges.
    :return: Modified image.
    """
    # Convert to float for blending
    img_float = img.astype(np.float32)
    blurred = cv2.GaussianBlur(img_float, (blur_ksize, blur_ksize), 0)

    # Weighted sum: new = (1 - intensity)*original + intensity*blurred
    out = (1 - intensity) * img_float + intensity * blurred
    out = np.clip(out, 0, 255).astype(np.uint8)

    return out

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
    warped = elastic_transform(warped, alpha=40, sigma=5)

    # 5. Apply local morphological transformations to vary thickness in patches
    warped = random_local_morphological_effect(warped, kernel_size=3, iterations=1, patch_count=5)

    # 6. Add random scribbles (like stray pen marks)
    warped = add_random_scribbles(warped, circle_count=10, line_count=5, max_radius=4)

    ink_bleed = ink_bleed_effect(warped, blur_ksize=3, intensity=0.5)

    # 7. Add random noise
    noisy = add_random_noise(ink_bleed, noise_level=30)

    # 8. Slight blur
    blurred = cv2.GaussianBlur(noisy, (3, 3), 0)

    # 9. Re-invert so final text is black on white
    final = 255 - blurred

    # 10. Optional final threshold to keep it more “binary”
    # _, final = cv2.threshold(final, 128, 255, cv2.THRESH_BINARY)

    # 11. Save result
    paper_texture = cv2.imread("paper_texture_scripts.jpg", cv2.IMREAD_GRAYSCALE)
    final_with_texture = add_paper_texture(final, paper_texture, alpha=0.8)
    cv2.imwrite(output_path, final_with_texture)
    print(f"Saved handwritten-style output to {output_path}")

if __name__ == "__main__":
    input_image_path = "data/original/10003.png"     # Replace with your input
    output_image_path = "handwritten_10003.png"   # Replace with your desired output
    handwritten_effect_pipeline(input_image_path, output_image_path)