import cv2
import numpy as np
import os
from skimage.feature import hog
from tqdm import tqdm
from bisect import bisect_left

input_dir = '../../photos/5-augmented'
output_dir = '../../photos/6-hog-rgb'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

file_list = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

existing_files = sorted([f for f in os.listdir(output_dir) if f.startswith("HOG_RGB_")])

def is_processed(filename, existing_files):
    target = f"HOG_RGB_{filename}"
    index = bisect_left(existing_files, target)
    return index < len(existing_files) and existing_files[index] == target

for filename in tqdm(file_list, desc="Processing images"):
    if is_processed(filename, existing_files):
        continue

    image_path = os.path.join(input_dir, filename)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not read image {image_path}")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hog_channels = []

    for channel_idx, channel in enumerate(cv2.split(image_rgb)):
        features, hog_image = hog(
            channel,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=True,
            block_norm='L2-Hys'
        )
        hog_image_rescaled = (hog_image - hog_image.min()) / (hog_image.max() - hog_image.min()) * 255
        hog_image_rescaled = hog_image_rescaled.astype(np.uint8)
        hog_channels.append(hog_image_rescaled)
    
    merged_hog_image = np.vstack(hog_channels)
    hog_output_path = os.path.join(output_dir, f"HOG_RGB_{filename}")
    cv2.imwrite(hog_output_path, merged_hog_image)
