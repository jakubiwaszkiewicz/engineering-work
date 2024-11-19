import numpy as np
import os
from PIL import Image
from tqdm import tqdm

input_folder ='../../photos/rembg'
output_folder = '../../photos/crop-and-black'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
def process_image(input_path, output_path):
    img = Image.open(input_path).convert("RGBA")
    img_data = np.array(img)
    non_transparent_mask = img_data[:, :, 3] > 0
    non_transparent_coords = np.argwhere(non_transparent_mask)

    if non_transparent_coords.size == 0:
        print(f"Pominięto pusty obraz: {input_path}")
        return
    top_left = non_transparent_coords.min(axis=0)
    bottom_right = non_transparent_coords.max(axis=0) + 1
    cropped_img_data = img_data[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
    cropped_img_data[..., :3][cropped_img_data[..., 3] == 0] = [0, 0, 0]
    cropped_img_data[..., 3] = 255
    cropped_img = Image.fromarray(cropped_img_data, "RGBA")
    cropped_img.save(output_path)

for filename in tqdm(os.listdir(input_folder), desc="Procesowanie zdjęć...", unit="zdjęcie"):
    if filename.lower().endswith('.png'):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        process_image(input_path, output_path)
