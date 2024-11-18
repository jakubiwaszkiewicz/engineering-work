import os
import numpy as np
from skimage import io, feature
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define folder paths
input_folder = '../../256x256'
preoutput_folder = '../../red-green-and-blue_256'
output_folder = '../../hog-images_rgb_256'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

os.makedirs(preoutput_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# Get list of image files in the input folder
image_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]

# Process each image in the input folder with a progress bar
for filename in tqdm(image_files, desc="Processing images"):
    # Load the image
    image_path = os.path.join(input_folder, filename)
    image = io.imread(image_path)
    
    # Check if the image has an alpha channel (4 channels) and remove it if present
    if image.shape[-1] == 4:
        image = image[:, :, :3]  # Discard alpha channel

    # Define color-specific colormaps for each channel
    colormaps = ['Reds', 'Greens', 'Blues']

    # Process each color channel (R, G, B) with a progress bar
    for i, color_name in enumerate(['r', 'g', 'b']):
        # Extract the color channel
        channel_image = image[:, :, i]

        # Save the color channel using color-specific colormap
        preoutput_image_path = os.path.join(preoutput_folder, f"{os.path.splitext(filename)[0]}-{color_name}.png")
        plt.imsave(preoutput_image_path, channel_image, cmap=colormaps[i])

        # Compute HOG features and HOG visualization for the color channel
        _, hog_image = feature.hog(
            channel_image,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            visualize=True
        )
        
        # Save the HOG visualization for the color channel in grayscale
        output_image_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}-{color_name}.png")
        plt.imsave(output_image_path, hog_image, cmap='gray')

print("All images have been processed, and HOG visualizations for each color channel have been saved.")
