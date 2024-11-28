import os
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import random
import numpy as np
from PIL import Image, ImageFilter

input_folder = "../../photos/4-256x256"
output_folder = "../../photos/5-augmented"
os.makedirs(output_folder, exist_ok=True)

def random_noise(image):
    np_image = np.array(image)
    noise = np.random.normal(loc=0.0, scale=15.0, size=np_image.shape).astype(np.int32)
    noisy_image = np.clip(np_image + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)

def random_blur(image):
    """
    Apply random Gaussian blur to the image.
    """
    if random.random() < 0.5:
        radius = random.uniform(0.5, 1.5)
        return image.filter(ImageFilter.GaussianBlur(radius))
    return image



augmentation_transforms = transforms.Compose([
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.1
    ),
    transforms.Lambda(random_noise),
    transforms.Lambda(random_blur),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),

])

image_files = []
for f in os.listdir(input_folder):
    if f.endswith('.png'):
        image_files.append(f)

for filename in tqdm(image_files, desc="Procesowanie zdjęć"):
    image_path = os.path.join(input_folder, filename)
    image = Image.open(image_path).convert("RGB")
    for i in range(100):
        augmented_image = augmentation_transforms(image)
        output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_aug_{i}.jpg")
        augmented_image.save(output_path, format="JPEG", quality=100)
