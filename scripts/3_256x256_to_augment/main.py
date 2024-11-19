import os
from tqdm import tqdm
from torchvision import transforms
from PIL import Image

input_folder = "../../photos/256x256"
output_folder = "../../photos/augmented"
os.makedirs(output_folder, exist_ok=True)

augmentation_transforms = transforms.Compose([
    transforms.RandomRotation(degrees=2),
    transforms.ColorJitter(brightness=0.2, contrast=0.5, saturation=0.1, hue=0.05),
])

image_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]

for filename in tqdm(image_files, desc="Procesowanie zdjęć"):
    image_path = os.path.join(input_folder, filename)
    image = Image.open(image_path).convert("RGB")
    for i in range(100):
        augmented_image = augmentation_transforms(image)
        output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_aug_{i}.jpg")
        augmented_image.save(output_path, format="JPEG", quality=100)
