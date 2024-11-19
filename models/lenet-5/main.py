import os
import json
import numpy as np
import torch
import torch.nn as nn

import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from PIL import Image
import datetime
import random
from tqdm import tqdm


# Dane konfiguracyjne
NUM_CLASSES_HARDNESS = 3  # Hardness
NUM_CLASSES_HONEY = 2     # Honey presence
NUM_CLASSES_CAPPED = 2    # Seal presence
IMAGES_DIR = "../../photos/5-augmented"
LABELS_FILE = "../../labels.json"

NUM_EPOCHS = 100
BATCH_SIZE = 32
# LIMIT_SAMPLES = 43400
# Hiperparametry modelu
LEARNING_RATE = 0.001

RESULTS_DIR = "results"
RUN_FOLDER = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
SAVE_DIR = os.path.join(RESULTS_DIR, RUN_FOLDER)

os.makedirs(SAVE_DIR, exist_ok=True)

LOG_FILE = os.path.join(SAVE_DIR, "log.json")
MODEL_FILE = os.path.join(SAVE_DIR, "model.pth")

print(f"Wyniki będą zapisane w folderze: {SAVE_DIR}")

#problem
# DEVICE = torch.device("cpu")
# na gpu jest wolniej niz na cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Używana jednostka obliczeniowa: {DEVICE}")

#wyłącza sie randomowo

# Załaduj etykiety
with open(LABELS_FILE, "r") as labels_file:
    labels_data = json.load(labels_file)

# Create a dictionary mapping image IDs to labels
original_labels_dict = {}
for item in labels_data:
    original_labels_dict[item["id"]] = (
        int(item["hardness"]),
        int(item["have_honey"]),
        int(item["have_seal"]),
    )

# Pobierz wszystkie pliki z folderu
extensions = tuple(extension.lower() for extension in [".png", ".jpg"])
name_files = [file for file in os.listdir(IMAGES_DIR) if file.lower().endswith(extensions)]      
# name_files = random.sample(name_files, min(len(name_files), LIMIT_SAMPLES))
# print(f"Wybrano {len(name_files)} zdjęć do przetwarzania.")

augmented_labels_dict = {}

for file in name_files:
    image_id = int(file.split("_")[0])
    hardness, have_honey, have_seal = original_labels_dict.get(image_id)
    augmented_labels_dict[file] = (hardness, have_honey, have_seal)

def create_augmented_dataset(image_folder, labels_dict, transform):
    dataset = []
    image_files = list(labels_dict.keys())

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = Image.open(image_path).convert("RGB")
        
        if transform:
            image = transform(image)

        labels = labels_dict[image_file]
        dataset.append((image, torch.tensor(labels, dtype=torch.long)))

    return dataset

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

])

dataset_with_labels = create_augmented_dataset(
    image_folder=IMAGES_DIR,
    labels_dict=augmented_labels_dict,
    transform=transform
)

train_size = int(0.8 * len(dataset_with_labels))
test_size = len(dataset_with_labels) - train_size

train_dataset, test_dataset = random_split(dataset_with_labels, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Zbiór zdjęć załadowany: {len(dataset_with_labels)} obrazów.")
print(f"Obrazy treningowe: {len(train_dataset)}, Obrazy testowe: {len(test_dataset)}")
print("Etap danych zakończony...")
# Define LeNet-5 model with multiple labels
# Define LeNet-5 model with multiple labels
class LeNet5MultiLabel(nn.Module):
    def __init__(self):
        super(LeNet5MultiLabel, self).__init__()
        
        # Feature extraction using Sequential
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Compute the flattened size dynamically
        self._flattened_size = self._get_flattened_size()

        # Fully connected layers using Sequential
        self.classifier = nn.Sequential(
            nn.Linear(self._flattened_size, 1024),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.5),

            nn.Linear(1024, 1024),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.5),

            nn.Linear(1024, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.5),

            nn.Linear(512, 5)  # Final output layer
        )

    def _get_flattened_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 256, 256)  # Example input size
            features = self.feature_extractor(dummy_input)
            return features.numel()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)

        # Outputs for multitask classification
        hardness_output = torch.softmax(x[:, :3], dim=1)  # First 3 outputs
        honey_output = torch.sigmoid(x[:, 3])             # Fourth output
        capped_output = torch.sigmoid(x[:, 4])            # Fifth output

        return hardness_output, honey_output, capped_output


model = LeNet5MultiLabel().to(DEVICE)

criterion_hardness = nn.CrossEntropyLoss()

# odmiana entropi krzyzowej dla problemu binarnego
criterion_honey = nn.BCEWithLogitsLoss()
criterion_capped = nn.BCEWithLogitsLoss()

optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)


def train_model():
    print("Rozpoczynam trenowanie modelu...")
    model.train()
    total_loss = 0
    for images, labels in tqdm(train_loader, desc="Training"):
        images = images.to(DEVICE)
        labels_hardness = labels[:, 0].to(DEVICE)
        labels_honey = labels[:, 1].float().to(DEVICE)
        labels_capped = labels[:, 2].float().to(DEVICE)

        hardness_output, honey_output, capped_output = model(images)

        loss_hardness = criterion_hardness(hardness_output, labels_hardness)
        loss_honey = criterion_honey(honey_output, labels_honey)
        loss_capped = criterion_capped(capped_output, labels_capped)

        loss = loss_hardness + loss_honey + loss_capped
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step(total_loss)
    return total_loss / len(train_loader)

# Function to test the model
def test_model():
    print("Rozpoczynam testowanie modelu...")
    model.eval()
    all_labels_hardness, all_preds_hardness = [], []
    all_labels_honey, all_preds_honey = [], []
    all_labels_capped, all_preds_capped = [], []
    

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels_hardness = labels[:, 0].to(DEVICE)
            labels_honey = labels[:, 1].float().to(DEVICE)
            labels_capped = labels[:, 2].float().to(DEVICE)

            hardness_output, honey_output, capped_output = model(images)

            preds_hardness = hardness_output.argmax(dim=1)
            preds_honey = (honey_output > 0.5).float()
            preds_capped = (capped_output > 0.5).float()

            all_labels_hardness.extend(labels_hardness.cpu().numpy())
            all_preds_hardness.extend(preds_hardness.cpu().numpy())
            all_labels_honey.extend(labels_honey.cpu().numpy())
            all_preds_honey.extend(preds_honey.cpu().numpy())
            all_labels_capped.extend(labels_capped.cpu().numpy())
            all_preds_capped.extend(preds_capped.cpu().numpy())
            
    acc_hardness = balanced_accuracy_score(all_labels_hardness, all_preds_hardness)
    acc_honey = balanced_accuracy_score(all_labels_honey, all_preds_honey)
    acc_capped = balanced_accuracy_score(all_labels_capped, all_preds_capped)
    return acc_hardness, acc_honey, acc_capped, all_labels_hardness, all_preds_hardness

# Function to log metrics
def log_metrics(epoch, train_loss, test_accuracies, labels, preds):
    log_data = {
        "epoch": epoch + 1,
        "total_train_loss": train_loss,
        "test_accuracies": {
            "hardness": test_accuracies[0],
            "honey": test_accuracies[1],
            "capped": test_accuracies[2],
        },
        "confusion_matrix_hardness": confusion_matrix(labels[0], preds[0]).tolist(),
    }
    with open(LOG_FILE, "a") as f:
        json.dump(log_data, f)
        f.write("\n,")

# Training and testing the model

with open(LOG_FILE, "a") as f:
        f.write("[")

# Training and testing the model
for epoch in range(NUM_EPOCHS):
    train_loss = train_model()
    acc_hardness, acc_honey, acc_capped, labels_hardness, preds_hardness = test_model()
    test_accuracies = (acc_hardness, acc_honey, acc_capped)
    log_metrics(epoch, train_loss, test_accuracies, [labels_hardness], [preds_hardness])
    
    # Zapis modelu po każdej epoce
    model_path = os.path.join(SAVE_DIR, f"model_epoch_{epoch + 1}.pth")
    torch.save(model.state_dict(), model_path)
    
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {train_loss:.4f}, Accuracies: Hardness={acc_hardness:.4f}, Honey={acc_honey:.4f}, Capped={acc_capped:.4f}")

# Zapis ostatecznego modelu
torch.save(model.state_dict(), MODEL_FILE)
print(f"Model zapisany do {MODEL_FILE}")

with open(LOG_FILE, "a") as f:
        f.write("]")