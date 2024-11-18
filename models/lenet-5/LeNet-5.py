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

# Dane konfiguracyjne
NUM_CLASSES_HARDNESS = 3  # Hardness
NUM_CLASSES_HONEY = 2     # Honey presence
NUM_CLASSES_CAPPED = 2    # Seal presence
IMAGES_DIR = "../../photos/5-augmented"
LABELS_FILE = "../../labels.json"

LIMIT_SAMPLES = 2000  

NUM_EPOCHS = 10
BATCH_SIZE = 32

# Hiperparametry modelu
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Stworz logi treningowe
LOG_FILE = "log" + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + ".json"

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

augmented_labels_dict = {}

for file in name_files:
    image_id = int(file.split("_")[0])
    hardness, have_honey, have_seal = original_labels_dict.get(image_id)
    augmented_labels_dict[file] = (hardness, have_honey, have_seal)

def create_augmented_dataset(image_folder, labels_dict, transform=None):
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
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Dataset loaded: {len(dataset_with_labels)} samples.")
print(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")


exit(1)


# Define LeNet-5 model with multiple labels
class LeNet5MultiLabel(nn.Module):
    def __init__(self):
        super(LeNet5MultiLabel, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(in_features=16 * 61 * 61, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc_out = nn.Linear(in_features=84, out_features=5)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 16 * 61 * 61)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # pierwsze 3 neurony naleza do twardosc
        hardness_output = torch.softmax(self.fc_out(x)[:, :3], dim=1)
        # 4 (3 liczac od 0) neuron nalezacy do miodu
        honey_output = torch.sigmoid(self.fc_out(x)[:, 3])
        # 5 (4 liczac od 0) neuron nalezacy do zasklepin
        capped_output = torch.sigmoid(self.fc_out(x)[:, 4])
        
        return hardness_output, honey_output, capped_output

model = LeNet5MultiLabel().to(DEVICE)

# Loss functions
criterion_hardness = nn.CrossEntropyLoss()
criterion_honey = nn.BCELoss()
criterion_capped = nn.BCELoss()

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

# Function to train the model
def train_model():
    model.train()
    total_loss = 0
    for images, labels in train_loader:
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

    return total_loss / len(train_loader)

# Function to test the model
def test_model():
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
        "train_loss": train_loss,
        "test_accuracies": {
            "hardness": test_accuracies[0],
            "honey": test_accuracies[1],
            "capped": test_accuracies[2]
        },
        "confusion_matrix_hardness": confusion_matrix(labels[0], preds[0]).tolist(),
    }
    with open(LOG_FILE, "a") as f:
        json.dump(log_data, f)
        f.write("\n")

# Training and testing the model
for epoch in range(NUM_EPOCHS):
    train_loss = train_model()
    acc_hardness, acc_honey, acc_capped, labels_hardness, preds_hardness = test_model()
    test_accuracies = (acc_hardness, acc_honey, acc_capped)
    log_metrics(epoch, train_loss, test_accuracies, [labels_hardness], [preds_hardness])
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {train_loss:.4f}, Accuracies: Hardness={acc_hardness:.4f}, Honey={acc_honey:.4f}, Capped={acc_capped:.4f}")
