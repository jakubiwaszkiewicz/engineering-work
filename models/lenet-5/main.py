import os
import json
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from imblearn.over_sampling import RandomOverSampler

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import cv2
import logging

NUM_CLASSES_HARDNESS = 2
NUM_CLASSES_HONEY = 2
NUM_CLASSES_CAPPED = 2

IMAGES_DIR = "../../photos/5-augmented"
LABELS_FILE = "../../labels-binary.json"
RESULTS_DIR = "results"
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001

RUN_FOLDER = datetime.now().strftime("%Y-%m-%d_%H-%M")
SAVE_DIR = os.path.join(RESULTS_DIR, RUN_FOLDER)
os.makedirs(SAVE_DIR, exist_ok=True)
log_file = os.path.join(SAVE_DIR, "log.txt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

LOSS_THRESHOLD = 0.1
PATIENCE = 3

with open(LABELS_FILE, "r") as labels_file:
    labels_data = json.load(labels_file)

id_to_images = {}
id_to_labels = {}
for item in labels_data:
    image_id = int(item["id"])
    label = (int(item["hardness"]), int(item["have_honey"]), int(item["have_seal"]))
    id_to_labels[image_id] = label
for file in os.listdir(IMAGES_DIR):
    if file.lower().endswith((".png", ".jpg")):
        image_id = int(file.split("_")[0])
        id_to_images.setdefault(image_id, []).append(file)

image_ids = list(id_to_images.keys())

labels = [[id_to_labels[image_id][0],id_to_labels[image_id][1],id_to_labels[image_id][2]] for image_id in image_ids]
train_labels_hardness = [label[0] for label in labels]
train_labels_honey = [label[1] for label in labels]
train_labels_capped = [label[2] for label in labels]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class LeNet5MultiLabel(nn.Module):
    def __init__(self):
        super(LeNet5MultiLabel, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.LeakyReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.LeakyReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.LeakyReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.LeakyReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.LeakyReLU(), nn.AdaptiveAvgPool2d(1)
        )

        self.flattened_size = self._compute_flattened_size()

        self.shared_fc = nn.Sequential(
            nn.Linear(self.flattened_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.hardness_fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.hardness_classifier = nn.Linear(512, NUM_CLASSES_HARDNESS)

        self.honey_fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.honey_classifier = nn.Linear(512, NUM_CLASSES_HONEY)

        self.capped_fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.capped_classifier = nn.Linear(512, NUM_CLASSES_CAPPED)

    def _compute_flattened_size(self):
        with torch.no_grad():
            return self.extractor(torch.zeros(1, 3, 256, 256)).numel()

    def forward(self, x):
        x = self.extractor(x).view(x.size(0), -1)
        
        shared_features = self.shared_fc(x)


        hardness_features = self.hardness_fc(shared_features)
        honey_features = self.honey_fc(shared_features)
        capped_features = self.capped_fc(shared_features)

        hardness_output = self.hardness_classifier(hardness_features)
        honey_output = self.honey_classifier(honey_features)
        capped_output = self.capped_classifier(capped_features)

        return hardness_output, honey_output, capped_output

model = LeNet5MultiLabel().to(DEVICE)
from sklearn.utils.class_weight import compute_class_weight

import numpy as np

# Convert `range` to `numpy.ndarray`
class_weights_hardness = compute_class_weight(
    class_weight='balanced', 
    classes=np.array(range(NUM_CLASSES_HARDNESS)), 
    y=train_labels_hardness
)
class_weights_honey = compute_class_weight(
    class_weight='balanced', 
    classes=np.array(range(NUM_CLASSES_HONEY)), 
    y=train_labels_honey
)
class_weights_capped = compute_class_weight(
    class_weight='balanced', 
    classes=np.array(range(NUM_CLASSES_CAPPED)), 
    y=train_labels_capped
)

criterion_hardness = nn.CrossEntropyLoss(weight=torch.tensor(class_weights_hardness, dtype=torch.float).to(DEVICE))
criterion_honey = nn.CrossEntropyLoss(weight=torch.tensor(class_weights_honey, dtype=torch.float).to(DEVICE))
criterion_capped = nn.CrossEntropyLoss(weight=torch.tensor(class_weights_capped, dtype=torch.float).to(DEVICE))


optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

def learn_in_batches(files, labels):
    model.train()
    total_loss = 0
    all_preds_hardness, all_preds_honey, all_preds_capped = [], [], []
    all_labels_hardness, all_labels_honey, all_labels_capped = [], [], []
    total_loss = 0
    for i in tqdm(range(0, len(files), BATCH_SIZE), desc="Training"):
        batch_files = files[i:i + BATCH_SIZE]
        batch_labels = labels[i:i + BATCH_SIZE]
        images = []
        batch_labels_tensors = []
        images_paths = []
        for file, label in zip(batch_files, batch_labels):
            image_path = os.path.join(IMAGES_DIR, file)
            images_paths.append(image_path)
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            transformed_image = transform(Image.fromarray(image))
            images.append(transformed_image)
            batch_labels_tensors.append(torch.tensor(label, dtype=torch.long))
        images = torch.stack(images).to(DEVICE)
        labels_torch = torch.stack(batch_labels_tensors).to(DEVICE)
        optimizer.zero_grad()
        hardness, honey, capped = model(images)
        loss = (
            criterion_hardness(hardness, labels_torch[:, 0]) +
            criterion_honey(honey, labels_torch[:, 1]) +
            criterion_capped(capped, labels_torch[:, 2])
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        all_preds_hardness.extend(hardness.argmax(1).cpu().numpy())
        all_preds_honey.extend(honey.argmax(1).cpu().numpy())
        all_preds_capped.extend(capped.argmax(1).cpu().numpy())
        all_labels_hardness.extend(labels_torch[:, 0].cpu().numpy())
        all_labels_honey.extend(labels_torch[:, 1].cpu().numpy())
        all_labels_capped.extend(labels_torch[:, 2].cpu().numpy())

    balanced_acc_hardness = balanced_accuracy_score(all_labels_hardness, all_preds_hardness)
    balanced_acc_honey = balanced_accuracy_score(all_labels_honey, all_preds_honey)
    balanced_acc_capped = balanced_accuracy_score(all_labels_capped, all_preds_capped)
    
    acc_hardness = accuracy_score(all_labels_hardness, all_preds_hardness)
    acc_honey = accuracy_score(all_labels_honey, all_preds_honey)
    acc_capped = accuracy_score(all_labels_capped, all_preds_capped)

    cm_hardness = confusion_matrix(all_labels_hardness, all_preds_hardness)
    cm_honey = confusion_matrix(all_labels_honey, all_preds_honey)
    cm_capped = confusion_matrix(all_labels_capped, all_preds_capped)

    return total_loss / max(1, len(files) // BATCH_SIZE), acc_hardness, acc_honey, acc_capped, balanced_acc_hardness, balanced_acc_honey, balanced_acc_capped, cm_hardness, cm_honey, cm_capped



logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logging.info("Starting training process...")

def test_in_batches(files, labels):
    model.eval()
    all_preds_hardness, all_preds_honey, all_preds_capped = [], [], []
    all_labels_hardness, all_labels_honey, all_labels_capped = [], [], []
    total_loss = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(files), BATCH_SIZE), desc="Evaluating"):
            batch_files = files[i:i + BATCH_SIZE]
            batch_labels = labels[i:i + BATCH_SIZE]
            images = []
            batch_labels_tensors = []

            for file, label in zip(batch_files, batch_labels):
                image_path = os.path.join(IMAGES_DIR, file)
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                transformed_image = transform(Image.fromarray(image))
                images.append(transformed_image)
                batch_labels_tensors.append(torch.tensor(label, dtype=torch.long))
            images = torch.stack(images).to(DEVICE)
            labels_torch = torch.stack(batch_labels_tensors).to(DEVICE)
            hardness, honey, capped = model(images)
            loss = (
                criterion_hardness(hardness, labels_torch[:, 0]) +
                criterion_honey(honey, labels_torch[:, 1]) +
                criterion_capped(capped, labels_torch[:, 2])
            )
            total_loss += loss.item()

            all_preds_hardness.extend(hardness.argmax(1).cpu().numpy())
            all_preds_honey.extend(honey.argmax(1).cpu().numpy())
            all_preds_capped.extend(capped.argmax(1).cpu().numpy())

            all_labels_hardness.extend(labels_torch[:, 0].cpu().numpy())
            all_labels_honey.extend(labels_torch[:, 1].cpu().numpy())
            all_labels_capped.extend(labels_torch[:, 2].cpu().numpy())

    balanced_acc_hardness = balanced_accuracy_score(all_labels_hardness, all_preds_hardness)
    balanced_acc_honey = balanced_accuracy_score(all_labels_honey, all_preds_honey)
    balanced_acc_capped = balanced_accuracy_score(all_labels_capped, all_preds_capped)
    
    acc_hardness = accuracy_score(all_labels_hardness, all_preds_hardness)
    acc_honey = accuracy_score(all_labels_honey, all_preds_honey)
    acc_capped = accuracy_score(all_labels_capped, all_preds_capped)

    cm_hardness = confusion_matrix(all_labels_hardness, all_preds_hardness)
    cm_honey = confusion_matrix(all_labels_honey, all_preds_honey)
    cm_capped = confusion_matrix(all_labels_capped, all_preds_capped)

    return total_loss / max(1, len(files) // BATCH_SIZE), balanced_acc_hardness, balanced_acc_honey, balanced_acc_capped, acc_hardness, acc_honey, acc_capped, cm_hardness, cm_honey, cm_capped

mskf = MultilabelStratifiedKFold(n_splits=10, shuffle=True)

import matplotlib.pyplot as plt
from collections import Counter

def calculate_class_distribution_and_plot(labels, num_classes, name, oversampled=False):
    counts = Counter(labels)
    total = sum(counts.values())
    percentages = {cls: round((counts[cls] / total) * 100, 1) for cls in range(num_classes)}
    # Define consistent colors for the bars
    colors = ['red', 'blue', 'green', 'purple', 'orange'][:num_classes]
    plt.figure(figsize=(6, 4))  # Ensure a fresh figure is created
    plt.bar(percentages.keys(), percentages.values(), color=colors, edgecolor="black")
    plt.xlabel("Class")
    plt.ylabel("Percentage (%)")
    plt.title(f"{name} Class Distribution{' (Oversampled)' if oversampled else ''}")
    plt.xticks(range(num_classes))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # Save the figure
    filename = f"{name}_class_distribution{'(Oversampled)' if oversampled else ''}.png"
    plt.savefig(filename, dpi=300)
    return percentages

logging.info("Starting training process...")


def plot_confusion_matrix(cm, classes, title, fold, epoch, phase, save_dir, normalize=False):
    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        cm_normalized = cm

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f" if normalize else "d", 
                cmap="coolwarm", xticklabels=classes, yticklabels=classes,
                cbar=True, annot_kws={"size": 12})
    
    plt.title(f"{title} - Fold {fold} - Epoch {epoch} ({phase})", fontsize=14)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)

    # Highlight the diagonal for correct predictions
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i == j:
                plt.gca().add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=2))

    plt.tight_layout()
    file_path = os.path.join(save_dir, f"{title}_fold_{fold}_epoch_{epoch}_{phase}.png")
    plt.savefig(file_path)
    plt.close()
    print(f"Confusion matrix saved to {file_path}")


for fold, (train_idx, test_idx) in enumerate(mskf.split(image_ids, labels)):
    print(f"--- Fold {fold + 1} ---")
    logging.info(f"--- Fold {fold + 1} ---")
    train_ids = [image_ids[i] for i in train_idx]
    test_ids = [image_ids[i] for i in test_idx]

    train_files = [f for tid in train_ids for f in id_to_images[tid]]
    test_files = [f for tid in test_ids for f in id_to_images[tid]]
    train_labels = [id_to_labels[tid] for tid in train_ids for _ in id_to_images[tid]]
    test_labels = [id_to_labels[tid] for tid in test_ids for _ in id_to_images[tid]]

    train_labels_combined = [
    f"{label[0]}_{label[1]}_{label[2]}" for label in train_labels
    ]
    
    train_files = np.array(train_files).reshape(-1, 1)

    # Oversample based on combined labels
    ros = RandomOverSampler()
    train_files_resampled, train_labels_combined_resampled = ros.fit_resample(
        train_files, train_labels_combined
    )

    # Split the combined labels back into individual labels
    train_labels_resampled = [
        tuple(map(int, label.split("_"))) for label in train_labels_combined_resampled
    ]

    # Flatten and convert back to lists
    train_files = train_files_resampled.flatten().tolist()
    train_labels = train_labels_resampled
    train_labels_hardness = [label[0] for label in train_labels]
    train_labels_honey = [label[1] for label in train_labels]
    train_labels_capped = [label[2] for label in train_labels]

    print("--- Oversampled class distribution in the training set ---")
    print(f"Hardness: {calculate_class_distribution_and_plot(train_labels_hardness, NUM_CLASSES_HARDNESS, "Hardness", True)}")
    print(f"Honey: {calculate_class_distribution_and_plot(train_labels_honey, NUM_CLASSES_HONEY, "Honey", True)}")
    print(f"Capped: {calculate_class_distribution_and_plot(train_labels_capped, NUM_CLASSES_CAPPED, "Capped", True)}")

    test_labels_hardness = [label[0] for label in test_labels]
    test_labels_honey = [label[1] for label in test_labels]
    test_labels_capped = [label[2] for label in test_labels]
    
    print("--- Rozklad klas w zbiorze testowym: ---")
    print(f"Hardness: {calculate_class_distribution_and_plot(test_labels_hardness, NUM_CLASSES_HARDNESS, "Hardness", False)}")
    print(f"Honey: {calculate_class_distribution_and_plot(test_labels_honey, NUM_CLASSES_HONEY, "Honey", False)}")
    print(f"Capped: {calculate_class_distribution_and_plot(test_labels_capped, NUM_CLASSES_CAPPED, "Capped", False)}")

    best_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        logging.info(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        train_loss, acc_hardness, acc_honey, acc_capped, balanced_acc_hardness, balanced_acc_honey, balanced_acc_capped, cm_hardness, cm_honey, cm_capped = learn_in_batches(train_files, train_labels)
        print(f"Train Loss: {train_loss:.3f}, Hardness Accuracy: {acc_hardness:.3f}, Honey Accuracy: {acc_honey:.3f}, Capped Accuracy: {acc_capped:.3f}")
        print(f"Hardness Balaced Accuracy: {balanced_acc_hardness:.3f}, Honey Balaced Accuracy: {balanced_acc_honey:.3f}, Capped Balaced Accuracy: {balanced_acc_capped:.3f}")
        logging.info(f"Train Loss: {train_loss:.3f}, Hardness Accuracy: {acc_hardness:.3f}, Honey Accuracy: {acc_honey:.3f}, Capped Accuracy: {acc_capped:.3f}")
        logging.info(f"Hardness Balaced Accuracy: {balanced_acc_hardness:.3f}, Honey Balaced Accuracy: {balanced_acc_honey:.3f}, Capped Balaced Accuracy: {balanced_acc_capped:.3f}")
        
        plot_confusion_matrix(cm_hardness, classes=range(NUM_CLASSES_HARDNESS), 
                            title="Hardness", fold=fold + 1, 
                            epoch=epoch + 1, phase="Train", save_dir=SAVE_DIR)
        plot_confusion_matrix(cm_honey, classes=range(NUM_CLASSES_HONEY),
                            title="Honey", fold=fold + 1, 
                            epoch=epoch + 1, phase="Train", save_dir=SAVE_DIR)
        plot_confusion_matrix(cm_capped, classes=range(NUM_CLASSES_CAPPED),
                            title="Capped", fold=fold + 1, 
                            epoch=epoch + 1, phase="Train", save_dir=SAVE_DIR)
        
        model_save_path = os.path.join(SAVE_DIR, f"model_fold_{fold + 1}_{epoch+1}.pth")
        
        torch.save(model.state_dict(), model_save_path)

        scheduler.step()

        if train_loss < best_loss:
            best_loss = train_loss

        if best_loss <= LOSS_THRESHOLD:
            print("Early stopping: Loss below threshold.")
            logging.info("Early stopping: Loss below threshold.")
            break

        test_loss, balanced_acc_hardness, balanced_acc_honey, balanced_acc_capped, acc_hardness, acc_honey, acc_capped, cm_hardness, cm_honey, cm_capped = test_in_batches(test_files, test_labels)
        print(f"Test Loss: {test_loss:.3f}")
        print(f" Hardness Accuracy: {acc_hardness:.3f}, Honey Accuracy: {acc_honey:.3f}, Capped Accuracy: {acc_capped:.3f}")
        print(f"Hardness Balaced Accuracy: {balanced_acc_hardness:.3f}, Honey Balaced Accuracy: {balanced_acc_honey:.3f}, Capped Balaced Accuracy: {balanced_acc_capped:.3f}")
        logging.info(f"Test Loss: {test_loss:.3f}")
        logging.info(f"Hardness Accuracy: {acc_hardness:.3f}, Honey Accuracy: {acc_honey:.3f}, Capped Accuracy: {acc_capped:.3f}")
        logging.info(f"Hardness Balaced Accuracy: {balanced_acc_hardness:.3f}, Honey Balaced Accuracy: {balanced_acc_honey:.3f}, Capped Balaced Accuracy: {balanced_acc_capped:.3f}")

        plot_confusion_matrix(cm_hardness, classes=range(NUM_CLASSES_HARDNESS), 
                            title="Hardness", fold=fold + 1, 
                            epoch=epoch + 1, phase="Test", save_dir=SAVE_DIR)

        plot_confusion_matrix(cm_honey, classes=range(NUM_CLASSES_HONEY), 
                            title="Honey", fold=fold + 1, 
                            epoch=epoch + 1, phase="Test", save_dir=SAVE_DIR)

        plot_confusion_matrix(cm_capped, classes=range(NUM_CLASSES_CAPPED), 
                            title="Capped", fold=fold + 1, 
                            epoch=epoch + 1, phase="Test", save_dir=SAVE_DIR)

    print(f"--- Fold {fold + 1} Completed ---")
    logging.info(f"--- Fold {fold + 1} Completed ---")
    


logging.info("Training process completed.")
