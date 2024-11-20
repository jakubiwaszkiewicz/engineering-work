import os
import json
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from datetime import datetime
import joblib

IMAGES_DIR = "../../photos/6-hog-rgb"
LABELS_FILE = "../../labels.json"

DATA_LIMIT = 10000
EPOCHS = 5

RESULTS_DIR = "results"
RUN_FOLDER = datetime.now().strftime("%Y-%m-%d_%H-%M")
SAVE_DIR = os.path.join(RESULTS_DIR, RUN_FOLDER)
os.makedirs(SAVE_DIR, exist_ok=True)

LOG_FILE = os.path.join(SAVE_DIR, "log.json")
print(f"Wyniki będą zapisane w folderze: {SAVE_DIR}")

with open(LABELS_FILE, "r") as labels_file:
    labels_data = json.load(labels_file)

original_labels_dict = {}
for item in labels_data:
    original_labels_dict[item["id"]] = (
        int(item["hardness"]),
        int(item["have_honey"]),
        int(item["have_seal"]),
    )

extensions = tuple(extension.lower() for extension in [".png", ".jpg"])
name_files = [file for file in os.listdir(IMAGES_DIR) if file.lower().endswith(extensions)]

# Limit to the first 10,000 files
name_files = name_files[:DATA_LIMIT]

augmented_labels_dict = np.zeros((len(name_files), 3))
for idx, file in enumerate(name_files):
    image_id = int(file.split("_")[2])
    hardness, have_honey, have_seal = original_labels_dict.get(image_id)
    augmented_labels_dict[idx, :] = np.array([hardness, have_honey, have_seal])

def load_dataset(image_folder, labels_array, transform):
    num_images = len(labels_array)
    images = np.zeros((num_images, 768 * 256))
    labels = np.zeros((num_images, 3))
    for idx in tqdm(range(num_images), desc="Loading images"):
        image_file = name_files[idx]
        label = labels_array[idx]
        image_path = os.path.join(image_folder, image_file)
        image = Image.open(image_path).convert("L")
        image = transform(image)
        images[idx, :] = np.array(image).flatten()
        labels[idx, :] = label
    return images, labels

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

X, y = load_dataset(IMAGES_DIR, augmented_labels_dict, transform)

y_hardness = y[:, 0]
y_honey = y[:, 1]
y_capped = y[:, 2]

X_train, X_test, y_hardness_train, y_hardness_test = train_test_split(X, y_hardness, test_size=0.2, random_state=42)
X_train, X_test, y_honey_train, y_honey_test = train_test_split(X, y_honey, test_size=0.2, random_state=42)
X_train, X_test, y_capped_train, y_capped_test = train_test_split(X, y_capped, test_size=0.2, random_state=42)

gnb_hardness = GaussianNB()
gnb_honey = GaussianNB()
gnb_capped = GaussianNB()

log_data = {
    "epochs": []
}

print("Rozpoczęcie treningu...")

for epoch in range(1, EPOCHS + 1):
    print(f"Epoch {epoch}/{EPOCHS}")
    # Train
    gnb_hardness.fit(X_train, y_hardness_train)
    gnb_honey.fit(X_train, y_honey_train)
    gnb_capped.fit(X_train, y_capped_train)

    # Test
    y_hardness_pred = gnb_hardness.predict(X_test)
    y_honey_pred = gnb_honey.predict(X_test)
    y_capped_pred = gnb_capped.predict(X_test)

    acc_hardness = balanced_accuracy_score(y_hardness_test, y_hardness_pred)
    acc_honey = balanced_accuracy_score(y_honey_test, y_honey_pred)
    acc_capped = balanced_accuracy_score(y_capped_test, y_capped_pred)

    print(f"Balanced Accuracy (Hardness): {acc_hardness:.4f}")
    print(f"Balanced Accuracy (Honey): {acc_honey:.4f}")
    print(f"Balanced Accuracy (Capped): {acc_capped:.4f}")

    # Save
    epoch_data = {
        "epoch": epoch,
        "balanced_accuracy": {
            "hardness": acc_hardness,
            "honey": acc_honey,
            "capped": acc_capped
        }
    }

    log_data["epochs"].append(epoch_data)

    # Save models
    model_save_path_hardness = os.path.join(SAVE_DIR, f"gnb_hardness_epoch_{epoch}.joblib")
    model_save_path_honey = os.path.join(SAVE_DIR, f"gnb_honey_epoch_{epoch}.joblib")
    model_save_path_capped = os.path.join(SAVE_DIR, f"gnb_capped_epoch_{epoch}.joblib")

    joblib.dump(gnb_hardness, model_save_path_hardness)
    joblib.dump(gnb_honey, model_save_path_honey)
    joblib.dump(gnb_capped, model_save_path_capped)

    print(f"Saved models for epoch {epoch}.")

# Save log
with open(LOG_FILE, "w") as log_file:
    json.dump(log_data, log_file, indent=4)

print(f"Trening zakończony. Wyniki zapisano w pliku {LOG_FILE}")
