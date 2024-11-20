import os
import json
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from datetime import datetime

IMAGES_DIR = "../../photos/6-hog-rgb"
LABELS_FILE = "../../labels.json"

DATA_LIMIT=3000

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


name_files = name_files[:DATA_LIMIT]

augmented_labels_dict = np.zeros((len(name_files), 3))

for idx, (file) in enumerate(name_files):
    image_id = int(file.split("_")[2])
    hardness, have_honey, have_seal = original_labels_dict.get(image_id)
    augmented_labels_dict[idx, :] = np.array([hardness, have_honey, have_seal])

def load_dataset(image_folder, labels_array, transform):
    num_images = len(labels_array)
    images = np.zeros((num_images, 768 * 256))
    labels = np.zeros((num_images, 3))
    for idx in tqdm(range(num_images), desc="Loading images"):
        image_file = name_files[idx]
        image_path = os.path.join(image_folder, image_file)
        label = labels_array[idx]
        image = Image.open(image_path).convert("L")
        image = transform(image)
        images[idx, :] = np.array(image).flatten()
        labels[idx, :] = label
    return np.array(images), np.array(labels)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

X, y = load_dataset(IMAGES_DIR, augmented_labels_dict, transform)

y_hardness = y[:, 0]
y_honey = y[:, 1]
y_capped = y[:, 2]

X_train, X_test, y_hardness_train, y_hardness_test = train_test_split(X, y_hardness, test_size=0.2)
X_train, X_test, y_honey_train, y_honey_test = train_test_split(X, y_honey, test_size=0.2)
X_train, X_test, y_capped_train, y_capped_test = train_test_split(X, y_capped, test_size=0.2)

svm_hardness = SVC(kernel='rbf', C=1, gamma='scale', probability=True)
svm_honey = SVC(kernel='rbf', C=1, gamma='scale', probability=True)
svm_capped = SVC(kernel='rbf', C=1, gamma='scale', probability=True)

print("Trenowanie modelu...")
svm_hardness.fit(X_train, y_hardness_train)
svm_honey.fit(X_train, y_honey_train)
svm_capped.fit(X_train, y_capped_train)

print("Testowanie modelu...")
y_hardness_pred = svm_hardness.predict(X_test)
y_honey_pred = svm_honey.predict(X_test)
y_capped_pred = svm_capped.predict(X_test)

acc_hardness = balanced_accuracy_score(y_hardness_test, y_hardness_pred)
acc_honey = balanced_accuracy_score(y_honey_test, y_honey_pred)
acc_capped = balanced_accuracy_score(y_capped_test, y_capped_pred)

cm_hardness = confusion_matrix(y_hardness_test, y_hardness_pred).tolist()

report_hardness = classification_report(y_hardness_test, y_hardness_pred, output_dict=True)
report_honey = classification_report(y_honey_test, y_honey_pred, output_dict=True)
report_capped = classification_report(y_capped_test, y_capped_pred, output_dict=True)

log_data = {
    "balanced_accuracy": {
        "hardness": acc_hardness,
        "honey": acc_honey,
        "capped": acc_capped
    },
    "confusion_matrices": {
        "hardness": cm_hardness,
    },
    "classification_reports": {
        "hardness": report_hardness,
        "honey": report_honey,
        "capped": report_capped
    }
}

with open(LOG_FILE, "w") as log_file:
    json.dump(log_data, log_file, indent=4)

print(f"Wyniki zapisano w pliku {LOG_FILE}")
