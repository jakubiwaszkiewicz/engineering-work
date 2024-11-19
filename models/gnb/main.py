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

# Dane konfiguracyjne
IMAGES_DIR = "../../photos/6-hog-rgb"
LABELS_FILE = "../../labels.json"

# Ścieżka do wyników
RESULTS_DIR = "results"
RUN_FOLDER = datetime.now().strftime("%Y-%m-%d_%H-%M")
SAVE_DIR = os.path.join(RESULTS_DIR, RUN_FOLDER)
os.makedirs(SAVE_DIR, exist_ok=True)

LOG_FILE = os.path.join(SAVE_DIR, "log.json")
print(f"Wyniki będą zapisane w folderze: {SAVE_DIR}")

# Załaduj etykiety
with open(LABELS_FILE, "r") as labels_file:
    labels_data = json.load(labels_file)

# Stwórz mapowanie ID obrazów do etykiet
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
    image_id = int(file.split("_")[2])
    hardness, have_honey, have_seal = original_labels_dict.get(image_id)
    augmented_labels_dict[file] = (hardness, have_honey, have_seal)

# Funkcja do ładowania danych obrazowych
def load_dataset(image_folder, labels_dict, transform):
    images = []
    labels = []
    for image_file, label in tqdm(labels_dict.items(), desc="Loading images"):
        image_path = os.path.join(image_folder, image_file)
        image = Image.open(image_path).convert("RGB")
        image = transform(image)
        images.append(np.array(image).flatten())  # Spłaszcz obraz do wektora
        labels.append(label)
    return np.array(images), np.array(labels)

# Przekształcenia obrazu
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Załaduj dane
X, y = load_dataset(IMAGES_DIR, augmented_labels_dict, transform)

# Podziel dane na klasy twardości, obecności miodu i pokrycia
y_hardness = y[:, 0]
y_honey = y[:, 1]
y_capped = y[:, 2]

# Podziel dane na zbiory treningowe i testowe
X_train, X_test, y_hardness_train, y_hardness_test = train_test_split(X, y_hardness, test_size=0.2, random_state=42)
X_train, X_test, y_honey_train, y_honey_test = train_test_split(X, y_honey, test_size=0.2, random_state=42)
X_train, X_test, y_capped_train, y_capped_test = train_test_split(X, y_capped, test_size=0.2, random_state=42)

# Model GNB
gnb_hardness = GaussianNB()
gnb_honey = GaussianNB()
gnb_capped = GaussianNB()

# Trenowanie modelu
print("Trenowanie modelu...")
gnb_hardness.fit(X_train, y_hardness_train)
gnb_honey.fit(X_train, y_honey_train)
gnb_capped.fit(X_train, y_capped_train)

# Testowanie modelu
print("Testowanie modelu...")
y_hardness_pred = gnb_hardness.predict(X_test)
y_honey_pred = gnb_honey.predict(X_test)
y_capped_pred = gnb_capped.predict(X_test)

# Wyniki
acc_hardness = balanced_accuracy_score(y_hardness_test, y_hardness_pred)
acc_honey = balanced_accuracy_score(y_honey_test, y_honey_pred)
acc_capped = balanced_accuracy_score(y_capped_test, y_capped_pred)

print(f"Balanced Accuracy (Hardness): {acc_hardness:.4f}")
print(f"Balanced Accuracy (Honey): {acc_honey:.4f}")
print(f"Balanced Accuracy (Capped): {acc_capped:.4f}")

# Macierze konfuzji
cm_hardness = confusion_matrix(y_hardness_test, y_hardness_pred).tolist()

# Raport klasyfikacji
report_hardness = classification_report(y_hardness_test, y_hardness_pred, output_dict=True)
report_honey = classification_report(y_honey_test, y_honey_pred, output_dict=True)
report_capped = classification_report(y_capped_test, y_capped_pred, output_dict=True)

# Zapis logów do pliku JSON
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
