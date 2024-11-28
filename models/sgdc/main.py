from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.base import clone
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score
from collections import Counter
import numpy as np
import json
import os
from datetime import datetime
import cv2
import torchvision.transforms as transforms
from tqdm import tqdm
import joblib
from collections import defaultdict
import random
import logging
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

IMAGES_DIR = "../../photos/6-hog-rgb"
LABELS_FILE = "../../labels.json"
RESULTS_DIR = "results"

RUN_FOLDER = datetime.now().strftime("%Y-%m-%d_%H-%M")
SAVE_DIR = os.path.join(RESULTS_DIR, RUN_FOLDER)
os.makedirs(SAVE_DIR, exist_ok=True)

N_SPLITS = 5
NUM_EPOCHS = 5
NUM_CLASSES_HARDNESS = 3
NUM_CLASSES_HONEY = 2
NUM_CLASSES_CAPPED = 2

LOG_FILE = os.path.join(SAVE_DIR, "log.txt")

logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

print(f"Wyniki będą zapisane w folderze: {SAVE_DIR}")

with open(LABELS_FILE, "r") as labels_file:
    labels_data = json.load(labels_file)

id_to_images = defaultdict(list)
id_to_labels = {}
for item in labels_data:
    image_id = int(item["id"])
    label = (int(item["hardness"]), int(item["have_honey"]), int(item["have_seal"]))
    id_to_labels[image_id] = label

extensions = (".png", ".jpg")
for file in os.listdir(IMAGES_DIR):
    if file.lower().endswith(extensions):
        try:
            image_id = int(file.split("_")[2])
            id_to_images[image_id].append(file)
        except (IndexError, ValueError, KeyError):
            print(f"Skipping file: {file}")

image_ids = list(id_to_images.keys())
labels = [[id_to_labels[image_id][0], id_to_labels[image_id][1], id_to_labels[image_id][2]] for image_id in image_ids]


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.ravel())
])

def learn_in_batches(model_to_learn, image_folder, name_files, labels, transform, batch_size=100):
    num_images = len(name_files)
    for start_idx in tqdm(range(0, num_images, batch_size)):
        end_idx = min(start_idx + batch_size, num_images)
        batch_images = []
        batch_labels = []
        for idx in range(start_idx, end_idx):
            image_file = name_files[idx]
            image_path = os.path.join(image_folder, image_file)
            label = labels[idx]
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                print(f"Error loading image: {image_path}")
                continue
            image = transform(image)
            batch_images.append(image)
            batch_labels.append(label)
        batch_images = np.array(batch_images)
        batch_labels = np.array(batch_labels)
        model_to_learn.partial_fit(batch_images, batch_labels, classes=np.unique(labels))

def evaluate_in_batches(model_to_evaluate, image_folder, name_files, labels, transform, batch_size=100):
    y_true = []
    y_pred = []
    num_images = len(name_files)
    for start_idx in tqdm(range(0, num_images, batch_size)):
        end_idx = min(start_idx + batch_size, num_images)
        batch_images = []
        batch_labels = []
        for idx in range(start_idx, end_idx):
            image_file = name_files[idx]
            image_path = os.path.join(image_folder, image_file)
            label = labels[idx]
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                print(f"Error loading image: {image_path}")
                continue
            image = transform(image)
            batch_images.append(image)
            batch_labels.append(label)
        batch_images = np.array(batch_images)
        batch_labels = np.array(batch_labels)
        batch_pred = model_to_evaluate.predict(batch_images)
        y_true.extend(batch_labels)
        y_pred.extend(batch_pred)
    accuracy = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return accuracy, cm

def plot_confusion_matrix(cm, classes, title, fold, phase, save_dir, normalize=False):
    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        cm_normalized = cm

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f" if normalize else "d", 
                cmap="coolwarm", xticklabels=classes, yticklabels=classes,
                cbar=True, annot_kws={"size": 12})
    
    plt.title(f"{title} - Fold {fold} ({phase})", fontsize=14)
    plt.xlabel("Predicted label", fontsize=12)
    plt.ylabel("True label", fontsize=12)

    for i in range(len(classes)):
        for j in range(len(classes)):
            if i == j:
                plt.gca().add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='red', lw=2))

    plt.tight_layout()
    file_path = os.path.join(save_dir, f"{title}_fold_{fold}_{phase}.png")
    plt.savefig(file_path)
    plt.close()
    print(f"Confusion matrix saved to {file_path}")


def calculate_class_distribution(labels, num_classes):
    counts = Counter(labels)
    total = sum(counts.values())
    percentages = {cls: round((counts[cls] / total) * 100, 1) for cls in range(num_classes)}
    return percentages

kf = MultilabelStratifiedKFold(n_splits=N_SPLITS, shuffle=True)

clf = SGDClassifier()

log_data = {"models": {}}
test_accuracies = {}

for fold, (train_idx, test_idx) in enumerate(kf.split(image_ids, labels)):
    print(f"--- Fold {fold + 1} ---")
    
    sgdc_hardness = clone(clf)
    sgdc_honey = clone(clf)
    sgdc_capped = clone(clf)

    train_ids = np.array(image_ids)[train_idx]
    test_ids = np.array(image_ids)[test_idx]

    train_files = []
    for image_id in train_ids:
        for file in id_to_images[image_id]:
            train_files.append(file)

    test_files = []
    for image_id in test_ids:
        for file in id_to_images[image_id]:
            test_files.append(file)

    train_labels_hardness = []
    train_labels_honey = []
    train_labels_capped = []
    for image_id in train_ids:
        for _ in id_to_images[image_id]:
            train_labels_hardness.append(id_to_labels[image_id][0])
            train_labels_honey.append(id_to_labels[image_id][1])
            train_labels_capped.append(id_to_labels[image_id][2])

    test_labels_hardness = []
    test_labels_honey = []
    test_labels_capped = []
    for image_id in test_ids:
        for _ in id_to_images[image_id]:
            test_labels_hardness.append(id_to_labels[image_id][0])
            test_labels_honey.append(id_to_labels[image_id][1])
            test_labels_capped.append(id_to_labels[image_id][2])

    train_data = list(zip(train_files, train_labels_hardness, train_labels_honey, train_labels_capped))
    test_data = list(zip(test_files, test_labels_hardness, test_labels_honey, test_labels_capped))
    
    random.shuffle(train_data)
    random.shuffle(test_data)
    
    train_files, train_labels_hardness, train_labels_honey, train_labels_capped = zip(*train_data)
    test_files, test_labels_hardness, test_labels_honey, test_labels_capped = zip(*test_data)

    train_files = list(train_files)
    train_labels_hardness = list(train_labels_hardness)
    train_labels_honey = list(train_labels_honey)
    train_labels_capped = list(train_labels_capped)
    test_files = list(test_files)
    test_labels_hardness = list(test_labels_hardness)
    test_labels_honey = list(test_labels_honey)
    test_labels_capped = list(test_labels_capped)

    for task, model, train_labels, test_labels, task_name in [
        (sgdc_hardness, "hardness", train_labels_hardness, test_labels_hardness, "hardness"),
        (sgdc_honey, "honey", train_labels_honey, test_labels_honey, "honey"),
        (sgdc_capped, "capped", train_labels_capped, test_labels_capped, "capped"),
    ]:
        print(f"Training {task_name} model for fold {fold + 1}...")
        logging.info(f"Training {task_name} model for fold {fold + 1}...")
        learn_in_batches(task, IMAGES_DIR, train_files, train_labels, transform, batch_size=100)

        logging.info("--- Rozklad klas w zbiorze treningowym: ---")
        logging.info(f"Hardness: {calculate_class_distribution(train_labels_hardness, NUM_CLASSES_HARDNESS)}")
        logging.info(f"Honey: {calculate_class_distribution(train_labels_honey, NUM_CLASSES_HONEY)}")
        logging.info(f"Capped: {calculate_class_distribution(train_labels_capped, NUM_CLASSES_CAPPED)}")
        
        print("--- Rozklad klas w zbiorze treningowym: ---")
        print(f"Hardness: {calculate_class_distribution(train_labels_hardness, NUM_CLASSES_HARDNESS)}")
        print(f"Honey: {calculate_class_distribution(train_labels_honey, NUM_CLASSES_HONEY)}")
        print(f"Capped: {calculate_class_distribution(train_labels_capped, NUM_CLASSES_CAPPED)}")

        print(f"Evaluating {task_name} model for fold {fold + 1}...")
        logging.info(f"Evaluating {task_name} model for fold {fold + 1}...")

        accuracy, cm = evaluate_in_batches(task, IMAGES_DIR, test_files, test_labels, transform, batch_size=100)

        print(f"Fold {fold + 1} Balanced Accuracy ({task_name}): {accuracy:.3f}")
        logging.info(f"Fold {fold + 1} Balanced Accuracy ({task_name}): {accuracy:.3f}")

        classes = range(NUM_CLASSES_HARDNESS) if task_name == "hardness" else range(NUM_CLASSES_HONEY) if task_name == "honey" else range(NUM_CLASSES_CAPPED)
        plot_confusion_matrix(cm, classes=classes, 
                              title=f"{task_name.capitalize()} Confusion Matrix", 
                              fold=fold + 1, phase="Test", save_dir=SAVE_DIR)

        test_accuracies[f"{task_name}_fold_{fold + 1}"] = accuracy

        model_path = os.path.join(SAVE_DIR, f"sgdc_{task_name}_fold_{fold + 1}.joblib")
        joblib.dump(task, model_path)
        log_data["models"].setdefault(task_name, {})[f"fold_{fold + 1}"] = {
            "accuracy": accuracy,
            "model_path": model_path,
        }

print(f"Training and evaluation completed. Results saved in {LOG_FILE}")
logging.info("Training and evaluation completed.")
