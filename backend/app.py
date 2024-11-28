from flask import Flask, request, jsonify
import joblib
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from flask_cors import CORS
import base64
from PIL import Image
import io

import cv2
import numpy as np
from skimage.feature import hog

transform = transforms.Compose([
    transforms.ToTensor(),
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
        self.hardness_classifier = nn.Linear(512, 2)

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
        self.honey_classifier = nn.Linear(512, 2)

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
        self.capped_classifier = nn.Linear(512, 2)

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
lenet5_model = LeNet5MultiLabel()
lenet5_model.load_state_dict(torch.load("lenet5_model.pth"))
lenet5_model.eval()
app = Flask(__name__)

CORS(app)

@app.route("/")
def home():
    return "Welcome to the ML Prediction API!"

@app.route("/lenet", methods=["POST"])
def lenet():
    
    data = request.get_json()
    image_data = base64.b64decode(data["image_base64"])
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0)
    with torch.no_grad():
        hardness, honey, capped = lenet5_model(input_tensor)
    response = {
        "Prediction": {
            "Hardness": torch.argmax(hardness.argmax()).item(),
            "Honey": torch.argmax(honey).item(),
            "Cap": torch.argmax(capped).item()
        }
    }
    return jsonify(response)

@app.route("/gnb", methods=["POST"])
def gnb():
    model_cap = joblib.load("gnb/gnb_capped_fold_1.joblib")
    model_hard = joblib.load("gnb/gnb_hardness_fold_1.joblib")
    model_honey = joblib.load("gnb/gnb_honey_fold_1.joblib")

    try:
        data = request.get_json()
        image_data = base64.b64decode(data["image_base64"])
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image = image.resize((256, 256))
        image_rgb = np.array(image)
        hog_channels = []
        for channel_idx, channel in enumerate(cv2.split(image_rgb)):
            _, hog_image = hog(
                channel,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                visualize=True,
                block_norm='L2-Hys'
            )
            # Normalize the HOG image
            hog_image_rescaled = (hog_image - hog_image.min()) / (hog_image.max() - hog_image.min()) * 255
            hog_image_rescaled = hog_image_rescaled.astype(np.uint8)
            hog_channels.append(hog_image_rescaled)
        merged_hog_image = np.vstack(hog_channels)
        input_data = np.array(merged_hog_image).reshape(1, -1)
        pred_hard = model_hard.predict(input_data)
        pred_honey = model_honey.predict(input_data)
        pred_cap = model_cap.predict(input_data)

        print(pred_hard, pred_honey, pred_cap)

        response = {
            "Prediction": {
                "Hardness": int(pred_hard[0]),
                "Honey": int(pred_honey[0]),
                "Cap": int(pred_cap[0])
            }
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/sgdc", methods=["POST"])
def sgdc():
    model_cap = joblib.load("sgdc/sgdc_capped_fold_1.joblib")
    model_hard = joblib.load("sgdc/sgdc_hardness_fold_1.joblib")
    model_honey = joblib.load("sgdc/sgdc_honey_fold_1.joblib")

    try:
        data = request.get_json()
        image_data = base64.b64decode(data["image_base64"])
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image = image.resize((256, 256))
        image_rgb = np.array(image)
        hog_channels = []
        for channel_idx, channel in enumerate(cv2.split(image_rgb)):
            _, hog_image = hog(
                channel,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                visualize=True,
                block_norm='L2-Hys'
            )
            hog_image_rescaled = (hog_image - hog_image.min()) / (hog_image.max() - hog_image.min()) * 255
            hog_image_rescaled = hog_image_rescaled.astype(np.uint8)
            hog_channels.append(hog_image_rescaled)
        merged_hog_image = np.vstack(hog_channels)
        input_data = np.array(merged_hog_image).reshape(1, -1)
        pred_hard = model_hard.predict(input_data)
        pred_honey = model_honey.predict(input_data)
        pred_cap = model_cap.predict(input_data)

        print(pred_hard, pred_honey, pred_cap)

        response = {
            "Prediction": {
                "Hardness": int(pred_hard[0]),
                "Honey": int(pred_honey[0]),
                "Cap": int(pred_cap[0])
            }
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
