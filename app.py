
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import datasets, transforms, models
from PIL import Image
import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models, datasets
from PIL import Image
import os


# Define the CustomResNet model
class CustomResNet(torch.nn.Module):
    def __init__(self, num_classes=38):
        super(CustomResNet, self).__init__()
        self.model = models.resnet50(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Sequential(
            torch.nn.Linear(num_ftrs, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform_rn50 = transforms.Compose([
    transforms.Resize(size=232, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define directories
train_dir = 'train'
val_dir = 'valid'

# Check if directories exist
if not os.path.exists(train_dir):
    print(f"Error: Training directory not found at {train_dir}")
if not os.path.exists(val_dir):
    print(f"Error: Validation directory not found at {val_dir}")

# Define datasets
train_dataset_rn50 = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset_rn50 = datasets.ImageFolder(root=val_dir, transform=val_transform_rn50)

# Define class labels (ensure this list matches your dataset)
class_labels = train_dataset_rn50.classes

# Load the trained ResNet50 model

path_model = '01_plant_diseases_classification_pytorch_rn50-.pth'
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(nn.Linear(in_features=model.fc.in_features, out_features=38))
model.load_state_dict(torch.load(path_model, map_location=torch.device('cpu')))

model.eval()

# Define the transformation for the test images
preprocess = transforms.Compose([
    transforms.Resize(size=232, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image_bytes, model):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert('RGB')
        img_tensor = preprocess(img)
        img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            confidence = torch.nn.functional.softmax(outputs, dim=1)[0] * 100

        predicted_label = class_labels[predicted.item()]
        confidence_value = confidence[predicted.item()].item()
        return predicted_label, confidence_value
    except Exception as e:
        print("Error during prediction:", e)
        return "Error", 0.0

# Define the Flask app
app = Flask(__name__)
CORS(app)

# Check if the model is loaded
model_loaded = False

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    img_bytes = file.read()

    try:
        predicted_label, confidence_value = predict_image(img_bytes, model)
        response = jsonify({'prediction': predicted_label, 'confidence': confidence_value})
        return response
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Define the model status route
@app.route('/model_status', methods=['GET'])
def model_status():
    global model_loaded
    return jsonify({'model_loaded': model_loaded})

# Set model_loaded to True
model_loaded = True

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
