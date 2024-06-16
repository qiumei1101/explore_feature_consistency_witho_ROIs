import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the transformation for the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Custom dataset to load images from subfolders
class VehicleDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for label in os.listdir(root_dir):
            subfolder_path = os.path.join(root_dir, label)
            if os.path.isdir(subfolder_path):
                for img_name in os.listdir(subfolder_path):
                    img_path = os.path.join(subfolder_path, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = transforms.functional.pil_to_tensor(transforms.functional.pil_image.open(img_path).convert('RGB'))
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Function to extract features using a pre-trained ResNet50 model
def extract_features(dataloader, model):
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for images, batch_labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            outputs = outputs.view(outputs.size(0), -1)
            features.append(outputs.cpu().numpy())
            labels.extend(batch_labels)

    return np.concatenate(features), labels

# Load pre-trained ResNet50 model
resnet = models.resnet50(pretrained=True).to(device)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove the last classification layer

# Load datasets from two cameras
dataset1 = VehicleDataset(root_dir='path_to_camera1', transform=transform)
dataset2 = VehicleDataset(root_dir='path_to_camera2', transform=transform)

dataloader1 = DataLoader(dataset1, batch_size=32, shuffle=False, num_workers=4)
dataloader2 = DataLoader(dataset2, batch_size=32, shuffle=False, num_workers=4)

# Extract features from both datasets
features1, labels1 = extract_features(dataloader1, resnet)
features2, labels2 = extract_features(dataloader2, resnet)

# Calculate cosine similarity between all pairs of features from both datasets
similarity_matrix = cosine_similarity(features1, features2)

# Define a threshold for matching
threshold = 0.8

# Create a dictionary to store matching subfolders
matches = {}

for i, label1 in enumerate(labels1):
    for j, label2 in enumerate(labels2):
        if similarity_matrix[i, j] >= threshold:
            if label1 not in matches:
                matches[label1] = []
            matches[label1].append(label2)

# Save the matching list to a JSON file
with open('matching_list.json', 'w') as f:
    json.dump(matches, f)

print("Matching list saved to matching_list.json")
