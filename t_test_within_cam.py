import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from transformers import ViTModel, SwinForImageClassification

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["font.size"] = 18

# Check if GPU is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VehicleDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for condition in os.listdir(root_dir):
            condition_path = os.path.join(root_dir, condition)
            if os.path.isdir(condition_path):
                for subfolder in ['inroi', 'outroi']:
                    subfolder_path = os.path.join(condition_path, subfolder)
                    if os.path.isdir(subfolder_path):
                        for vehicle_id in os.listdir(subfolder_path):
                            vehicle_path = os.path.join(subfolder_path, vehicle_id)
                            if os.path.isdir(vehicle_path):
                                for img_name in os.listdir(vehicle_path):
                                    img_path = os.path.join(vehicle_path, img_name)
                                    if os.path.isfile(img_path):  # Ensure it's a file
                                        self.image_paths.append(img_path)
                                        self.labels.append(condition + '_' + subfolder + '_' + vehicle_id)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the dataset
dataset = VehicleDataset(root_dir='/home/meiqiu@ads.iu.edu/Mei_all/Data_VeReID/tracking_images_inroi_classfolder', transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

# Model configuration dictionary
models_dict = {
    'ResNet50': models.resnet50(pretrained=True),
    'ResNeXt50': models.resnext50_32x4d(pretrained=True),
    'ViT': ViTModel.from_pretrained('google/vit-base-patch16-224-in21k'),
    'Swin-Transformer': SwinForImageClassification.from_pretrained('microsoft/swin-base-patch4-window7-224')
}

# Function to extract features
def extract_features(model, dataloader, model_type):
    features = {}
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])  # Use GPUs 0, 1, 2, 3
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            if model_type == 'ViT':
                outputs = model(images).last_hidden_state[:, 0].cpu().numpy()  # Extract the CLS token for ViT
            elif model_type == 'Swin-Transformer':
                outputs = model(images).logits.cpu().numpy()  # Use logits for Swin
            else:
                outputs = model(images).squeeze().cpu().numpy()
            for output, label in zip(outputs, labels):
                condition, subfolder, vehicle_id = label.split('_')[0], label.split('_')[1], label.split('_')[2]
                if condition not in features:
                    features[condition] = {'inroi': {}, 'outroi': {}}
                if vehicle_id not in features[condition][subfolder]:
                    features[condition][subfolder][vehicle_id] = []
                features[condition][subfolder][vehicle_id].append(output)
    return features

# Perform t-tests within each condition
def compute_cosine_similarity(features):
    condition_similarities = {}

    for condition in features:
        inroi_similarities = []
        outroi_similarities = []

        inroi_vehicles = features[condition]['inroi']
        outroi_vehicles = features[condition]['outroi']

        # Inroi similarities (same vehicle)
        for vehicle_id in inroi_vehicles:
            vehicle_features = inroi_vehicles[vehicle_id]
            for i in range(len(vehicle_features)):
                for j in range(i + 1, len(vehicle_features)):
                    sim = 1 - cosine(vehicle_features[i], vehicle_features[j])
                    inroi_similarities.append(sim)

        # Outroi similarities (same vehicle)
        for vehicle_id in outroi_vehicles:
            vehicle_features = outroi_vehicles[vehicle_id]
            for i in range(len(vehicle_features)):
                for j in range(i + 1, len(vehicle_features)):
                    sim = 1 - cosine(vehicle_features[i], vehicle_features[j])
                    outroi_similarities.append(sim)

        # Between similarities (same vehicle, different subsets)
        for vehicle_id in inroi_vehicles:
            if vehicle_id in outroi_vehicles:
                inroi_features = inroi_vehicles[vehicle_id]
                outroi_features = outroi_vehicles[vehicle_id]
                for inroi_feat in inroi_features:
                    for outroi_feat in outroi_features:
                        sim = 1 - cosine(inroi_feat, outroi_feat)
                        outroi_similarities.append(sim)

        condition_similarities[condition] = (inroi_similarities, outroi_similarities)

    return condition_similarities

# Draw result bar charts
def plot_results(model_name, features, save_path):
    condition_similarities = compute_cosine_similarity(features)

    t_test_results = {}
    for condition, (inroi_similarities, outroi_similarities) in condition_similarities.items():
        t_stat, p_value = ttest_ind(inroi_similarities, outroi_similarities)
        t_test_results[condition] = (t_stat, p_value)

    condition_order = ['sunny1', 'sunny2', 'rainy1', 'rainy2', 'night1', 'night2', 'congestion1', 'congestion2']
    inroi_means = [np.mean(condition_similarities[condition][0]) for condition in condition_order]
    outroi_means = [np.mean(condition_similarities[condition][1]) for condition in condition_order]
    inroi_stds = [np.std(condition_similarities[condition][0]) for condition in condition_order]
    outroi_stds = [np.std(condition_similarities[condition][1]) for condition in condition_order]
    p_values = [t_test_results[condition][1] for condition in condition_order]

    plt.figure(figsize=(8, 8))
    x = np.arange(len(condition_order))
    width = 0.3

    # Bar chart for means and standard deviations
    plt.bar(x - width/2, inroi_means, width, yerr=inroi_stds, capsize=5, label='Inroi-Inroi', color='blue')
    plt.bar(x + width/2, outroi_means, width, yerr=outroi_stds, capsize=5, label='Inroi-Outroi', color='red')
    plt.ylim(0.25, 1.25)

    # Show significance marker "*" if p-value < 0.05
    for i, p in enumerate(p_values):
        if p < 0.05:
            plt.text(x[i], max(inroi_means[i] + inroi_stds[i], outroi_means[i] + outroi_stds[i]) + 0.02, '*', ha='center', va='bottom', fontsize=20, color='red', fontweight='bold')

    plt.xlabel('Conditions', fontweight='bold', fontsize=15)
    plt.ylabel('Cosine Similarity', fontweight='bold', fontsize=15)
    plt.title(f'T-tests Inroi-Inroi vs Inroi-Outroi/{model_name}', fontweight='bold', fontsize=15)
    plt.xticks(x, condition_order, fontweight='bold', fontsize=15)
    plt.yticks(fontweight='bold')
    # plt.legend(fontsize=12, loc='upper right')
    plt.xticks(rotation=15)
    plt.legend(loc='best', fontsize=15)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Extract features and plot results for each model
for model_name, model in models_dict.items():
    if model_name in ['ViT', 'Swin-Transformer']:
        model.classifier = torch.nn.Identity()  # Remove the classification layer
    elif model_name in ['ResNet50', 'ResNeXt50']:
        model = torch.nn.Sequential(*list(model.children())[:-1])
    features = extract_features(model, dataloader, model_name)
    plot_results(model_name, features, f"{model_name}_t_test.png")
