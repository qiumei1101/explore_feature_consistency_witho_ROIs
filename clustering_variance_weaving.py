import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import timm
from PIL import Image
from timm import create_model
from scipy.stats import entropy

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["font.size"] = 11

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Function to compute mean and std of the dataset
def compute_mean_std(data_dir):
    image_paths = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_paths.append(os.path.join(root, file))

    mean = np.zeros(3)
    std = np.zeros(3)
    count = 0
    
    for path in image_paths:
        try:
            image = Image.open(path).convert('RGB')
            np_image = np.array(image) / 255.0
            mean += np.mean(np_image, axis=(0, 1))
            std += np.std(np_image, axis=(0, 1))
            count += 1
        except Exception as e:
            print(f"Error processing {path}: {e}")

    if count > 0:
        mean /= count
        std /= count

    return mean, std

# Compute the mean and std of the dataset
data_dir = '/home/meiqiu@ads.iu.edu/Mei_all/feature_extraction/weaving_in_outroi_backup/weaving1'
mean, std = compute_mean_std(data_dir)

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

# Load and modify models
def load_model(name):
    if name == 'resnet50':
        model = models.resnet50(pretrained=True)
        num_features = model.fc.in_features
        model.fc = torch.nn.Identity()
    elif name == 'resnext50_32x4d':
        model = models.resnext50_32x4d(pretrained=True)
        num_features = model.fc.in_features
        model.fc = torch.nn.Identity()
    elif name == 'vit_base_patch16_224':
        model = create_model('vit_base_patch16_224', pretrained=True)
        num_features = model.head.in_features
        model.head = torch.nn.Identity()
    elif name == 'swin_base_patch4_window7_224':
        model = create_model('swin_base_patch4_window7_224', pretrained=True)
        num_features = model.head.in_features
        model.head = torch.nn.Identity()
    return model, num_features

models_dict = {
    'ResNet50': 'resnet50',
    'ResNeXt50': 'resnext50_32x4d',
    'ViT': 'vit_base_patch16_224',
    'Swin-Transformer': 'swin_base_patch4_window7_224'
}

# Custom Dataset to load subfolder names as labels
class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

# Function to extract features from the models
def extract_features(model, dataloader):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for images, label in dataloader:
            images = images.cuda()
            output = model(images)
            output = output.view(output.size(0), -1)  # Flatten the features
            features.append(output.cpu().numpy())
            labels.append(label.cpu().numpy())
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)

# Function to load dataset and extract features for each class
def load_and_extract_features(data_dir, model, transform):
    features = {}
    classes = ['cam1_inroi', 'cam2_inroi', 'cam2_outroi']

    for cls in classes:
        class_path = os.path.join(data_dir, cls)
        dataset = CustomImageFolder(root=class_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
        class_features, class_labels = extract_features(model, dataloader)
        features[cls] = (class_features, class_labels, dataset)
        
    return features

# Function to calculate cross-entropy between two sets of features
def calculate_cross_entropy(features1, features2):
    hist1, _ = np.histogram(features1.flatten(), bins=128, density=True)
    hist2, _ = np.histogram(features2.flatten(), bins=128, density=True)
    hist1 += 1e-12  # to avoid log(0)
    hist2 += 1e-12  # to avoid log(0)
    return entropy(hist1, hist2)

# Function to normalize features by subtracting the average feature of the corresponding folder
def normalize_features(features):
    mean_feature = np.mean(features, axis=0)
    normalized_features = features - mean_feature
    return normalized_features

# Move models to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Extract features for each model and calculate cross-entropy
def extract_and_calculate_entropy(models_dict, data_dir, transform):
    entropy_dict = {}

    for model_name, model_key in models_dict.items():
        model, num_features = load_model(model_key)
        model = model.to(device)
        model_features = load_and_extract_features(data_dir, model, transform)
        
        # Normalize features
        features_cam1_inroi = normalize_features(model_features['cam1_inroi'][0])
        features_cam2_inroi = normalize_features(model_features['cam2_inroi'][0])
        features_cam2_outroi = normalize_features(model_features['cam2_outroi'][0])

        cross_entropy_in_in = calculate_cross_entropy(features_cam1_inroi, features_cam2_inroi)
        cross_entropy_in_out = calculate_cross_entropy(features_cam1_inroi, features_cam2_outroi)

        entropy_dict[model_name] = {
            'cross_entropy_in_in': cross_entropy_in_in,
            'cross_entropy_in_out': cross_entropy_in_out
        }

    return entropy_dict

# Calculate entropy metrics
entropy_dict = extract_and_calculate_entropy(models_dict, data_dir, transform)

# Plot cross-entropy for all models
def plot_entropy(entropy_dict, save_name):
    bar_width = 0.15
    conditions = ['cross_entropy_in_in', 'cross_entropy_in_out']
    x = np.arange(len(conditions))
    
    plt.figure(figsize=(6, 6))
    for i, model_name in enumerate(entropy_dict.keys()):
        metrics = [entropy_dict[model_name][cond] for cond in conditions]
        plt.bar(x + i * bar_width, metrics, bar_width, label=f'{model_name}')

    plt.title('Cross-Entropy Across Cameras', fontsize=15, fontweight='bold')
    plt.ylabel('Value', fontsize=15, fontweight='bold')
    plt.xticks(x + bar_width * 1.5, conditions, rotation=15)
    plt.legend(loc='best', fontsize=10)
    plt.savefig(save_name)
    plt.show()

# Plot entropy metrics for all models
plot_entropy(entropy_dict, 'weaving_1_specific_vehicles_cross_entropy.png')
