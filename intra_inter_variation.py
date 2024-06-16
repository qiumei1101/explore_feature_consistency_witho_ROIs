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

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["font.size"] = 12

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
    elif name == 'swin_base_patch4_window7_224':
        model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=0)
    return model

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
        # Normalize features
        class_features = (class_features - np.mean(class_features, axis=0)) / np.std(class_features, axis=0)
        features[cls] = (class_features, class_labels, dataset)
        
    return features

# Function to calculate intra-class variation
def calculate_intra_class_variation(features):
    intra_class_variations = []
    for vehicle_id in set(features[1]):
        idx = np.where(features[1] == vehicle_id)[0]
        if len(idx) > 1:
            vehicle_features = features[0][idx]
            variances = np.var(vehicle_features, axis=0)
            intra_class_variations.append(np.mean(variances))
    return np.mean(intra_class_variations)

# Function to calculate inter-class variation
def calculate_inter_class_variation(features1, features2):
    inter_class_variations = []
    common_vehicle_ids = set(features1[1]) & set(features2[1])
    
    for vehicle_id in common_vehicle_ids:
        idx1 = np.where(features1[1] == vehicle_id)[0]
        idx2 = np.where(features2[1] == vehicle_id)[0]
        
        if len(idx1) > 0 and len(idx2) > 0:
            for i in idx1:
                for j in idx2:
                    feature1 = features1[0][i]
                    feature2 = features2[0][j]
                    variance = np.var(feature1 - feature2)
                    inter_class_variations.append(variance)
                    
    return np.mean(inter_class_variations)

# Move models to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths to your data folders
data_dir = '/home/meiqiu@ads.iu.edu/Mei_all/feature_extraction/weaving_in_outroi_backup/weaving1'  # Modify this with your actual data path

# Extract features and calculate variations
results = {}
for model_name, model_key in models_dict.items():
    model = load_model(model_key).to(device)
    model_features = load_and_extract_features(data_dir, model, transform)
    
    intra_class_variation_cam1 = calculate_intra_class_variation(model_features['cam1_inroi'])
    intra_class_variation_cam2 = calculate_intra_class_variation(model_features['cam2_inroi'])
    intra_class_variation_cam3 = calculate_intra_class_variation(model_features['cam2_outroi'])

    inter_class_variation_cam1_cam2_inroi = calculate_inter_class_variation(model_features['cam1_inroi'], model_features['cam2_inroi'])
    inter_class_variation_cam1_cam2_outroi = calculate_inter_class_variation(model_features['cam1_inroi'], model_features['cam2_outroi'])
    # inter_class_variation_cam2_inroi_cam2_outroi = calculate_inter_class_variation(model_features['cam2_inroi'], model_features['cam2_outroi'])

    results[model_name] = {
        'intra_class_variation_cam1': intra_class_variation_cam1,
        'intra_class_variation_cam2': intra_class_variation_cam2,
        'intra_class_variation_cam3': intra_class_variation_cam3,
        'inter_class_variation_cam1_cam2_inroi': inter_class_variation_cam1_cam2_inroi,
        'inter_class_variation_cam1_cam2_outroi': inter_class_variation_cam1_cam2_outroi,
        # 'inter_class_variation_cam2_inroi_cam2_outroi': inter_class_variation_cam2_inroi_cam2_outroi
    }

# Plotting the results
fig, ax = plt.subplots(figsize=(10, 6))
x_labels = ['Intra-Cam1_Inroi', 'Intra-Cam2_Inroi', 'Intra-Cam2_Outroi', 
            'Inter-Cam1_Inroi-Cam2_Inroi', 'Inter-Cam1_Inroi-Cam2_Outroi']
x = np.arange(len(x_labels))

bar_width = 0.15
for i, (model_name, res) in enumerate(results.items()):
    metrics = [res['intra_class_variation_cam1'], res['intra_class_variation_cam2'], res['intra_class_variation_cam3'], 
               res['inter_class_variation_cam1_cam2_inroi'], res['inter_class_variation_cam1_cam2_outroi']]
    plt.bar(x + i * bar_width, metrics, bar_width, label=model_name)

plt.title('Intra-Class and Inter-Class Variations Across Cameras', fontsize=15, fontweight='bold')
plt.ylabel('Variation', fontsize=15, fontweight='bold')
plt.xticks(x + bar_width * 1.5, x_labels, rotation=25, fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.tight_layout()
plt.savefig('intra_inter_class_variations_weaving1.png')
plt.show()
