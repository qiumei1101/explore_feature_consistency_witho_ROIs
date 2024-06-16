import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
from timm import create_model  # For ViT model
from torch.utils.data import DataLoader
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["font.size"] = 18
# p.rcParams.update({'font.size': 22})

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define the path to the dataset
data_dir = '/home/meiqiu@ads.iu.edu/Mei_all/Data_VeReID/tracking_images_inroi_classfolder'

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load ResNet50 model
resnet = models.resnet50(pretrained=True)
# Load ViT model
vit = create_model('vit_base_patch16_224', pretrained=True)

# Remove the last classification layer to get the features
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
vit = torch.nn.Sequential(*list(vit.children())[:-1])

# Function to extract features from the models
def extract_features(model, dataloader):
    model.eval()
    features = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.cuda()
            output = model(images)
            output = output.view(output.size(0), -1)
            features.append(output.cpu().numpy())
    return np.concatenate(features, axis=0)

# Function to load dataset and extract features for each condition
def load_and_extract_features(data_dir, model, transform):
    features = {}
    conditions = os.listdir(data_dir)
    for condition in conditions:
        condition_path = os.path.join(data_dir, condition)
        inside_path = os.path.join(condition_path, 'inroi')
        outside_path = os.path.join(condition_path, 'outroi')

        inside_dataset = datasets.ImageFolder(root=inside_path, transform=transform)
        outside_dataset = datasets.ImageFolder(root=outside_path, transform=transform)

        inside_dataloader = DataLoader(inside_dataset, batch_size=32, shuffle=True, num_workers=2)
        outside_dataloader = DataLoader(outside_dataset, batch_size=32, shuffle=True, num_workers=2)

        inside_features = extract_features(model, inside_dataloader)
        outside_features = extract_features(model, outside_dataloader)

        features[condition] = {'inroi': inside_features, 'outroi': outside_features}
    return features

# Move models to GPU
resnet = resnet.cuda()
vit = vit.cuda()

# Extract features for ResNet50 and ViT
resnet_features = load_and_extract_features(data_dir, resnet, transform)
vit_features = load_and_extract_features(data_dir, vit, transform)

# Apply T-SNE
def apply_tsne(features, label):
    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(features)
    return reduced_features

# Plot T-SNE results
def plot_tsne(tsne_results, title, color):
    plt.figure(figsize=(10, 6))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], s=5, c=color)
    plt.title(title,fontsize=15, fontweight='bold')
    plt.xlabel('TSNE Component 1', fontsize=15, fontweight='bold')
    plt.ylim(-100, 100)
    plt.xlim(-100, 100)
    plt.ylabel('TSNE Component 2', fontsize=15, fontweight='bold')
    plt.savefig(str(title) + '.png')
    plt.show()

# Process and plot T-SNE results for all conditions
for condition in resnet_features:
    resnet_inside_tsne = apply_tsne(resnet_features[condition]['inroi'], f'ResNet Inside {condition}')
    resnet_outside_tsne = apply_tsne(resnet_features[condition]['outroi'], f'ResNet Outside {condition}')
    vit_inside_tsne = apply_tsne(vit_features[condition]['inroi'], f'ViT Inside {condition}')
    vit_outside_tsne = apply_tsne(vit_features[condition]['outroi'], f'ViT Outside {condition}')

    plot_tsne(resnet_inside_tsne, f'ResNet50 Features Inside ROI from {condition}', 'pink')
    plot_tsne(resnet_outside_tsne, f'ResNet50 Features Outside ROI from {condition}', 'blue')
    plot_tsne(vit_inside_tsne, f'ViT Features Inside ROI from {condition}', 'pink')
    plot_tsne(vit_outside_tsne, f'ViT Features Outside ROI from {condition}', 'blue')
