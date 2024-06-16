import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
from timm import create_model  # For ViT and Swin-Transformer models
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import entropy
from PIL import Image

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["font.size"] = 15

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set GPU devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

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
data_dir = '/home/meiqiu@ads.iu.edu/Mei_all/Data_VeReID/tracking_images_inroi_classfolder'
mean, std = compute_mean_std(data_dir)

# Define the path to the dataset
# data_dir = '/home/meiqiu@ads.iu.edu/Mei_all/Data_VeReID/tracking_images_inroi_classfolder'

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

# Function to extract features from the models
def extract_features(model, dataloader):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for images, label in dataloader:
            images = images.cuda()
            output = model(images)         
            features.append(output.cpu().numpy())
            labels.append(label.cpu().numpy())
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)

# Function to normalize features using softmax
def normalize_features(features):
    exp_features = np.exp(features)
    return exp_features / np.sum(exp_features, axis=1, keepdims=True)

# Function to load dataset and extract features for each condition
def load_and_extract_features(data_dir, model, transform, conditions_order):
    features = {}
    for condition in conditions_order:
        condition_path = os.path.join(data_dir, condition)
        inside_path = os.path.join(condition_path, 'inroi')
        outside_path = os.path.join(condition_path, 'outroi')

        inside_dataset = datasets.ImageFolder(root=inside_path, transform=transform)
        outside_dataset = datasets.ImageFolder(root=outside_path, transform=transform)

        inside_dataloader = DataLoader(inside_dataset, batch_size=32, shuffle=True, num_workers=2)
        outside_dataloader = DataLoader(outside_dataset, batch_size=32, shuffle=True, num_workers=2)

        inside_features, inside_labels = extract_features(model, inside_dataloader)
        outside_features, outside_labels = extract_features(model, outside_dataloader)

        features[condition] = {'inroi': (inside_features, inside_labels), 'outroi': (outside_features, outside_labels)}
    return features

# Function to calculate entropy for each feature vector and average them using vectorized operations
def calculate_average_entropy(features):
    normalized_features = normalize_features(features)
    entropies = entropy(normalized_features.T, base=2)
    return np.mean(entropies)

# Move models to GPU and use DataParallel for multi-GPU support
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the condition order
conditions_order = ['sunny1', 'sunny2', 'rainy1', 'rainy2', 'night1', 'night2', 'congestion1', 'congestion2']

# Extract features for each model
def extract_features_for_models(models_dict, data_dir, transform, conditions_order):
    features_dict = {}
    for model_name, model_key in models_dict.items():
        print(f"Processing model: {model_name}")
        model, num_features = load_model(model_key)
        model = torch.nn.DataParallel(model).to(device)
        model_features = load_and_extract_features(data_dir, model, transform, conditions_order)
        features_dict[model_name] = model_features
        print(f"Completed model: {model_name}")
    return features_dict

# Extract features
features_dict = extract_features_for_models(models_dict, data_dir, transform, conditions_order)

# Calculate and plot average entropy
def plot_average_entropy(models_dict, features_dict, conditions_order, save_name):
    plt.figure(figsize=(10, 6))
    plt.ylim(0, 16)
    for model_name in models_dict.keys():
        avg_entropy_inroi = []
        avg_entropy_outroi = []
        for condition in conditions_order:
            inroi_features, _ = features_dict[model_name][condition]['inroi']
            outroi_features, _ = features_dict[model_name][condition]['outroi']

            entropy_inroi = calculate_average_entropy(inroi_features)
            entropy_outroi = calculate_average_entropy(outroi_features)

            avg_entropy_inroi.append(entropy_inroi)
            avg_entropy_outroi.append(entropy_outroi)
        
        plt.plot(conditions_order, avg_entropy_inroi, marker='o', label=f'{model_name} Inside ROI')
        plt.plot(conditions_order, avg_entropy_outroi, marker='s', label=f'{model_name} Outside ROI')

    plt.title('Average Information Entropy Across Conditions', fontsize=13, fontweight='bold')
    plt.xlabel('Condition', fontsize=13, fontweight='bold')
    plt.ylabel('Average Information Entropy', fontsize=13, fontweight='bold')
    plt.xticks(rotation=15)
    plt.legend(loc='best', fontsize=10)
    plt.savefig(save_name, bbox_inches='tight')
    plt.show()

# Plot average entropy for each model
plot_average_entropy(models_dict, features_dict, conditions_order, 'average_information_entropy.png')
