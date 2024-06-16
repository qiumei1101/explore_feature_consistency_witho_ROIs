import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
from timm import create_model  # For ViT and Swin-Transformer models
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
from PIL import Image

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["font.size"] = 15

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
            output = output.view(output.size(0), -1)  # Flatten the features
            features.append(output.cpu().numpy())
            labels.append(label.cpu().numpy())
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)

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

# Function to calculate RMSE
def calculate_rmse(features):
    variances = np.var(features, axis=0)
    rmse = np.sqrt(np.mean(variances))
    return rmse

# Move models to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the condition order
conditions_order = ['sunny1', 'sunny2', 'rainy1', 'rainy2', 'night1', 'night2', 'congestion1', 'congestion2']

# Extract features for each model and apply t-SNE
def extract_and_tsne(models_dict, data_dir, transform, conditions_order):
    features_dict = {}
    all_tsne_results = []

    for model_name, model_key in models_dict.items():
        model, num_features = load_model(model_key)
        model = model.to(device)
        model_features = load_and_extract_features(data_dir, model, transform, conditions_order)
        
        all_features = np.concatenate([model_features[condition]['inroi'][0] for condition in conditions_order] +
                                      [model_features[condition]['outroi'][0] for condition in conditions_order], axis=0)
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(all_features)
        all_tsne_results.append(tsne_results)
        
        tsne_index = 0
        for condition in conditions_order:
            model_features[condition]['inroi_tsne'] = tsne_results[tsne_index:tsne_index + len(model_features[condition]['inroi'][0])]
            tsne_index += len(model_features[condition]['inroi'][0])
            model_features[condition]['outroi_tsne'] = tsne_results[tsne_index:tsne_index + len(model_features[condition]['outroi'][0])]
            tsne_index += len(model_features[condition]['outroi'][0])
        
        features_dict[model_name] = model_features

    # Determine global min and max for the combined t-SNE results
    all_tsne_results = np.concatenate(all_tsne_results, axis=0)
    x_min, x_max = all_tsne_results[:, 0].min(), all_tsne_results[:, 0].max()
    y_min, y_max = all_tsne_results[:, 1].min(), all_tsne_results[:, 1].max()

    return features_dict, x_min, x_max, y_min, y_max

# Apply t-SNE and extract features
features_dict, x_min, x_max, y_min, y_max = extract_and_tsne(models_dict, data_dir, transform, conditions_order)

# Define colors for 'inroi' and 'outroi' for all conditions
inroi_color = 'blue'
outroi_color = 'red'
inroi_marker = 'o'
outroi_marker = 'x'

# Function to plot clustering results for each condition of each model
def plot_condition_clustering_tsne(features_dict, conditions_order, inroi_color, outroi_color, inroi_marker, outroi_marker, x_min, x_max, y_min, y_max):
    for model_name, features in features_dict.items():
        for i, condition in enumerate(conditions_order):
            plt.figure(figsize=(10, 6))
            inside_tsne = features[condition]['inroi_tsne']
            outside_tsne = features[condition]['outroi_tsne']

            plt.scatter(inside_tsne[:, 0], inside_tsne[:, 1], s=5, c=inroi_color, marker=inroi_marker, label='Inside ROI')
            plt.scatter(outside_tsne[:, 0], outside_tsne[:, 1], s=5, c=outroi_color, marker=outroi_marker, label='Outside ROI')

            plt.title(f'{model_name} Feature Clustering for {condition} (t-SNE)', fontsize=15, fontweight='bold')
            plt.xlabel('Component 1', fontsize=15, fontweight='bold')
            plt.ylabel('Component 2', fontsize=15, fontweight='bold')
            plt.xticks(rotation=15)
            plt.legend(loc='best', fontsize=11,ncol=2)
            # plt.legend(loc='upper right', fontsize=8, ncol=2)
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.savefig(f'{model_name}_{condition}_feature_clustering_tsne.png')
            plt.show()

# Plot clustering results for each condition of each model
plot_condition_clustering_tsne(features_dict, conditions_order, inroi_color, outroi_color, inroi_marker, outroi_marker, x_min, x_max, y_min, y_max)

# Plot RMSE of clustering variance
def plot_rmse(models_dict, features_dict, conditions_order, save_name):
    plt.figure(figsize=(10, 6))
    plt.ylim(0,50)
    for model_name in models_dict.keys():
        rmse_inroi = []
        rmse_outroi = []
        for condition in conditions_order:
            rmse_inroi.append(calculate_rmse(features_dict[model_name][condition]['inroi_tsne']))
            rmse_outroi.append(calculate_rmse(features_dict[model_name][condition]['outroi_tsne']))

        plt.plot(conditions_order, rmse_inroi, marker='o', label=f'{model_name} Inside ROI')
        plt.plot(conditions_order, rmse_outroi, marker='s', label=f'{model_name} Outside ROI')

    plt.title('RMSE of Clustering Variance Across Conditions', fontsize=15, fontweight='bold')
    plt.xlabel('Condition', fontsize=15, fontweight='bold')
    plt.ylabel('RMSE of Clustering Variance', fontsize=15, fontweight='bold')
    plt.xticks(rotation=15)
    plt.legend(loc='best', fontsize=11)
    plt.savefig(save_name)
    plt.show()

# Plot RMSE of clustering variance for all models
plot_rmse(models_dict, features_dict, conditions_order, 'rmse_of_clustering_variance.png')
