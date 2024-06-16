import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import timm
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from timm import create_model
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
# mean, std = compute_mean_std(data_dir)

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

# Move models to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Extract features for each model and plot decision boundaries
def extract_and_plot_decision_boundaries(models_dict, data_dir, transform):
    for model_name, model_key in models_dict.items():
        model, num_features = load_model(model_key)
        model = model.to(device)
        model_features = load_and_extract_features(data_dir, model, transform)
        
        features_cam1_inroi = model_features['cam1_inroi'][0]
        features_cam2_inroi = model_features['cam2_inroi'][0]
        features_cam2_outroi = model_features['cam2_outroi'][0]

        # Labels for binary classification
        labels_in_in = np.concatenate([np.zeros(len(features_cam1_inroi)), np.ones(len(features_cam2_inroi))])
        labels_in_out = np.concatenate([np.zeros(len(features_cam1_inroi)), np.ones(len(features_cam2_outroi))])

        # Features for binary classification
        features_in_in = np.concatenate([features_cam1_inroi, features_cam2_inroi])
        features_in_out = np.concatenate([features_cam1_inroi, features_cam2_outroi])

        # Apply t-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42)
        features_in_in_tsne = tsne.fit_transform(features_in_in)
        features_in_out_tsne = tsne.fit_transform(features_in_out)

        # Standardize the features
        scaler_in_in = StandardScaler().fit(features_in_in_tsne)
        features_in_in_tsne = scaler_in_in.transform(features_in_in_tsne)

        scaler_in_out = StandardScaler().fit(features_in_out_tsne)
        features_in_out_tsne = scaler_in_out.transform(features_in_out_tsne)

        # Train logistic regression classifier
        clf_in_in = LogisticRegression(random_state=42).fit(features_in_in_tsne, labels_in_in)
        clf_in_out = LogisticRegression(random_state=42).fit(features_in_out_tsne, labels_in_out)

        # Create a mesh grid for plotting decision boundaries
        h = .02  # step size in the mesh
        x_min_in_in, x_max_in_in = features_in_in_tsne[:, 0].min() - 1, features_in_in_tsne[:, 0].max() + 1
        y_min_in_in, y_max_in_in = features_in_in_tsne[:, 1].min() - 1, features_in_in_tsne[:, 1].max() + 1
        xx_in_in, yy_in_in = np.meshgrid(np.arange(x_min_in_in, x_max_in_in, h),
                                         np.arange(y_min_in_in, y_max_in_in, h))
        Z_in_in = clf_in_in.predict(np.c_[xx_in_in.ravel(), yy_in_in.ravel()])
        Z_in_in = Z_in_in.reshape(xx_in_in.shape)

        x_min_in_out, x_max_in_out = features_in_out_tsne[:, 0].min() - 1, features_in_out_tsne[:, 0].max() + 1
        y_min_in_out, y_max_in_out = features_in_out_tsne[:, 1].min() - 1, features_in_out_tsne[:, 1].max() + 1
        xx_in_out, yy_in_out = np.meshgrid(np.arange(x_min_in_out, x_max_in_out, h),
                                           np.arange(y_min_in_out, y_max_in_out, h))
        Z_in_out = clf_in_out.predict(np.c_[xx_in_out.ravel(), yy_in_out.ravel()])
        Z_in_out = Z_in_out.reshape(xx_in_out.shape)

        # Plot decision boundaries
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        ax[0].contourf(xx_in_in, yy_in_in, Z_in_in, alpha=0.3, cmap=plt.cm.Paired)
        ax[0].scatter(features_in_in_tsne[labels_in_in == 0][:, 0], features_in_in_tsne[labels_in_in == 0][:, 1], label='cam1_inroi', c='blue', marker='o')
        ax[0].scatter(features_in_in_tsne[labels_in_in == 1][:, 0], features_in_in_tsne[labels_in_in == 1][:, 1], label='cam2_inroi', c='green', marker='s')
        ax[0].set_title(f'{model_name}: cam1_inroi vs cam2_inroi')
        ax[0].legend()

        ax[1].contourf(xx_in_out, yy_in_out, Z_in_out, alpha=0.3, cmap=plt.cm.Paired)
        ax[1].scatter(features_in_out_tsne[labels_in_out == 0][:, 0], features_in_out_tsne[labels_in_out == 0][:, 1], label='cam1_inroi', c='blue', marker='o')
        ax[1].scatter(features_in_out_tsne[labels_in_out == 1][:, 0], features_in_out_tsne[labels_in_out == 1][:, 1], label='cam2_outroi', c='red', marker='x')
        ax[1].set_title(f'{model_name}: cam1_inroi vs cam2_outroi')
        ax[1].legend()

        plt.suptitle(f'Decision Boundaries for {model_name}', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f'decision_boundaries_{model_name}.png')
        plt.show()

# Extract features and plot decision boundaries
extract_and_plot_decision_boundaries(models_dict, data_dir, transform)
