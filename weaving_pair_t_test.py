import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from scipy.stats import ttest_ind
import numpy as np
import matplotlib.pyplot as plt
import timm

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

# Initialize the device and set a single GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the transformation for the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Function to extract features from CNN-based models
def extract_features_cnn(model, folder_path):
    dataset = ImageFolder(root=folder_path, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    features = {}
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            for i, label in enumerate(labels):
                folder_name = dataset.classes[label]
                if folder_name not in features:
                    features[folder_name] = []
                features[folder_name].append(outputs[i].cpu().numpy())
    return features

# Function to extract features from transformer-based models
def extract_features_transformer(model, folder_path):
    dataset = ImageFolder(root=folder_path, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    features = {}
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            if hasattr(outputs, 'last_hidden_state'):
                outputs = outputs.last_hidden_state[:, 0, :]  # Use CLS token for ViT
            elif hasattr(outputs, 'pooler_output'):
                outputs = outputs.pooler_output  # Use pooler_output for Swin
            else:
                outputs = outputs
            for i, label in enumerate(labels):
                folder_name = dataset.classes[label]
                if folder_name not in features:
                    features[folder_name] = []
                features[folder_name].append(outputs[i].cpu().numpy())
    return features

# Function to compute cosine similarity between two sets of features
def cosine_similarity(features1, features2):
    norm1 = features1 / np.linalg.norm(features1, axis=1, keepdims=True)
    norm2 = features2 / np.linalg.norm(features2, axis=1, keepdims=True)
    return np.dot(norm1, norm2.T)

# Load pre-trained models and remove the last classification layer
def load_resnet50():
    model = models.resnet50(pretrained=True)
    model.fc = nn.Identity()  # Remove the fully connected layer
    return model.to(device)

def load_resnext50():
    model = models.resnext50_32x4d(pretrained=True)
    model.fc = nn.Identity()  # Remove the fully connected layer
    return model.to(device)

def load_vit():
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)  # Remove the classification layer
    return model.to(device)

def load_swin():
    model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=0)  # Remove the classification layer
    return model.to(device)

# Paths to your data folders
cam1_inroi_path = '/home/meiqiu@ads.iu.edu/Mei_all/feature_extraction/weaving_in_outroi_backup/weaving2/cam1_inroi'
cam2_inroi_path = '/home/meiqiu@ads.iu.edu/Mei_all/feature_extraction/weaving_in_outroi_backup/weaving2/cam2_inroi'
cam2_outroi_path = '/home/meiqiu@ads.iu.edu/Mei_all/feature_extraction/weaving_in_outroi_backup/weaving2/cam2_outroi'

# Load models
models = {
    'ResNet50': load_resnet50(),
    'ResNext50': load_resnext50(),
    'ViT-B/16': load_vit(),
    'Swin-Transformer': load_swin()
}

# Extract features for each model
results = {}
for model_name, model in models.items():
    model.eval()  # Ensure the model is in evaluation mode
    if model_name in ['ResNet50', 'ResNext50']:
        extract_features = extract_features_cnn
    else:
        extract_features = extract_features_transformer
    
    try:
        features_cam1_inroi = extract_features(model, cam1_inroi_path)
        features_cam2_inroi = extract_features(model, cam2_inroi_path)
        features_cam2_outroi = extract_features(model, cam2_outroi_path)
    except IndexError as e:
        print(f"Error processing model {model_name}: {e}")
        continue

    # Calculate cosine similarities and average them for each pair
    avg_sim_in_in = []
    avg_sim_in_out = []
    for key in features_cam1_inroi:
        if key in features_cam2_inroi:
            sim_in_in = cosine_similarity(np.array(features_cam1_inroi[key]), np.array(features_cam2_inroi[key]))
            avg_sim_in_in.extend(sim_in_in.flatten())
        if key in features_cam2_outroi:
            sim_in_out = cosine_similarity(np.array(features_cam1_inroi[key]), np.array(features_cam2_outroi[key]))
            avg_sim_in_out.extend(sim_in_out.flatten())

    # Convert lists to numpy arrays for statistical analysis
    avg_sim_in_in = np.array(avg_sim_in_in)
    avg_sim_in_out = np.array(avg_sim_in_out)

    # Log intermediate results for debugging
    print(f"Model: {model_name}")
    print(f"Avg Cosine Similarity (In-In): {avg_sim_in_in}")
    print(f"Avg Cosine Similarity (In-Out): {avg_sim_in_out}")

    # Statistical analysis
    t_stat, p_value = ttest_ind(avg_sim_in_in, avg_sim_in_out)
    mean_in_in = np.mean(avg_sim_in_in)
    mean_in_out = np.mean(avg_sim_in_out)
    std_in_in = np.std(avg_sim_in_in)
    std_in_out = np.std(avg_sim_in_out)

    # Log t-test results for debugging
    print(f"t-statistic: {t_stat}, p-value: {p_value}")

    results[model_name] = {
        'mean_in_in': mean_in_in,
        'mean_in_out': mean_in_out,
        'std_in_in': std_in_in,
        'std_in_out': std_in_out,
        'p_value': p_value
    }

# Plotting the results
fig, ax = plt.subplots()
x_labels = []
x_pos = []
means_in_in = []
means_in_out = []
stds_in_in = []
stds_in_out = []

for i, (model_name, res) in enumerate(results.items()):
    x_labels.extend([f'{model_name}\nCam1_In-Cam2_In', f'{model_name}\nCam1_In-Cam2_Out'])
    x_pos.extend([i*2, i*2+1])
    means_in_in.append(res['mean_in_in'])
    means_in_out.append(res['mean_in_out'])
    stds_in_in.append(res['std_in_in'])
    stds_in_out.append(res['std_in_out'])

# Plot bars
bars1 = ax.bar(x_pos[::2], means_in_in, yerr=stds_in_in, capsize=5, label='In-In')
bars2 = ax.bar(x_pos[1::2], means_in_out, yerr=stds_in_out, capsize=5, label='In-Out')

# Add significant markers
for i, (model_name, res) in enumerate(results.items()):
    if res['p_value'] < 0.05:
        ax.text(i*2 + 0.5, max(res['mean_in_in'] + res['std_in_in'], res['mean_in_out'] + res['std_in_out']) + 0.02, '*',
                ha='center', va='bottom', color='k', fontsize=14)

ax.set_ylabel('Cosine Similarity', fontsize=12)
ax.set_title('Cosine Similarity Comparison and T-Test of Different Models', fontweight='bold', fontsize=12)
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=10)
ax.set_ylim(0.3, 1.0)  # Adjust y-axis limits
ax.legend()

plt.tight_layout()
plt.savefig("weaving_2_models_comparison_cosine_similarity.png")
plt.show()
