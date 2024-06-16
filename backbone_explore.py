import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
from timm import create_model
from torch.utils.data import DataLoader
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load and preprocess the dataset
data_dir = 'path/to/your/dataset'
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

# Load pre-trained models
resnet = models.resnet50(pretrained=True).eval().cuda()
vit = create_model('vit_base_patch16_224', pretrained=True).eval().cuda()

# Remove the last classification layer to get features
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
vit = torch.nn.Sequential(*list(vit.children())[:-1])

# Function to extract features
def extract_features(model, dataloader):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for images, target in dataloader:
            images = images.cuda()
            output = model(images)
            output = output.view(output.size(0), -1)
            features.append(output.cpu().numpy())
            labels.append(target.numpy())
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)

# Extract features
resnet_features, resnet_labels = extract_features(resnet, dataloader)
vit_features, vit_labels = extract_features(vit, dataloader)
# Apply PCA to reduce dimensions to 50 for t-SNE visualization
pca = PCA(n_components=50)
resnet_features_pca = pca.fit_transform(resnet_features)
vit_features_pca = pca.fit_transform(vit_features)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
resnet_features_tsne = tsne.fit_transform(resnet_features_pca)
vit_features_tsne = tsne.fit_transform(vit_features_pca)
def plot_features(features, labels, title):
    plt.figure(figsize=(10, 6))
    for label in np.unique(labels):
        indices = np.where(labels == label)
        plt.scatter(features[indices, 0], features[indices, 1], label=f'Class {label}', alpha=0.5)
    plt.title(title)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.show()

# Plot t-SNE features
plot_features(resnet_features_tsne, resnet_labels, 'ResNet50 t-SNE')
plot_features(vit_features_tsne, vit_labels, 'ViT t-SNE')
def add_noise(images, noise_factor=0.5):
    noise = noise_factor * torch.randn_like(images)
    return images + noise

def test_robustness(model, dataloader, perturbation_func):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = perturbation_func(images).cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.cuda()).sum().item()
    return correct / total

# Create a new DataLoader for testing robustness
dataloader_no_shuffle = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

# Test robustness to noise
resnet_noise_accuracy = test_robustness(resnet, dataloader_no_shuffle, add_noise)
vit_noise_accuracy = test_robustness(vit, dataloader_no_shuffle, add_noise)

print(f'ResNet50 accuracy with noise: {resnet_noise_accuracy * 100:.2f}%')
print(f'ViT accuracy with noise: {vit_noise_accuracy * 100:.2f}%')
from scipy.stats import ttest_ind

# Compare accuracies
resnet_accuracies = np.array([resnet_noise_accuracy])  # Extend this to multiple tests
vit_accuracies = np.array([vit_noise_accuracy])  # Extend this to multiple tests

t_stat, p_value = ttest_ind(resnet_accuracies, vit_accuracies)
print(f'T-test statistic: {t_stat}, p-value: {p_value}')
