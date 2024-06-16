import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
from timm import create_model  # For ViT model
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import ViTModel, ViTConfig
import torchvision.transforms as T
from pathlib import Path
import numpy as np
import PIL.Image as Image
from scipy.stats import spearmanr

# Load the configuration and create an instance of ViT
config = ViTConfig.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTModel(config)

# Load the pretrained ViT-B16 model
vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)
# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the dataset
# dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
# dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
data_dir = '/home/meiqiu@ads.iu.edu/Mei_all/feature_extraction/reid_challenges'
# image_paths = [f'/{i}.jpg' for i in range(1, 13)]
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

# Load ResNet50 model
resnet = models.resnet50(pretrained=True)
# Load ViT model
# vit = create_model('vit_base_patch16_224', pretrained=True)

# Remove the last classification layer to get the features
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
# vit = torch.nn.Sequential(*list(vit.children())[:-1])
# Function to extract features from the models
def extract_features_ViT(model, dataloader):
    # Forward pass, extract hidden states before the classifier

    model.eval()
    features = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.cuda()
            outputs = model(images, output_hidden_states=True)
    # Get the last hidden state (features before the classification head)
            last_hidden_state = outputs.last_hidden_state
    # Pool the outputs into a single batch wise vector (CLS token)
            pooled_output = last_hidden_state[:, 0]
            # output = model(images)
            # output = output.view(output.size(0), -1)
            features.append(pooled_output.cpu().numpy())
    return np.concatenate(features, axis=0)
    # return pooled_output

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
# Move models to GPU
resnet = resnet.cuda()
vit = vit.cuda()

# Extract features
resnet_features = extract_features(resnet, dataloader)
vit_features = extract_features_ViT(vit, dataloader)
print("resnet_features",resnet_features.shape)
print("vit_features",vit_features.shape)
# Determine the minimum feature size between ResNet and ViT
min_features = min(resnet_features.shape[1], vit_features.shape[1])

# Apply PCA to reduce feature size to the minimum feature size
pca = PCA(n_components=10)

# Fit PCA on ResNet features and transform
resnet_features_reduced = pca.fit_transform(resnet_features)
# Fit PCA on ViT features and transform
vit_features_reduced = pca.fit_transform(vit_features)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# Assume features_resnet and features_vit are the feature matrices
# Calculate Spearman Rank Correlation
correlation_matrix, p_value = spearmanr(resnet_features_reduced, vit_features_reduced,axis=0)
print("p_value",p_value)
# Create a heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation, annot=True, cmap='coolwarm')
# plt.title('Spearman Rank Correlation between ResNet50 and ViT Features')
# plt.show()

# correlation_matrix, _ = spearmanr(resnet_features_reduced, vit_features_reduced, axis=1)
print("Correlation matrix shape:", correlation_matrix.shape)
print("Average correlation:", np.mean(correlation_matrix))
plt.imshow(correlation_matrix, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Spearman Correlation between ResNet50 and ViT Features')
plt.xlabel('ViT Features')
plt.ylabel('ResNet50 Features')
plt.savefig("Spearman Correlation between ResNet50 and ViT Features.png")
plt.show()

# Compute cosine similarity between ResNet50 and ViT features
# similarity_matrix = cosine_similarity(resnet_features_reduced, vit_features_reduced)
# dissimilarity_matrix = 1 - similarity_matrix

# # Analyze the similarity matrix
# print("DisSimilarity matrix shape:", dissimilarity_matrix.shape)
# print("Average DisSimilarity:", np.mean(dissimilarity_matrix))
# # Plot similarity matrix
# plt.imshow(dissimilarity_matrix, cmap='hot', interpolation='nearest')
# plt.colorbar()
# plt.title('Representation DisSimilarity between ResNet50 and ViT')
# plt.xlabel('ViT Features')
# plt.ylabel('ResNet50 Features')
# plt.show()
