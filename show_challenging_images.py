import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["font.size"] = 13
from scipy.spatial.distance import cosine

# Load a pre-trained model
model = models.resnet50(pretrained=True)
model.eval()

# Define a transform to preprocess the image
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image_path, model, preprocess):
    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

    with torch.no_grad():
        features = model(input_batch)
    
    return features.squeeze().numpy()

# Paths to your 12 images
image_paths = [f'/home/meiqiu@ads.iu.edu/Mei_all/feature_extraction/reid_challenges/{i}.jpg' for i in range(1, 13)]

# Extract features for all images
features = []
for image_path in image_paths:
    features.append(extract_features(image_path, model, preprocess))

features = np.array(features)


def calculate_similarity_matrix(features):
    n = features.shape[0]
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                similarity_matrix[i, j] = 1 - cosine(features[i], features[j])
            else:
                similarity_matrix[i, j] = 1.0  # Self-similarity is 1
    
    return similarity_matrix

similarity_matrix = calculate_similarity_matrix(features)


def visualize_similarity_matrix(similarity_matrix, image_paths):
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm', xticklabels=range(1,13), yticklabels=range(1,13))
    plt.title('Feature Cosine Similarity Matrix',fontweight='bold')
    plt.xlabel('Vehicles')
    plt.ylabel('Vehicles')
    plt.savefig("reid_challenges.png")
    plt.show()

# Display the similarity matrix
visualize_similarity_matrix(similarity_matrix, image_paths)
