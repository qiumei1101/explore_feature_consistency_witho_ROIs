import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np

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

def extract_features_and_scores(image_path, model, preprocess):
    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

    with torch.no_grad():
        features = model(input_batch)
        confidence_scores = F.softmax(features, dim=1)
    
    return features.squeeze(), confidence_scores.squeeze()

# Example usage
image_path = 'path_to_your_image.jpg'
features, confidence_scores = extract_features_and_scores(image_path, model, preprocess)


from scipy.spatial.distance import cosine

def calculate_feature_similarity(features):
    n = features.shape[0]
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                similarity_matrix[i, j] = 1 - cosine(features[i], features[j])
            else:
                similarity_matrix[i, j] = 1.0  # Self-similarity is 1
    
    return similarity_matrix

# Assuming `features` is a 2D tensor of shape (n_regions, n_features)
similarity_matrix = calculate_feature_similarity(features.numpy())
def calculate_confidence_differences(confidence_scores):
    n = confidence_scores.shape[0]
    difference_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            difference_matrix[i, j] = abs(confidence_scores[i] - confidence_scores[j])
    
    return difference_matrix

confidence_differences = calculate_confidence_differences(confidence_scores.numpy())
from scipy.stats import pearsonr

def calculate_correlation(difference_matrix, similarity_matrix):
    n = difference_matrix.shape[0]
    correlation_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            correlation_matrix[i, j], _ = pearsonr(difference_matrix[i], similarity_matrix[i])
    
    return correlation_matrix

correlation_matrix = calculate_correlation(confidence_differences, similarity_matrix)
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation between Confidence Score Differences and Feature Similarity')
plt.xlabel('Region Index')
plt.ylabel('Region Index')
plt.show()
