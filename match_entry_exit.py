import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os
import json

# Load the pre-trained ResNet-50 model
model = models.resnet101(pretrained=True)
model.fc = torch.nn.Identity()  # Remove the last classifier layer
model.eval()
model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(folder_path):
    dataset = ImageFolder(root=folder_path, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    features = []
    labels = []

    with torch.no_grad():
        for images, labels_batch in loader:
            images = images.to('cuda' if torch.cuda.is_available() else 'cpu')
            output = model(images)
            features.append(output)
            labels.extend(labels_batch.numpy())  # Collecting batch labels

    # Calculate average features for each subfolder
    features = torch.cat(features, dim=0)
    unique_labels = list(set(labels))
    avg_features = [torch.mean(features[torch.tensor(labels) == label, :], dim=0) for label in unique_labels]
    return torch.stack(avg_features), unique_labels, dataset.classes

def match_vehicles(folder1, folder2):
    features1, labels1, classes1 = extract_features(folder1)
    features2, labels2, classes2 = extract_features(folder2)

    # Calculate cosine similarity between features
    similarity_matrix = torch.cosine_similarity(features1.unsqueeze(1), features2.unsqueeze(0), dim=2)

    matches = []
    matched_in_folder2 = set()
    for i in range(len(features1)):
        # Sort the similarities in descending order
        sorted_similarities, sorted_indices = torch.sort(similarity_matrix[i], descending=True)

        for similarity, j in zip(sorted_similarities, sorted_indices):
            if similarity.item() >= 0.9 and j.item() not in matched_in_folder2:
                matches.append({
                    'Folder1_Vehicle': classes1[labels1[i]],
                    'Folder2_Vehicle': classes2[labels2[j.item()]],
                    'Similarity_Score': similarity.item()
                })
                matched_in_folder2.add(j.item())
                break

    return matches

# Paths to the folders
folder1 = '/home/meiqiu@ads.iu.edu/Mei_all/feature_extraction/weaving_in_outroi/weaving1/entry_images/inroi'
folder2 = '/home/meiqiu@ads.iu.edu/Mei_all/feature_extraction/weaving_in_outroi/weaving1/exit_images/inroi'

# Run the matching process and print results
matches = match_vehicles(folder1, folder2)

# Save the matches to a JSON file
output_file = 'weaving_1_matches.json'
with open(output_file, 'w') as f:
    json.dump(matches, f, indent=4)

print(f"Matching results saved to {output_file}")
