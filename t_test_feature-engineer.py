import os
import random
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from skimage.feature import hog
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import torchvision.transforms as transforms

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

# Check if GPU is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VehicleDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for condition in os.listdir(root_dir):
            condition_path = os.path.join(root_dir, condition)
            if os.path.isdir(condition_path):
                for subfolder in ['inroi', 'outroi']:
                    subfolder_path = os.path.join(condition_path, subfolder)
                    if os.path.isdir(subfolder_path):
                        for vehicle_id in os.listdir(subfolder_path):
                            vehicle_path = os.path.join(subfolder_path, vehicle_id)
                            if os.path.isdir(vehicle_path):
                                for img_name in os.listdir(vehicle_path):
                                    img_path = os.path.join(vehicle_path, img_name)
                                    if os.path.isfile(img_path):  # Ensure it's a file
                                        self.image_paths.append(img_path)
                                        self.labels.append(condition + '_' + subfolder + '_' + vehicle_id)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load the dataset
dataset = VehicleDataset(root_dir='/home/meiqiu@ads.iu.edu/Mei_all/Data_VeReID/tracking_images_inroi_classfolder', transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

# Function to extract HOG features
def extract_hog_features(image):
    image = np.array(image.cpu().permute(1, 2, 0))  # Convert to HWC format
    features, _ = hog(image, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, multichannel=True)
    return features

# Extract features
def extract_features(dataloader):
    features = {}
    with torch.no_grad():
        for images, labels in dataloader:
            for image, label in zip(images, labels):
                image = image.to(device)
                hog_features = extract_hog_features(image)
                condition, subfolder, vehicle_id = label.split('_')[0], label.split('_')[1], label.split('_')[2]
                if condition not in features:
                    features[condition] = {'inroi': {}, 'outroi': {}}
                if vehicle_id not in features[condition][subfolder]:
                    features[condition][subfolder][vehicle_id] = []
                features[condition][subfolder][vehicle_id].append(hog_features)
    return features

features = extract_features(dataloader)

# Function to draw histograms and calculate cosine similarity for randomly selected pairs
def draw_histograms_and_cosine_similarity(features):
    same_vehicle_pairs = []
    different_vehicle_pairs = []

    # Collect pairs of the same vehicle
    for condition in features:
        for subfolder in features[condition]:
            for vehicle_id in features[condition][subfolder]:
                vehicle_features = features[condition][subfolder][vehicle_id]
                if len(vehicle_features) >= 2:
                    same_vehicle_pairs.append((vehicle_features[0], vehicle_features[1]))

    # Collect pairs of different vehicles
    all_features = [(features[condition][subfolder][vehicle_id][0], label)
                    for condition in features
                    for subfolder in features[condition]
                    for vehicle_id in features[condition][subfolder]]
    for i in range(5):
        pair = random.sample(all_features, 2)
        if pair[0][1] != pair[1][1]:  # Ensure they are from different vehicles
            different_vehicle_pairs.append((pair[0][0], pair[1][0]))

    # Plot histograms and calculate cosine similarity
    fig, axs = plt.subplots(5, 2, figsize=(20, 20))
    fig.suptitle('Histogram of HOG Features and Cosine Similarities', fontsize=16, fontweight='bold')

    for i, ((same_feat1, same_feat2), (diff_feat1, diff_feat2)) in enumerate(zip(same_vehicle_pairs, different_vehicle_pairs)):
        axs[i, 0].hist(same_feat1, bins=30, color='skyblue', edgecolor='black', alpha=0.5, label='Same Vehicle 1')
        axs[i, 0].hist(same_feat2, bins=30, color='orange', edgecolor='black', alpha=0.5, label='Same Vehicle 2')
        axs[i, 0].set_title(f'Same Vehicle Pair {i+1}', fontsize=15, fontweight='bold')
        axs[i, 0].set_xlabel('HOG Feature Value', fontsize=15, fontweight='bold')
        axs[i, 0].set_ylabel('Frequency', fontsize=15, fontweight='bold')
        axs[i, 0].legend()

        axs[i, 1].hist(diff_feat1, bins=30, color='skyblue', edgecolor='black', alpha=0.5, label='Different Vehicle 1')
        axs[i, 1].hist(diff_feat2, bins=30, color='orange', edgecolor='black', alpha=0.5, label='Different Vehicle 2')
        axs[i, 1].set_title(f'Different Vehicle Pair {i+1}', fontsize=15, fontweight='bold')
        axs[i, 1].set_xlabel('HOG Feature Value', fontsize=15, fontweight='bold')
        axs[i, 1].set_ylabel('Frequency', fontsize=15, fontweight='bold')
        axs[i, 1].legend()

        same_cosine_sim = 1 - cosine(same_feat1, same_feat2)
        diff_cosine_sim = 1 - cosine(diff_feat1, diff_feat2)

        axs[i, 0].text(0.95, 0.95, f'Cosine Similarity: {same_cosine_sim:.2f}', ha='right', va='top', transform=axs[i, 0].transAxes, fontsize=12, fontweight='bold', color='blue')
        axs[i, 1].text(0.95, 0.95, f'Cosine Similarity: {diff_cosine_sim:.2f}', ha='right', va='top', transform=axs[i, 1].transAxes, fontsize=12, fontweight='bold', color='blue')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("hog_features_cosine_similarity_hist.png")
    plt.show()

# Draw histograms and calculate cosine similarity
draw_histograms_and_cosine_similarity(features)
