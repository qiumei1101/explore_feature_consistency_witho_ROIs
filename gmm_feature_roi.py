import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

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

def extract_features(image, model, preprocess):
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

    with torch.no_grad():
        features = model(input_batch)
    
    return features.squeeze().numpy()

def get_roi_features(image, rois, model, preprocess):
    inside_features = []
    outside_features = []

    for roi in rois:
        x, y, w, h = roi
        # Extract inside ROI
        roi_image = image.crop((x, y, x+w, y+h))
        inside_features.append(extract_features(roi_image, model, preprocess))

        # Create a mask for the ROI
        mask = np.zeros(image.size[::-1], dtype=np.uint8)
        mask[y:y+h, x:x+w] = 1

        # Extract outside ROI by inverting the mask
        masked_image = np.array(image) * (1 - mask[:, :, np.newaxis])
        outside_image = Image.fromarray(masked_image)
        outside_features.append(extract_features(outside_image, model, preprocess))

    return np.array(inside_features), np.array(outside_features)

# Example usage
image_path = 'path_to_your_image.jpg'
image = Image.open(image_path).convert('RGB')
rois = [(50, 50, 100, 100), (150, 150, 100, 100)]  # Example ROIs

inside_features, outside_features = get_roi_features(image, rois, model, preprocess)

from sklearn.mixture import GaussianMixture

def fit_gmm(features, n_components):
    gmm = GaussianMixture(n_components=n_components, covariance_type='full')
    gmm.fit(features)
    return gmm

n_components = 2  # Choose the number of components

inside_gmm = fit_gmm(inside_features, n_components)
outside_gmm = fit_gmm(outside_features, n_components)

from sklearn.mixture import GaussianMixture

def fit_gmm(features, n_components):
    gmm = GaussianMixture(n_components=n_components, covariance_type='full')
    gmm.fit(features)
    return gmm

n_components = 2  # Choose the number of components

inside_gmm = fit_gmm(inside_features, n_components)
outside_gmm = fit_gmm(outside_features, n_components)

def compare_gmms(inside_gmm, outside_gmm):
    inside_means = inside_gmm.means_
    outside_means = outside_gmm.means_

    inside_covariances = inside_gmm.covariances_
    outside_covariances = outside_gmm.covariances_

    print("Inside ROI GMM Means:\n", inside_means)
    print("Outside ROI GMM Means:\n", outside_means)

    print("\nInside ROI GMM Covariances:\n", inside_covariances)
    print("Outside ROI GMM Covariances:\n", outside_covariances)

compare_gmms(inside_gmm, outside_gmm)
