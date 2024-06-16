import os     
import argparse
from pathlib import Path
from datetime import datetime
import glob
import cv2
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
import shutil
from scipy import spatial
import matplotlib.pyplot as plt
import torch


# Check if GPU is available and set the device accordingly
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-d', type=str, required=True, help='Input image data folder')
   
    # parser.add_argument('--output-path', '-o', type=str, default='./logs', help='Ouptut Directory (default=./logs/YYYYMMDD-HHMMSS)')
    args = parser.parse_args()

    assert(os.path.exists(args.data_dir))
    return args

image_path_list_inroi = []
image_path_list_outroi = []
objid_inroi = []
objid_outroi = []
image_inroi = []
image_outroi = []
root = '/home/meiqiu@ads.iu.edu/Mei_all/Data_VeReID/tracking_images_inroi_classfolder'
if __name__ == '__main__':
    input_setting = get_args()
    print('input_setting',input_setting.data_dir.split('/'))
    class_name = input_setting.data_dir.split('/')[-1]
    
    if not os.path.exists(os.path.join(root,class_name+'/inroi')):
        os.makedirs(os.path.join(root,class_name+'/inroi'))
    if not os.path.exists(os.path.join(root,class_name+'/outroi')):
        os.makedirs(os.path.join(root,class_name+'/outroi'))
    
    

    for img in glob.glob(input_setting.data_dir+'/*.jpg'):
        name_ = Path(img).stem
        fid = int(name_.split('_')[0])
        if fid%5==0:
            # print(list(name_.split('_')))
            if " 'car'" in list(name_.split('_')) or " 'truck'" in list(name_.split('_')):
                if ' True' in list(name_.split('_')):
                    # print(img)
                    image_path_list_inroi.append(img)
                    shutil.copy(img, os.path.join(root,class_name+'/inroi'))

                    objid_inroi.append(int(name_.split('_')[1]))
                    image_inroi.append(Image.open(img))
                else:
                    image_path_list_outroi.append(img)
                    shutil.copy(img, os.path.join(root,class_name+'/outroi'))

                    objid_outroi.append(int(name_.split('_')[1]))
                    image_outroi.append(Image.open(img))



    print(len(image_path_list_inroi))    
    print(len(objid_inroi))
    print(len(image_path_list_outroi))
    print(len(objid_outroi))

    # Load the pre-trained ResNet-50 model
    resnet50 = models.resnet50(pretrained=True).to(device)

    # Remove the final fully connected layer to use the model as a feature extractor
    resnet50.fc = torch.nn.Identity()

    # Define transformations for the images
    transform = transforms.Compose([
        transforms.Resize(256),              # Resize images to 256x256
        transforms.CenterCrop(224),          # Crop a 224x224 patch from the center
        transforms.ToTensor(),               # Convert images to tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for ResNet
    ])

    # Dummy dataset - replace 'path_to_images' with the path to your dataset
    # print("image_path_list_inroi",image_path_list_inroi)
    dataset = ImageFolder(root=os.path.join(root,class_name), transform=transform)
    print("dataset",dataset)
    # DataLoader for handling large batches of images
    batch_size = 256  # Adjust this size based on your system's memory capacity
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Function to extract features
  # Function to extract features
    def extract_features(model, dataloader):
        model.eval()  # Set model to evaluation mode
        features = []
        with torch.no_grad():  # Disable gradient computation
            for inputs, _ in dataloader:
                outputs = model(inputs.to(device))  # Forward pass: compute the output of the modified ResNet-50
                features.append(outputs)
                print("done")
        return torch.cat(features, dim=0)  # Concatenate all features from all batches
    
    # Extract features
    features = extract_features(resnet50, dataloader)
    print(features.shape)  # Shape will be (number of images, 2048)
    inroi_features = features[:len(image_path_list_inroi)]
    outroi_features = features[len(image_path_list_inroi):]
    print(inroi_features.shape)  # Shape will be (number of images, 2048)
    def conditional_pairwise_cosine_similarity(matrix_a, matrix_b,image_path_list1,image_path_list2):
        # Normalize the rows of both matrices
        norm_a = matrix_a / matrix_a.norm(dim=1, keepdim=True)
        norm_b = matrix_b / matrix_b.norm(dim=1, keepdim=True)
        
        # Compute cosine similarity matrix between A and B
        sim_matrix = torch.mm(norm_a, norm_b.t())
        
        # Generate a mask for condition i < j
        num_rows_a, num_cols_b = matrix_a.size(0), matrix_b.size(0)
        mask = torch.ones((num_rows_a, num_cols_b), dtype=torch.bool)
        for i in range(num_rows_a):
            name_1 = Path(image_path_list1[i]).stem
            objid1 = int(name_1.split('_')[1])

            for j in range(num_cols_b):
                name_2 = Path(image_path_list2[j]).stem
                objid2 = int(name_2.split('_')[1])

                if name_1 == name_2 or objid1!=objid2:
                    mask[i, j] = False
        # Filter the similarity values based on the mask
        filtered_sim_values = sim_matrix[mask]
        
        return filtered_sim_values


    similarities = conditional_pairwise_cosine_similarity(inroi_features, inroi_features,image_path_list_inroi,image_path_list_inroi)
    print("Filtered pairwise cosine similarities (i < j):")
    print("mean",torch.mean(similarities))
    print("std",torch.std(similarities))

    # plt.hist(similarities.detach().cpu().numpy())
    # plt.savefig("night_2_coss_inroi.png")
    

    # def pairwise_cosine_similarity(matrix):
    #     # Normalize the rows of the matrix
    #     norm_matrix = matrix / matrix.norm(dim=1, keepdim=True)
        
    #     # Compute cosine similarity matrix
    #     sim_matrix = torch.mm(norm_matrix, norm_matrix.t())
        
    #     # Apply the condition i < j by masking the upper triangle of the similarity matrix
    #     # Including the diagonal where i == j
    #     upper_tri_mask = torch.triu(torch.ones(sim_matrix.size(), dtype=torch.bool), diagonal=1)
        
    #     # Filter the similarity values based on the mask
    #     filtered_sim_values = sim_matrix[upper_tri_mask]
        
    #     return filtered_sim_values
    # similarities = pairwise_cosine_similarity(inroi_features)
    # print("Filtered pairwise cosine similarities (i < j):")

    # inroi_cs = []
    # for img1 in image_path_list_inroi:
    #     name_1 = Path(img1).stem
    #     objid1 = int(name_1.split('_')[1])
    #     for img2 in image_path_list_inroi:
    #         name_2 = Path(img2).stem
    #         objid2 = int(name_2.split('_')[1])
    #         if img1!=img2 and objid1==objid2:
    #             result = 1 - spatial.distance.cosine(inroi_features.cpu()[image_path_list_inroi.index(img1)], inroi_features.cpu()[image_path_list_inroi.index(img2)])
    #             inroi_cs.append(result)
    # plt.hist(similarities.cpu())

    # plt.savefig("coss_inroi.png")
    

