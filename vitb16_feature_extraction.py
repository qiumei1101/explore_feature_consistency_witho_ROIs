#images in ROI
#images outside ROI
cosine_similarity_inROI =[]
cosine_similarity_outROI=[]
cosine_similarity_inoutROI=[]
from transformers import ViTModel, ViTConfig
import torchvision.transforms as T
from pathlib import Path
import numpy as np
import PIL.Image as Image

# Load the configuration and create an instance of ViT
config = ViTConfig.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTModel(config)

# Load the pretrained ViT-B16 model
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

# Function to extract features
def extract_features(model, input_ids):
    # Forward pass, extract hidden states before the classifier
    outputs = model(input_ids, output_hidden_states=True)
    # Get the last hidden state (features before the classification head)
    last_hidden_state = outputs.last_hidden_state
    # Pool the outputs into a single batch wise vector (CLS token)
    pooled_output = last_hidden_state[:, 0]
    return pooled_output

# Example usage:
import torch

# Dummy input resembling image patches tokenized (batch size, sequence length, hidden dimension)
# Note: Actual input should be tokenized appropriately for ViT

process=T.Compose([
    T.Resize([224,224]),
    T.ToTensor(),
    T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

img_type = 'RGB'
# save_path = "/home/meiqiu@ads.iu.edu/test/Data_VeReID/Weaving_1_exit_feature/"
file_name_list=[]
features = []
file_paths = ['/home/meiqiu@ads.iu.edu/Mei_all/Data_VeReID/VeRi/small_veri/0174_c006_00043265_0.jpg','/home/meiqiu@ads.iu.edu/Mei_all/Data_VeReID/VeRi/small_veri/0196_c011_00078460_0.jpg',
              '/home/meiqiu@ads.iu.edu/Mei_all/Data_VeReID/VeRi/small_veri/0174_c004_00072270_0.jpg','/home/meiqiu@ads.iu.edu/Mei_all/Data_VeReID/VeRi/small_veri/0196_c011_00078475_0.jpg']
for img in file_paths:
    file_name_list.append(Path(img).stem)
    # save_file_name = os.path.join(save_path,Path(img).stem)
    img = Image.open(img).convert(img_type)

    resized_img = process(img)
    print("type",type(resized_img))
    resized_img= torch.from_numpy(np.array(resized_img))

    resized_img=resized_img[None,]
    print("shape",resized_img.shape[:4])
    data = resized_img
    # feat = model(data).detach().cpu()

    feat = extract_features(model,data).detach().numpy()
    print("feat", feat.size)

    features.append(feat)