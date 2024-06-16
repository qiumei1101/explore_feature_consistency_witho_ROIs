#images in ROI
#images outside ROI
cosine_similarity_inROI =[]
cosine_similarity_outROI=[]
cosine_similarity_inoutROI=[]


import torch
import torchvision.models as models
import PIL.Image as Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torch.nn import Sequential

model_ft = models.resnet50(pretrained=True)
### strip the last layer
feature_extractor: Sequential = torch.nn.Sequential(*list(model_ft.children())[:-1])
from pathlib import Path
import numpy as np
### check this works
# x = torch.randn([1,3,224,314])
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

    feat = feature_extractor(data).detach().numpy()
    print("feat", feat.size)

    features.append(feat)
