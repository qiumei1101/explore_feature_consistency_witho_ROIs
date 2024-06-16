import os     
import argparse
from pathlib import Path
from datetime import datetime
import glob
import cv2
from PIL import Image
import numpy as np
import shutil
from scipy import spatial
import matplotlib.pyplot as plt
import torch

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-d', type=str, required=True, help='Input image data folder')
   
    # parser.add_argument('--output-path', '-o', type=str, default='./logs', help='Ouptut Directory (default=./logs/YYYYMMDD-HHMMSS)')
    args = parser.parse_args()

    assert(os.path.exists(args.data_dir))
    return args

# image_path_list_inroi = []
# image_path_list_outroi = []
# objid_inroi = []
# objid_outroi = []
# image_inroi = []
# image_outroi = []
# root = '/home/meiqiu@ads.iu.edu/Mei_all/Data_VeReID/tracking_images_inroi_classfolder'

if __name__ == '__main__':
    input_setting = get_args()
    print('input_setting',input_setting.data_dir.split('/'))
    # class_name = input_setting.data_dir.split('/')[-1]
    
    # if not os.path.exists(os.path.join(root,class_name+'/inroi')):
    #     os.makedirs(os.path.join(root,class_name+'/inroi'))
    # if not os.path.exists(os.path.join(root,class_name+'/outroi')):
    #     os.makedirs(os.path.join(root,class_name+'/outroi'))
    
    objid_list = []

    for img in glob.glob(input_setting.data_dir+'/*.jpg'):
        name_ = Path(img).stem
        objid = int(name_.split('_')[1])
        objid_list.append(objid)
        if not os.path.exists(os.path.join(input_setting.data_dir,str(objid))):
             os.makedirs(os.path.join(input_setting.data_dir,str(objid)))
             shutil.copy(img, os.path.join(input_setting.data_dir,str(objid)))
        else:
            shutil.copy(img, os.path.join(input_setting.data_dir,str(objid)))

