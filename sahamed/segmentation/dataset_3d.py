#%%
import config 
import utils
import numpy as np
import os
from glob import glob
import pandas as pd
import torch
import torchvision
import matplotlib.pyplot as plt
import torchsummary as summary
from torch.utils.data import Dataset, DataLoader
from random import shuffle 
import nibabel as nib
import cv2
import glob
import torch.nn.functional as F
from transformations import resize_3dtensor_bilinear,resize_3dtensor_nearest, trunc_scale_preprocess_pt, trunc_scale_preprocess_ct, pt_preprocess 
# from transformations import *
# %%
# dataset class

class Segmentation3DDataset(Dataset):
    def __init__(self, inputs_pt, inputs_ct, targets):
        self.inputs_pt = inputs_pt 
        self.inputs_ct = inputs_ct 
        self.targets = targets
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.int64

    
    def __len__(self):
        return len(self.inputs_pt)
    
    def __getitem__(self, index):
        
        ptimg_path = self.inputs_pt[index]
        ctimg_path = self.inputs_ct[index]
        gtimg_path = self.targets[index]
        
        
        xpt, _ = utils.nib2numpy(ptimg_path)  # get image 3D np array
        xct, _ = utils.nib2numpy(ctimg_path)
        y, _ = utils.nib2numpy(gtimg_path)

        '''
        Transformations:
        - Resize
        - trunc_normalize (Ivan/Yixi preprocess)
        - Stack 3 slices to make 3 channel image (first two channels PT, 3rd channel CT)
        - permute axes to get first axes as channel
        '''
        # print('PT max before resize:', np.max(xpt))
        # print('CT max before resize:', np.max(xct))
        # print("GT uniques:", np.unique(y, return_counts=True))
        # fig1, ax1 = plt.subplots(1,2, figsize=(15,8))
        # fig1.patch.set_facecolor('white')
        # fig1.patch.set_alpha(0.7)

        # ax1[0].hist(xct.flatten())d
        # ax1[0].set_title('CT pixel histogram before resize')
        # ax1[1].imshow(xct)
        
        # fig2, ax2 = plt.subplots(1,2, figsize=(15,8))
        # fig2.patch.set_facecolor('white')
        # fig2.patch.set_alpha(0.7)
        # Resizing
        xct = resize_3dtensor_bilinear(xct, 128, 128, 128)
        xpt = resize_3dtensor_bilinear(xpt, 128, 128, 128)
        y = resize_3dtensor_nearest(y, 128, 128, 128)
        # print('PT max after resize:', np.max(xpt))
        # print('CT max after resize:',np.max(xct))
        # print("GT uniques:", np.unique(y, return_counts=True))

        # ax2[0].hist(xct.flatten())
        # ax2[0].set_title('CT pixel histogram after resize')
        # ax2[1].imshow(xct)
        # scaling PT and CT intensities

        # fig3, ax3 = plt.subplots(1,2, figsize=(15,8))
        # fig3.patch.set_facecolor('white')
        # fig3.patch.set_alpha(0.7)

        xpt = trunc_scale_preprocess_pt(xpt)
        xct = trunc_scale_preprocess_ct(xct) 

        # print('PT max after scaling:', np.max(xpt))
        # print('CT max after scaling:', np.max(xct))
        # ax3[0].hist(xct.flatten())
        # ax3[0].set_title('CT pixel histogram after resize+rescale')
        # ax3[1].imshow(xct)
        # plt.show()
        # stack in 2 channels
        x = np.stack((xpt, xct), axis=0)
        
        # make first axes as number of channels
        # x = np.moveaxis(x, source=-1, destination=0)
        
        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)
        y = F.one_hot(y, num_classes=3)
        y = torch.moveaxis(y, source=-1, destination=0)
        # y = torch.unsqueeze(y, axis=0)
        return x, y

