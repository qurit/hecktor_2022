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
# from transformations import *
# %%
# dataset class

class LesionBackgroundDataset(Dataset):
    def __init__(self, inputs, targets, transform = None):
        # fg_slice_folder = sorted(glob(os.path.join(folder + 'pt_fg/*.nii.gz')))
        # bg_slice_folder = sorted(glob(os.path.join(folder + 'pt_bg/*.nii.gz')))
        self.inputs = inputs 
        self.targets = targets
        self.transform = transform 
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.float32
        # self.fpaths = fg_slice_folder + bg_slice_folder
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],               
            std=[0.229, 0.224, 0.225])

    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        img_path = self.inputs[index]
    
        x, _ = utils.nib2numpy(img_path) # get image 2D np array
        y = self.targets[index]
        '''
        Transformations:
        - Resize
        - Normalize_01
        - Stack 3 slices to make 3 channel image
        - permute axes to get first axes as channel
        '''

        '''normalize_pretrain outside of transformation
        (I will do this normalization in the later runs)
        ''' 

        if self.transform is not None:
            x, y = self.transform(x, y)
        
        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.tensor([y]).type(self.targets_dtype)
      
        return x.to(config.DEVICE), y.to(config.DEVICE)




