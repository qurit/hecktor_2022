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
import SimpleITK as sitk
import glob
import torch.nn.functional as F
from transformations import resize_3dtensor_bilinear,resize_3dtensor_nearest, pt_preprocess_1, pt_preprocess_2, ct_preprocess_1 
from utils import get_required_paths_3d
# from transformations import *
# %%
# dataset class

class Segmentation3DDatasetPTCT_ptpreprocess1(Dataset):
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
        
        xpt = sitk.ReadImage(ptimg_path)  # get image 3D np array
        xct = sitk.ReadImage(ctimg_path)
        y = sitk.ReadImage(gtimg_path)

        
        
        xpt_array = sitk.GetArrayFromImage(xpt)
        xct_array = sitk.GetArrayFromImage(xct)
        y_array = sitk.GetArrayFromImage(y)
        # print("Before resize:", np.unique(y_array))
        
        
        xpt_array = resize_3dtensor_bilinear(xpt_array, 128, 128, 128)
        xct_array = resize_3dtensor_bilinear(xct_array, 128, 128, 128)
        y_array = resize_3dtensor_nearest(y_array, 128, 128, 128)
        # print("After resize:", np.unique(y_array))
        # y2 = y_array.copy()

        xpt_array = pt_preprocess_1(xpt_array)
        xct_array = ct_preprocess_1(xct_array) 

        x = np.stack((xpt_array, xct_array), axis=0)
        y = y_array 
        
        x = torch.from_numpy(x).type(self.inputs_dtype)
        y = torch.from_numpy(y).type(self.targets_dtype)
        
        y = F.one_hot(y, num_classes=3)
        y = torch.moveaxis(y, source=-1, destination=0)

        # y_copy = y.clone().detach()
        # y_copy = torch.squeeze(y_copy, axis=0)
        # y_copy = torch.argmax(y_copy, axis=0)
        # print("From OHE:", np.unique(y_copy))
        # y3 = y.clone().detach()
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

        

        # print('PT max after scaling:', np.max(xpt))
        # print('CT max after scaling:', np.max(xct))
        # ax3[0].hist(xct.flatten())
        # ax3[0].set_title('CT pixel histogram after resize+rescale')
        # ax3[1].imshow(xct)
        # plt.show()
        # stack in 2 channels
        
        
        return x, y


# #%%
# FOLD = 0

# imagesTr_dir = '/data/blobfuse/hecktor2022_training/hecktor2022/imagesTr_resampled'
# labelsTr_dir = '/data/blobfuse/hecktor2022_training/hecktor2022/labelsTr_resampled'

# ct_paths = sorted(glob.glob(os.path.join(imagesTr_dir, '*CT.nii.gz')))
# pt_paths = sorted(glob.glob(os.path.join(imagesTr_dir, '*PT.nii.gz')))
# gt_paths = sorted(glob.glob(os.path.join(labelsTr_dir, '*.nii.gz')))

# train_folds_fname = '/home/shadab/Projects/hecktor_2022/sahamed/data_preprocessing/train_5folds.csv'
# valid_folds_fname = '/home/shadab/Projects/hecktor_2022/sahamed/data_preprocessing/valid_5folds.csv'
# train_folds_df = pd.read_csv(train_folds_fname)
# valid_folds_df = pd.read_csv(valid_folds_fname)

# train_ids = train_folds_df['fold'+str(FOLD)].values
# valid_ids = valid_folds_df['fold'+str(FOLD)].values

# ct_paths_train = get_required_paths_3d(ct_paths, train_ids, 'CT')
# ct_paths_valid = get_required_paths_3d(ct_paths, valid_ids, 'CT')
# pt_paths_train = get_required_paths_3d(pt_paths, train_ids, 'PT')
# pt_paths_valid = get_required_paths_3d(pt_paths, valid_ids, 'PT')
# gt_paths_train = get_required_paths_3d(gt_paths, train_ids, 'GT')
# gt_paths_valid = get_required_paths_3d(gt_paths, valid_ids, 'GT')



# #%%

# dataset_train = Segmentation3DDatasetPTCT_ptpreprocess1(pt_paths_train,ct_paths_train,gt_paths_train)
# dataset_valid = Segmentation3DDatasetPTCT_ptpreprocess1(pt_paths_valid,ct_paths_valid,gt_paths_valid)
# #%%
# dataloader_train = DataLoader(dataset=dataset_train, batch_size=4, shuffle=True, pin_memory=True, num_workers=24)
# dataloader_valid = DataLoader(dataset=dataset_valid, batch_size=1, shuffle=True, pin_memory=True, num_workers=24)

# # %%
# for (x,y) in dataset_train:
#     print('done')
# # %%
