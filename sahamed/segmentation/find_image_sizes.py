#%%
import config 
import utils
import numpy as np
import os
from glob import glob
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
import glob
from sklearn.model_selection import train_test_split
from trainer import Trainer 
from dataset_ptct import LesionBackgroundDatasetPTCT
from dataset_3d import Segmentation3DDataset
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss

#%%
train_dir = '/data/blobfuse/hecktor2022/resampledCTPTGT/train/'
pt_dir = os.path.join(train_dir, 'pt')
ct_dir = os.path.join(train_dir, 'ct')
gt_dir = os.path.join(train_dir, 'gt')

pt_paths = sorted(glob.glob(os.path.join(pt_dir, '*.nii.gz')))
ct_paths = sorted(glob.glob(os.path.join(ct_dir, '*.nii.gz')))
gt_paths = sorted(glob.glob(os.path.join(gt_dir, '*.nii.gz')))

# %%
z_shape = []

for i in range(len(pt_paths)):
    pt, _ = utils.nib2numpy(pt_paths[i])
    z_shape.append(pt.shape[2])
    print(i)


# %%
