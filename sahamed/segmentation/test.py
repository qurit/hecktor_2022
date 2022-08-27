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
# from dataset_2d import Segmentation2DDataset, Segmentation2DDataset3ChannelInput
import torch.nn as nn
from monai.networks.nets import UNet
from monai.transforms import *
from monai.losses import *
#%%

train_axial_dir = '/data/blobfuse/hecktor2022/resampledCTPTGT/test/axial_data/'
ctfg_dir = os.path.join(train_axial_dir, 'ct_fg')
ctbg_dir = os.path.join(train_axial_dir, 'ct_bg')
ptfg_dir = os.path.join(train_axial_dir, 'pt_fg')
ptbg_dir = os.path.join(train_axial_dir, 'pt_bg')
gtfg_dir = os.path.join(train_axial_dir, 'gt_fg')
gtbg_dir = os.path.join(train_axial_dir, 'gt_bg')

ctfg_paths = sorted(glob.glob(os.path.join(ctfg_dir, '*.nii.gz')))
ctbg_paths = sorted(glob.glob(os.path.join(ctbg_dir, '*.nii.gz')))
ptfg_paths = sorted(glob.glob(os.path.join(ptfg_dir, '*.nii.gz')))
ptbg_paths = sorted(glob.glob(os.path.join(ptbg_dir, '*.nii.gz')))
gtfg_paths = sorted(glob.glob(os.path.join(gtfg_dir, '*.nii.gz')))
gtbg_paths = sorted(glob.glob(os.path.join(gtbg_dir, '*.nii.gz')))


#%%
ct_paths = ctfg_paths + ctbg_paths
pt_paths = ptfg_paths + ptbg_paths
gt_paths = gtfg_paths + gtbg_paths


#%%
random_seed = 42
train_size = 0.8

inputs_ptct = np.column_stack((pt_paths, ct_paths))
targets = gt_paths
inputs_ptct_train, inputs_ptct_valid, targets_train, targets_valid = train_test_split(
    inputs_ptct, 
    targets, 
    random_state=random_seed,
    shuffle = True,
    train_size=train_size,
    )
# %%
