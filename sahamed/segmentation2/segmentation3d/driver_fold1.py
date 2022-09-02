#%%
import config 
import utils
import numpy as np
import os
import glob 
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from trainer import Trainer 
from dataset_3d import Segmentation3DDatasetPTCT_ptpreprocess1
import torch.nn as nn
from monai.networks.nets import UNet
from monai.losses import *
from utils import get_required_paths_3d
# %%
FOLD = 1

imagesTr_dir = '/data/blobfuse/hecktor2022_training/hecktor2022/imagesTr_resampled'
labelsTr_dir = '/data/blobfuse/hecktor2022_training/hecktor2022/labelsTr_resampled'

ct_paths = sorted(glob.glob(os.path.join(imagesTr_dir, '*CT.nii.gz')))
pt_paths = sorted(glob.glob(os.path.join(imagesTr_dir, '*PT.nii.gz')))
gt_paths = sorted(glob.glob(os.path.join(labelsTr_dir, '*.nii.gz')))

train_folds_fname = '/home/shadab/Projects/hecktor_2022/sahamed/data_preprocessing/train_5folds.csv'
valid_folds_fname = '/home/shadab/Projects/hecktor_2022/sahamed/data_preprocessing/valid_5folds.csv'
train_folds_df = pd.read_csv(train_folds_fname)
valid_folds_df = pd.read_csv(valid_folds_fname)

train_ids = train_folds_df['fold'+str(FOLD)].values
valid_ids = valid_folds_df['fold'+str(FOLD)].values

ct_paths_train = get_required_paths_3d(ct_paths, train_ids, 'CT')
ct_paths_valid = get_required_paths_3d(ct_paths, valid_ids, 'CT')
pt_paths_train = get_required_paths_3d(pt_paths, train_ids, 'PT')
pt_paths_valid = get_required_paths_3d(pt_paths, valid_ids, 'PT')
gt_paths_train = get_required_paths_3d(gt_paths, train_ids, 'GT')
gt_paths_valid = get_required_paths_3d(gt_paths, valid_ids, 'GT')



#%%

dataset_train = Segmentation3DDatasetPTCT_ptpreprocess1(pt_paths_train,ct_paths_train,gt_paths_train)
dataset_valid = Segmentation3DDatasetPTCT_ptpreprocess1(pt_paths_valid,ct_paths_valid,gt_paths_valid)
#%%
dataloader_train = DataLoader(dataset=dataset_train, batch_size=4, shuffle=True, pin_memory=True, num_workers=24)
dataloader_valid = DataLoader(dataset=dataset_valid, batch_size=4, shuffle=True, pin_memory=True, num_workers=24)


# %%
#UNet small
model = UNet(spatial_dims=3,\
            in_channels=2,\
            out_channels=3,\
            channels=(16, 32, 64, 128, 256),\
            strides=(2, 2, 2, 2)).to(config.MAIN_DEVICE)
model = nn.DataParallel(model, device_ids=[0,1,2,3])

#%%
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=3e-5, amsgrad=True)
# criterion = DiceCELoss(lambda_ce=100., ce_weight=torch.tensor([1.,100.,100.]).to(config.MAIN_DEVICE))
criterion = DiceLoss().to(config.MAIN_DEVICE)
scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)


# %%
trainer = Trainer(model=model,
                  device=config.MAIN_DEVICE,
                  criterion=criterion,
                  optimizer=optimizer,
                  training_DataLoader=dataloader_train,
                  validation_DataLoader=dataloader_valid,
                  lr_scheduler=scheduler,
                  epochs=400,
                  epoch=0,
                  notebook=False)
# %%
import time
start = time.time()
training_losses, validation_losses, lr_rates = trainer.run_trainer()
# %%
el = time.time() - start 
print(el)
# %%
