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
from dataset_3d import Segmentation3DDataset
import torch.nn as nn
from monai.networks.nets import UNet
from monai.transforms import *
from monai.losses import *
from sklearn.preprocessing import OneHotEncoder

# %%

train_dir = '/data/blobfuse/hecktor2022/resampledCTPTGT/train/'
pt_dir = os.path.join(train_dir, 'pt')
ct_dir = os.path.join(train_dir, 'ct')
gt_dir = os.path.join(train_dir, 'gt')

pt_paths = sorted(glob.glob(os.path.join(pt_dir, '*.nii.gz')))
ct_paths = sorted(glob.glob(os.path.join(ct_dir, '*.nii.gz')))
gt_paths = sorted(glob.glob(os.path.join(gt_dir, '*.nii.gz')))


# %%
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

inputs_pt_train, inputs_ct_train = inputs_ptct_train[:,0], inputs_ptct_train[:,1]
inputs_pt_valid, inputs_ct_valid = inputs_ptct_valid[:,0], inputs_ptct_valid[:,1]
dataset_train = Segmentation3DDataset(inputs_pt_train,inputs_ct_train, targets_train)#, transform=transforms)
dataset_valid = Segmentation3DDataset(inputs_pt_valid, inputs_ct_valid, targets_valid)#, transform=transforms)
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
# optimal_model_name = 'segmentation_ep=100.pth'
# optimal_model_dir = '/data/blobfuse/saved_models_hecktor/segmentation/saved_models_unet3dsmallmonai_generalizeddiceloss'
# optimal_model_path = os.path.join(optimal_model_dir, optimal_model_name)
# model.load_state_dict(torch.load(optimal_model_path))
#%%
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = GeneralizedDiceLoss(include_background=False,\
                                to_onehot_y=False,\
                                sigmoid=True).to(config.MAIN_DEVICE)

# %%
trainer = Trainer(model=model,
                  device=config.MAIN_DEVICE,
                  criterion=criterion,
                  optimizer=optimizer,
                  training_DataLoader=dataloader_train,
                  validation_DataLoader=dataloader_valid,
                  lr_scheduler=None,
                  epochs=200,
                  epoch=0,
                  notebook=False)
# %%
import time
start = time.time()
training_losses, validation_losses, lr_rates = trainer.run_trainer()
# %%
el = time.time() - start 
print(el)