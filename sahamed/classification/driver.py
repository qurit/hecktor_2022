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
from transformations import ComposeDouble, FunctionWrapperDouble, resize_2dtensor_bilinear, normalize_01, stack_slices, scale_01, trunc_scale_preprocess
from dataset import LesionBackgroundDataset
from losses import FocalLoss
from trainer import Trainer 





# %%
# data_dir = config.DATA_DIR 

# dlbcl_fg_dir = os.path.join(data_dir, 'DLBCL_bccancer/processed_data/axial/pt_fg')
# dlbcl_bg_dir = os.path.join(data_dir, 'DLBCL_bccancer/processed_data/axial/pt_bg')
# pmbcl_fg_dir = os.path.join(data_dir, 'PMBCL_bccancer/processed_data/axial/pt_fg')
# pmbcl_fg_dir = os.path.join(data_dir, 'PMBCL_bccancer/processed_data/axial/pt_bg')

# dlbcl_fg_dir = '/data/blobfuse/DLBCL_bccancer/processed_data/axial/pt_fg'
# dlbcl_bg_dir = '/data/blobfuse/DLBCL_bccancer/processed_data/axial/pt_bg'
# pmbcl_fg_dir = '/data/blobfuse/PMBCL_bccancer/processed_data/axial/pt_fg'
# pmbcl_bg_dir = '/data/blobfuse/PMBCL_bccancer/processed_data/axial/pt_bg'

# inputs_fg_dlbcl = sorted(glob.glob(os.path.join(dlbcl_fg_dir, '*.nii.gz')))
# inputs_bg_dlbcl = sorted(glob.glob(os.path.join(dlbcl_bg_dir, '*.nii.gz'))) 
# inputs_fg_pmbcl = sorted(glob.glob(os.path.join(pmbcl_fg_dir, '*.nii.gz'))) 
# inputs_bg_pmbcl = sorted(glob.glob(os.path.join(pmbcl_bg_dir, '*.nii.gz'))) 

fg_dir = '/data/blobfuse/bccancer_axial_data/train/pt_fg'
bg_dir = '/data/blobfuse/bccancer_axial_data/train/pt_bg'
inputs_fg = sorted(glob.glob(os.path.join(fg_dir, '*.nii.gz')))
inputs_bg = sorted(glob.glob(os.path.join(bg_dir, '*.nii.gz')))
#%%
# inputs = inputs_fg_dlbcl + inputs_bg_dlbcl + inputs_fg_pmbcl + inputs_bg_pmbcl
inputs = inputs_fg + inputs_bg
targets = []
for path in inputs:
    img_name = os.path.basename(path)[:-7]
    if img_name.endswith('fg'):
        targets.append(1)
    else:
        targets.append(0)


#%%
transforms = ComposeDouble([
    FunctionWrapperDouble(resize_2dtensor_bilinear,
                          input=True,
                          target=False),
    FunctionWrapperDouble(trunc_scale_preprocess,
                          input=True,
                          target=False),
    # FunctionWrapperDouble(scale_01,
    #                       input=True,
    #                       target=False),
    FunctionWrapperDouble(stack_slices,
                          input=True,
                          target=False),  
    FunctionWrapperDouble(np.moveaxis, input=True, target=False, source=-1, destination=0),
    ])


#%%
"""Train-validation splitting
    - will use a stratified splitting here
"""
random_seed = 42
train_size = 0.8
inputs_train, inputs_valid, targets_train, targets_valid = train_test_split(
    inputs, 
    targets, 
    random_state=random_seed,
    shuffle = True,
    train_size=train_size,
    stratify=targets)


#%%
dataset_train = LesionBackgroundDataset(inputs_train, targets_train, transform=transforms)
dataset_valid = LesionBackgroundDataset(inputs_valid, targets_valid, transform=transforms)

#%%
dataloader_training = DataLoader(dataset=dataset_train, batch_size=256, shuffle=True)
dataloader_validation = DataLoader(dataset=dataset_valid, batch_size=256, shuffle=True)

# %%
from models import get_model 
model = get_model().to(config.DEVICE) 
# %%
criterion = FocalLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
# %%
# trainer
trainer = Trainer(model=model,
                  device=config.DEVICE,
                  criterion=criterion,
                  optimizer=optimizer,
                  training_DataLoader=dataloader_training,
                  validation_DataLoader=dataloader_validation,
                  lr_scheduler=None,
                  epochs=100,
                  epoch=0,
                  notebook=False)
                  

# %%
# start training
training_losses, validation_losses, lr_rates = trainer.run_trainer()
# %%
from visual import plot_training

fig = plot_training(training_losses,
                  validation_losses,
                  lr_rates,
                  gaussian=False,
                  sigma=1,
                  figsize=(10, 4)
                  )
# %%
fig.savefig('losses_nofreeze.jpg')