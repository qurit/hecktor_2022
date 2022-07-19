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
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss



#%%
train_axial_dir = '/data/blobfuse/hecktor2022/resampledCTPTGT/train/axial_data/'
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

# ctfg_paths = sorted(glob.glob(os.path.join(ctfg_dir, '*.nii.gz')))[:10]
# ctbg_paths = sorted(glob.glob(os.path.join(ctbg_dir, '*.nii.gz')))[:100]
# ptfg_paths = sorted(glob.glob(os.path.join(ptfg_dir, '*.nii.gz')))[:10]
# ptbg_paths = sorted(glob.glob(os.path.join(ptbg_dir, '*.nii.gz')))[:100]
# gtfg_paths = sorted(glob.glob(os.path.join(gtfg_dir, '*.nii.gz')))[:10]
# gtbg_paths = sorted(glob.glob(os.path.join(gtbg_dir, '*.nii.gz')))[:100]

ct_paths = ctfg_paths + ctbg_paths
pt_paths = ptfg_paths + ptbg_paths
gt_paths = gtfg_paths + gtbg_paths


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


# %%
inputs_pt_train, inputs_ct_train = inputs_ptct_train[:,0], inputs_ptct_train[:,1]
inputs_pt_valid, inputs_ct_valid = inputs_ptct_valid[:,0], inputs_ptct_valid[:,1]
dataset_train = LesionBackgroundDatasetPTCT(inputs_pt_train,inputs_ct_train, targets_train)#, transform=transforms)
dataset_valid = LesionBackgroundDatasetPTCT(inputs_pt_valid, inputs_ct_valid, targets_valid)#, transform=transforms)
# dataset_train = LesionBackgroundDatasetPTCT(inputs_pt_train[:300],inputs_ct_train[:300], targets_train[:300])#, transform=transforms)
# dataset_valid = LesionBackgroundDatasetPTCT(inputs_pt_valid[:75], inputs_ct_valid[:75], targets_valid[:75])#, transform=transforms)

#%%
dataloader_training = DataLoader(dataset=dataset_train, batch_size=256, shuffle=True)
dataloader_validation = DataLoader(dataset=dataset_valid, batch_size=256, shuffle=True)

#%%
# get model

model = smp.Unet(
    encoder_name="resnet34",        
    encoder_weights="imagenet",     
    in_channels=3,                 
    classes=3,                 
).to(config.DEVICE)
# %%

# %%
criterion = DiceLoss(mode='multiclass')
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

#%%
training_losses, validation_losses, lr_rates = trainer.run_trainer()

# %%
# max_labels = []
# for (x,y) in dataset_train:
#     max_labels.append(np.unique(y.cpu()))
# # %%
# plt.hist(max_labels)
# %%
