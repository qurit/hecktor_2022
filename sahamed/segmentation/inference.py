#%%
from sklearn.metrics import jaccard_score
import torch
from torch.utils import data
import utils as utils
import glob
import config
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
from trainer import Trainer
import matplotlib.pyplot as plt
from torchsummary import summary
from dataset_ptct import LesionBackgroundDatasetPTCT
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
import torch.nn as nn
#%%
train_axial_dir = '/data/blobfuse/hecktor2022/resampledCTPTGT/test/axial_data/'
ctfg_dir = os.path.join(train_axial_dir, 'ct_fg')
ctbg_dir = os.path.join(train_axial_dir, 'ct_bg')
ptfg_dir = os.path.join(train_axial_dir, 'pt_fg')
ptbg_dir = os.path.join(train_axial_dir, 'pt_bg')
gtfg_dir = os.path.join(train_axial_dir, 'gt_fg')
gtbg_dir = os.path.join(train_axial_dir, 'gt_bg')
# testing pushing from another location
# ctfg_paths = sorted(glob.glob(os.path.join(ctfg_dir, '*.nii.gz')))
# ctbg_paths = sorted(glob.glob(os.path.join(ctbg_dir, '*.nii.gz')))
# ptfg_paths = sorted(glob.glob(os.path.join(ptfg_dir, '*.nii.gz')))
# ptbg_paths = sorted(glob.glob(os.path.join(ptbg_dir, '*.nii.gz')))
# gtfg_paths = sorted(glob.glob(os.path.join(gtfg_dir, '*.nii.gz')))
# gtbg_paths = sorted(glob.glob(os.path.join(gtbg_dir, '*.nii.gz')))

ctfg_paths = sorted(glob.glob(os.path.join(ctfg_dir, '*.nii.gz')))
ctbg_paths = sorted(glob.glob(os.path.join(ctbg_dir, '*.nii.gz')))
ptfg_paths = sorted(glob.glob(os.path.join(ptfg_dir, '*.nii.gz')))
ptbg_paths = sorted(glob.glob(os.path.join(ptbg_dir, '*.nii.gz')))
gtfg_paths = sorted(glob.glob(os.path.join(gtfg_dir, '*.nii.gz')))
gtbg_paths = sorted(glob.glob(os.path.join(gtbg_dir, '*.nii.gz')))

ct_paths = ctfg_paths + ctbg_paths
pt_paths = ptfg_paths + ptbg_paths
gt_paths = gtfg_paths + gtbg_paths

# inputs_ptct = np.column_stack((pt_paths, ct_paths))
inputs_ct = ct_paths
inputs_pt = pt_paths
targets = gt_paths
#%%
dataset_test = LesionBackgroundDatasetPTCT(inputs_pt,inputs_ct, targets) 
                                   

dataloader_test = data.DataLoader(dataset=dataset_test, batch_size=1, shuffle=False, pin_memory=True, num_workers=24)

ImageIDs = []
for path in targets:
    imageid = os.path.basename(path)[:-7]
    ImageIDs.append(imageid)


#%%

model = smp.Unet(
    encoder_name="resnet34",        
    encoder_weights="imagenet",     
    in_channels=3,                 
    classes=3,                 
).to(config.MAIN_DEVICE)
model = nn.DataParallel(model, device_ids=[0,1,2,3])

#%%
optimal_model_name = 'segmentation_ep=170.pth'
saved_model_path = os.path.join(config.SAVE_MODEL_DIR, optimal_model_name)
model.load_state_dict(torch.load(saved_model_path))

#%%
# def dice_score(prediction, target, k):
#     eps = 1e-4
#     dsc = np.sum(prediction[target==k])*2.0 / (np.sum(prediction) + np.sum(target) + eps)
#     return dsc

# def dice_score_multilabel(prediction, target):
#     target_labels = np.unique(target)

#     target_labels = target_labels[target_labels != 0]
#     if len(target_labels)
#     # prediction_labels = np.unique(prediction) 
#     dsc_multilabel = 0
#     for k in target_labels:
#         dsc_multilabel += dice_score(prediction, target, k)
    
#     dsc_multilabel /= len(target_labels)
#     return dsc_multilabel

#%%
def compute_intersection(prediction, target, label):
    intersection = np.sum(prediction[target==label]==label)
    return intersection

def compute_union(prediction, target, label):
    union = np.sum(prediction[prediction==label]==label) + np.sum(target[target==label]==label) 
    return union
#%%
# postprocess function
def postprocess_prediction(img: torch.tensor):
    img = torch.argmax(img, dim=1)  # perform argmax to generate 1 channel
    img = img.cpu().numpy()  # send to cpu and transform to numpy.ndarray
    img = np.squeeze(img)  # remove batch dim and channel dim -> [H, W]
    # img = re_normalize(img)  # scale it to the range [0-255]
    return img
# postprocess function
def postprocess_target(img: torch.tensor):
    # perform argmax to generate 1 channel
    img = img.cpu().numpy()  # send to cpu and transform to numpy.ndarray
    img = np.squeeze(img)  # remove batch dim and channel dim -> [H, W]
    # img = re_normalize(img)  # scale it to the range [0-255]
    return img

#%%
def predict(img,
            tar,
            model,
            postprocess_prediction,
            postprocess_target,
            device,
            ):
    model.eval()
    
    x = img.to(device)  # to torch, send to device
    with torch.no_grad():
        out = model(x)  # send through model/network

    out_softmax = torch.softmax(out, dim=1)  # perform softmax on outputs
    prediction = postprocess_prediction(out_softmax)  # postprocess outputs
    postprocess_tar = postprocess_target(tar)
    return prediction, postprocess_tar


#%%
intersections_1 = []
unions_1 = []
intersections_2 = []
unions_2 = []
# HausdorffDistance = []
# jaccard_score = []
#%%
count = 0
for i, (x, y) in enumerate(dataloader_test):
    prediction, target = predict(x, y, model, postprocess_prediction, postprocess_target, config.MAIN_DEVICE)

    if len(np.nonzero(target)) != 0:
        fig, ax = plt.subplots(1,2)
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(0.7) 

        ax[0].imshow(target)
        ax[1].imshow(prediction)
        plt.show()

    int1 = compute_intersection(prediction, target, 1)
    uni1 = compute_union(prediction, target, 1)
    intersections_1.append(int1)
    unions_1.append(uni1)

    int2 = compute_intersection(prediction, target, 2)
    uni2 = compute_union(prediction, target, 2)
    intersections_2.append(int2)
    unions_2.append(uni2)
    print("Done with", ImageIDs[count])
    count += 1


#%%
dice_agg_1 = np.sum(np.array(intersections_1))/np.sum(np.array(unions_1))
dice_agg_2 = np.sum(np.array(intersections_2))/np.sum(np.array(unions_2))
dice_agg_avg = (dice_agg_1 + dice_agg_2)/2.0

print(dice_agg_avg)     
print(dice_agg_avg)
#%% take dice score of prediction and target
# %%

# %%
import pandas as pd
all_inference = np.column_stack((ImageIDs, intersections_1, unions_1, intersections_2, unions_2))
all_inf_df = pd.DataFrame(data=all_inference, columns=['ImageID', 'Int1', 'Uni1', 'Int2', 'Uni2'])
all_inf_df.to_csv('inference_unet_resnet34enc_diceloss_tfe200.csv')
# %%
import pandas as pd
import numpy as np
all_inf_df = pd.read_csv('inference_unet_resnet34enc_diceloss_tfe200.csv')
fgslice_df = all_inf_df[0:2179]

int1 = fgslice_df['Int1'].values.astype(int)
uni1 = fgslice_df['Uni1'].values.astype(int)
int2 = fgslice_df['Int2'].values.astype(int)
uni2 = fgslice_df['Uni2'].values.astype(int)

# %%
dsc_agg_1 = 2*np.sum(int1)/np.sum(uni1)
dsc_agg_2 = 2*np.sum(int2)/np.sum(uni2)

# %%
