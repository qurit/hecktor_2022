#%%
import config 
import utils
import numpy as np
import os
import pandas as pd
import torch
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from random import shuffle 
import nibabel as nib
import glob
import torch.nn.functional as F
from transformations import resize_2dtensor_bilinear,resize_2dtensor_nearest, trunc_scale_preprocess_pt, trunc_scale_preprocess_ct
from dataset_2d import Segmentation2DDataset2ChannelInput
from monai.networks.nets import UNet
from monai.losses import *
import torch.nn as nn
import segmentation_models_pytorch as smp
import time
#%%
def postprocess_prediction(inp, resampled_size):
    # inp is predicted tensor
    inp_label = torch.argmax(inp, axis=1) # ohe to labels
    inp_label = torch.squeeze(inp_label, axis=0) # squeeze batch dimension
    inp_label = inp_label.cpu().numpy() # get numpy array from torch tensor
    inp_label_resized = resize_2dtensor_nearest(inp_label, resampled_size, resampled_size) # resize to original resampled size
    
    return inp_label_resized  # 2D numpy array

def postprocess_input(inp, resampled_size):
    inp = torch.squeeze(inp, axis=0)
    inp = inp[0] # get the pet channel
    inp = inp.cpu().numpy()
    inp_resized = resize_2dtensor_bilinear(inp, resampled_size, resampled_size)
    return inp_resized 

def predict(img, model, device):
    model.eval()
    x = img.to(device)  # to torch, send to device
    with torch.no_grad():
        out = model(x)  # send through model/network
    prediction = torch.sigmoid(out)  # perform sigmoid on outputs
    return prediction

def save_as_nifti(inp, savepath):
    image = nib.Nifti1Image(inp, affine=np.eye(4))
    nib.save(image, savepath)
#%%

start = time.time()

axial_dir = '/data/blobfuse/hecktor2022/resampledCTPTGT/test/axial_data/'
ctfg_dir = os.path.join(axial_dir, 'ct_fg')
ctbg_dir = os.path.join(axial_dir, 'ct_bg')
ptfg_dir = os.path.join(axial_dir, 'pt_fg')
ptbg_dir = os.path.join(axial_dir, 'pt_bg')
gtfg_dir = os.path.join(axial_dir, 'gt_fg')
gtbg_dir = os.path.join(axial_dir, 'gt_bg')

# patientid = 'HMR-013'
# ctfg_paths = sorted(glob.glob(os.path.join(ctfg_dir, patientid+'*.nii.gz')))
# ctbg_paths = sorted(glob.glob(os.path.join(ctbg_dir, patientid+'*.nii.gz')))
# ptfg_paths = sorted(glob.glob(os.path.join(ptfg_dir, patientid+'*.nii.gz')))
# ptbg_paths = sorted(glob.glob(os.path.join(ptbg_dir, patientid+'*.nii.gz')))
# gtfg_paths = sorted(glob.glob(os.path.join(gtfg_dir, patientid+'*.nii.gz')))
# gtbg_paths = sorted(glob.glob(os.path.join(gtbg_dir, patientid+'*.nii.gz')))


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

gt3dresampled_dir = '/data/blobfuse/hecktor2022/resampledCTPTGT/test/gt'
gt3dresampled_paths = sorted(glob.glob(os.path.join(gt3dresampled_dir, '*.nii.gz')))

ImageIDs = []
ResampledSizes = []
count = 0
for path in gt_paths:
    imageid = os.path.basename(path)[:-7]
    array2d, _ = utils.nib2numpy(path)
    ResampledSizes.append(array2d.shape[0])
    ImageIDs.append(imageid)
    print(count)
    count+= 1

#%%
inputs_ct = ct_paths
inputs_pt = pt_paths
targets = gt_paths
#%%

dataset = Segmentation2DDataset2ChannelInput(inputs_pt,inputs_ct,targets)#, transform=transforms)

#%%
dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

#%%

# %%
model = UNet(spatial_dims=2,\
            in_channels=2,\
            out_channels=3,\
            channels=(16, 32, 64, 128, 256),\
            strides=(2, 2, 2, 2)).to(config.MAIN_DEVICE)

# model = smp.Unet(
#     encoder_name="resnet34",        
#     encoder_weights="imagenet",     
#     in_channels=3,                 
#     classes=3,                 
# ).to(config.MAIN_DEVICE)

model = nn.DataParallel(model, device_ids=[0,1,2,3])
#%%


#%%
experiment_code = 'unet2dsmallmonai_generalizeddiceloss'
save2ddir = '/data/blobfuse/hecktor2022/resampledCTPTGT/test/axial_data/pred2d'
savedir_pred2d = os.path.join(save2ddir, experiment_code)
os.makedirs(savedir_pred2d, exist_ok=True)


optimal_model_name = 'segmentation_ep=200.pth'
saved_models_dir = '/data/blobfuse/saved_models_hecktor/segmentation/'
optimal_model_dir = os.path.join(saved_models_dir, 'saved_models_'+experiment_code)
optimal_model_path = os.path.join(optimal_model_dir, optimal_model_name)
model.load_state_dict(torch.load(optimal_model_path))

#%%
def compute_intersection(prediction, target, label):
    intersection = np.sum(prediction[target==label]==label)
    return intersection

def compute_union(prediction, target, label):
    union = np.sum(prediction[prediction==label]==label) + np.sum(target[target==label]==label) 
    return union
#%%


for idx, (x, y) in enumerate(dataloader):

    # predict output
    prediction = predict(x, model, config.MAIN_DEVICE)
    # postprocessed_pt = postprocess_input(x, ResampledSizes[idx])
    
    # output preprocess
    ## ohe to labels (2D slice) 
    ## resize the predicted slice size to corresponding resampled size
    
    postprocessed_prediction = postprocess_prediction(prediction, ResampledSizes[idx])
    postprocessed_target = postprocess_prediction(y, ResampledSizes[idx])

    savename = ImageIDs[idx]
    savepath = os.path.join(savedir_pred2d, savename)
    save_as_nifti(postprocessed_prediction, savepath)
    print('prediction:', idx)

    # fig, ax = plt.subplots(1,3, figsize=(10,5))
    # fig.patch.set_facecolor('white')
    # fig.patch.set_alpha(0.7)
    # ax[0].imshow(postprocessed_target)
    # ax[1].imshow(postprocessed_prediction)
    # ax[2].imshow(postprocessed_pt)
    # ax[0].set_title('GT')
    # ax[1].set_title('Predicted')
    # ax[2].set_title('PT')
    # plt.show()
    # plt.close('all')
    ## save it with the correct name (patientID, sliceID)
    


# All 2d predictions have been saved now
# need to stack them as 3D images (done)
# then they will be the size of resampled GT (done)
# Find Dice_agg at this stage 
# then resample all 3D predictions to original GT/CT size (done)
# then find Dice_agg at this stage as well 
# then resample both original GT and 3D predictions to 1x1x1
# find Dice_agg at this stage too
#%%
save3ddir = '/data/blobfuse/hecktor2022/resampledCTPTGT/test/pred3d'
savedir_pred3d = os.path.join(save3ddir, experiment_code)
os.makedirs(savedir_pred3d, exist_ok=True)

pred2d_paths = sorted(glob.glob(os.path.join(savedir_pred2d, '*.nii.gz')))

# find unique patient names
PatientIDs = []
Pred3D_dict = {}
for path in pred2d_paths:
    patientID = os.path.basename(path).split('_')[0]
    print(patientID)
    PatientIDs.append(patientID)

Unique_PatientIDs = np.unique(PatientIDs)
for i in range(len(Unique_PatientIDs)):
    Pred3D_dict[Unique_PatientIDs[i]] = []

for path in pred2d_paths:
    patientID = os.path.basename(path).split('_')[0]
    array2d, vox = utils.nib2numpy(path)
    Pred3D_dict[patientID].append(array2d)

for ptid in Unique_PatientIDs:
    pred3d = np.stack(Pred3D_dict[ptid], axis=-1)
    pred3d_img = nib.Nifti1Image(pred3d, affine=np.eye(4))
    
    pred3dsavepath = os.path.join(savedir_pred3d, ptid+'_.nii.gz')
    nib.save(pred3d_img, pred3dsavepath)
    print('saving 3d: ' + ptid)


#%%
save3dresdir = '/data/blobfuse/hecktor2022/resampledCTPTGT/test/pred3d_resampled'
savedir_pred3dresamp = os.path.join(save3dresdir, experiment_code)
os.makedirs(savedir_pred3dresamp, exist_ok=True)

# get all the original gt images
orig_gtdir = '/data/blobfuse/hecktor2022/labelsTr/test'
orig_gtpaths = sorted(glob.glob(os.path.join(orig_gtdir, '*.nii.gz')))

pred3d_paths = sorted(glob.glob(os.path.join(savedir_pred3d, '*.nii.gz')))

#%%
import SimpleITK as sitk
for i in range(len(pred3d_paths)):
    gt_orig_path = orig_gtpaths[i]
    gt_pred_path = pred3d_paths[i]
    filename = os.path.basename(gt_orig_path)

    gt_orig_image = sitk.ReadImage(gt_orig_path, imageIO="NiftiImageIO") 
    gt_pred_image = sitk.ReadImage(gt_pred_path, imageIO="NiftiImageIO") 
    gt_pred_resampled = sitk.Resample(gt_pred_image, gt_orig_image, interpolator=sitk.sitkNearestNeighbor, defaultPixelValue=0)
    gtpredresampsavepath = os.path.join(savedir_pred3dresamp, filename)
    sitk.WriteImage(gt_pred_resampled, gtpredresampsavepath)
    print('resampled to orig:', i)


#%%
pred3d_resampled_paths = sorted(glob.glob(os.path.join(savedir_pred3dresamp, '*.nii.gz')))

intersection1 = []
intersection2 = []
union1 = []
union2 = []

PatientIDs = []

for i in range(len(pred3d_resampled_paths)):

    orig_gt_path = orig_gtpaths[i]
    pred_gt_path = pred3d_resampled_paths[i]

    patientid = os.path.basename(orig_gt_path)[:-7]
    PatientIDs.append(patientid)
    orig_gt, vox_orig = utils.nib2numpy(orig_gt_path)
    pred_gt, vox_pred = utils.nib2numpy(pred_gt_path)

    int1 = compute_intersection(pred_gt, orig_gt, 1)
    int2 = compute_intersection(pred_gt, orig_gt, 2)
    uni1 = compute_union(pred_gt, orig_gt, 1)
    uni2 = compute_union(pred_gt, orig_gt, 2)

    intersection1.append(int1)
    intersection2.append(int2)
    union1.append(uni1)
    union2.append(uni2)
    print('evaluation: ', i)

#%%
all_inference = np.column_stack((PatientIDs, intersection1, union1, intersection2, union2))
all_inf_df = pd.DataFrame(data=all_inference, columns=['PatientID', 'Int1', 'Uni1', 'Int2', 'Uni2'])
fname = 'inference_' + experiment_code + '.csv'
all_inf_df.to_csv(fname)

dice_agg_1 = np.sum(np.array(intersection1))/np.sum(np.array(union1))
dice_agg_2 = np.sum(np.array(intersection2))/np.sum(np.array(union2))
dice_agg_avg = (dice_agg_1 + dice_agg_2)/2.0
print("DSC_agg(1) = ", dice_agg_1)
print("DSC_agg(2) = ", dice_agg_2)
print("DSC_agg_avg = ", dice_agg_avg)

elapsed = time.time() - start 
print("Time taken: " + str(round(elapsed,2)) + ' s')
#%%







# %%
# dice_agg_1 = np.sum(np.array(intersections_1))/np.sum(np.array(unions_1))
# dice_agg_2 = np.sum(np.array(intersections_2))/np.sum(np.array(unions_2))
# dice_agg_avg = (dice_agg_1 + dice_agg_2)/2.0
# # %%
# intersections_1 = []
# unions_1 = []
# intersections_2 = []
# unions_2 = []
# int1 = compute_intersection(postprocessed_prediction, postprocessed_target, 1)
# int2 = compute_intersection(postprocessed_prediction, postprocessed_target, 2)
# uni1 = compute_union(postprocessed_prediction, postprocessed_target, 1)
# uni2 = compute_union(postprocessed_prediction, postprocessed_target, 2)
# intersections_1.append(int1)
# intersections_2.append(int2)
# unions_1.append(uni1)
# unions_2.append(uni2)