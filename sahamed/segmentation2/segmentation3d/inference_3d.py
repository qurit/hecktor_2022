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
from transformations import resize_2dtensor_bilinear,resize_2dtensor_nearest
from dataset_3d import Segmentation3DDatasetPTCT_ptpreprocess1
from monai.networks.nets import UNet
from monai.losses import *
import torch.nn as nn
import segmentation_models_pytorch as smp
import time
from utils import get_required_paths_3d
#%%
def postprocess_prediction(inp, resampled_size):
    # inp is predicted tensor
    inp_label = torch.squeeze(inp_label, axis=0) # squeeze batch dimension
    inp_label = torch.argmax(inp, axis=0) # ohe to labels
    inp_label = inp_label.cpu().numpy() # get numpy array from torch tensor
    inp_label_resized = resize_2dtensor_nearest(inp_label, resampled_size, resampled_size) # resize to original resampled size
    
    return inp_label_resized  # 2D numpy array

def postprocess_target(inp, resampled_size):
    # perform argmax to generate 1 channel
    inp = inp.cpu().numpy()  # send to cpu and transform to numpy.ndarray
    inp = np.squeeze(inp)  # remove batch dim and channel dim -> [H, W]
    inp_resized = resize_2dtensor_nearest(inp, resampled_size, resampled_size)
    # img = re_normalize(img)  # scale it to the range [0-255]
    return inp_resized

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
    prediction = torch.softmax(out, dim=1)  # perform sigmoid on outputs
    return prediction

def save_as_nifti(inp, savepath):
    image = nib.Nifti1Image(inp, affine=np.eye(4))
    nib.save(image, savepath)

def compute_intersection(prediction, target, label):
    if label in target:
        intersection = sum(prediction[target==label]==label)
    else:
        intersection = np.nan
    return intersection


def compute_union(prediction, target, label):
    if label in target:
        union = sum(prediction[prediction==label]==label) + sum(target[target==label]==label) 
    else:
        union = np.nan
    return union
    

def compute_dicescore(prediction, target, label):
    intersection = compute_intersection(prediction, target, label)
    union = compute_union(prediction, target, label)
    dsc = 2*intersection/(union + 1e-5)
    return dsc
#%%

start = time.time()
FOLD = 0


experiment_code = 'unet3dmonai_diceceloss_fold'+str(FOLD)
save3dpred_dir = '/data/blobfuse/hecktor2022_training/hecktor2022/prediction_seg_valid'
savedir_pred3d = os.path.join(save3dpred_dir, experiment_code)
os.makedirs(savedir_pred3d, exist_ok=True)


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

# PatientIDs = []
# ResampledSizes = []
# count = 0
# for path in gt_paths:
#     imageid = os.path.basename(path)[:-7]
#     # array2d, _ = utils.nib2numpy(path)
#     gtimg = gtimg
#     ResampledSizes.append(array2d.shape[0])
#     PatientIDs.append(imageid)
#     print(count)
#     count+= 1

#%%
dataset_valid = Segmentation3DDatasetPTCT_ptpreprocess1(pt_paths_valid,ct_paths_valid,gt_paths_valid)#, transform=transforms)
dataloader_valid = DataLoader(dataset=dataset_valid, batch_size=1, shuffle=False, pin_memory=True, num_workers=24)


model = UNet(spatial_dims=3,\
            in_channels=2,\
            out_channels=3,\
            channels=(16, 32, 64, 128, 256),\
            strides=(2, 2, 2, 2)).to(config.MAIN_DEVICE)
model = nn.DataParallel(model, device_ids=[0,1,2,3])


optimal_model_name = 'segmentation_ep=400.pth'
saved_models_dir = '/data/blobfuse/saved_models_hecktor/segmentation2/'
optimal_model_dir = os.path.join(saved_models_dir, 'saved_models_'+experiment_code)
optimal_model_path = os.path.join(optimal_model_dir, optimal_model_name)
model.load_state_dict(torch.load(optimal_model_path))

#%%
DSC1 = []
DSC2 = []
int1 = []
uni1 = []
int2 = []
uni2 = []
for idx, (x, y) in enumerate(dataloader_valid):

    # predict output
    prediction = predict(x, model, config.MAIN_DEVICE)

    y_copy = y.clone().detach()
    y_copy = torch.squeeze(y_copy, axis=0)
    y_copy = torch.argmax(y_copy, axis=0)
    # print("From y:", np.unique(y_copy))
    

    pred_copy = prediction.clone().detach().cpu()
    pred_copy = torch.squeeze(pred_copy, axis=0)
    pred_copy = torch.argmax(pred_copy, axis=0)
    # print("From prediction:", np.unique(pred_copy))
    dsc1 = compute_dicescore(pred_copy, y_copy, 1)
    dsc2 = compute_dicescore(pred_copy, y_copy, 2)
    i1 = compute_intersection(pred_copy, y_copy, 1)
    i2 = compute_intersection(pred_copy, y_copy, 2)
    u1 = compute_union(pred_copy, y_copy, 1)
    u2 = compute_union(pred_copy, y_copy, 2)
    # print('DSC(1):', dsc1)
    # print('DSC(2):', dsc2)
    DSC1.append(dsc1)
    DSC2.append(dsc2)
    int1.append(i1)
    int2.append(i2)
    uni1.append(u1)
    uni2.append(u2)


mean_dsc_1 = np.nansum(DSC1)
mean_dsc_2 = np.nansum(DSC2)
agg_dsc_1 = 2*np.nansum(int1)/np.nansum(uni1)
agg_dsc_2 = 2*np.nansum(int2)/np.nansum(uni2)
agg_dsc_avg = (agg_dsc_1 + agg_dsc_2)/2

print(experiment_code)
print("Mean DSC 1", agg_dsc_avg)
print("Mean DSC 2", mean_dsc_2)
print("AggDSC 1",  agg_dsc_1)
print("AggDSC 2", agg_dsc_2)
print("AggDSC Avg", agg_dsc_avg)
    # # postprocessed_pt = postprocess_input(x, ResampledSizes[idx])
    
    # # output preprocess
    # ## ohe to labels (2D slice) 
    # ## resize the predicted slice size to corresponding resampled size
    
    # postprocessed_prediction = postprocess_prediction(prediction, ResampledSizes[idx])
    # postprocessed_target = postprocess_target(y, ResampledSizes[idx])

    
    # savename = ImageIDs[idx]+'.nii.gz'
    # savepath = os.path.join(savedir_pred2d, savename)
    # save_as_nifti(postprocessed_prediction, savepath)
    

    # int1 = compute_intersection(postprocessed_prediction, postprocessed_target, 1)
    # int2 = compute_intersection(postprocessed_prediction, postprocessed_target, 2)
    # uni1 = compute_union(postprocessed_prediction, postprocessed_target, 1)
    # uni2 = compute_union(postprocessed_prediction, postprocessed_target, 2)
    # print('prediction:', ImageIDs[idx])
    # print('Intersection(1):', int1)
    # print('Intersection(2):', int2)
    # print('Union(1):', uni1)
    # print('Union(2):', uni2)
    # print('\n\n')
    # fig, ax = plt.subplots(1,2, figsize=(10,5))
    # fig.patch.set_facecolor('white')
    # fig.patch.set_alpha(0.7)
    # ax[0].imshow(postprocessed_target)
    # ax[1].imshow(postprocessed_prediction)
    # # ax[2].imshow(postprocessed_pt)
    # ax[0].set_title('GT')
    # ax[1].set_title('Predicted')
    # # ax[2].set_title('PT')
    # plt.show()
    # plt.close('all')


#%%
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

# imgid_size_data = np.column_stack((ImageIDs, ResampledSizes))
# imgid_size_df = pd.DataFrame(data=imgid_size_data, columns=['ImageID', 'ResampledSize'])
# imgid_size_df.to_csv('imageids_resampledsizes.csv', index=False)

# imgid_sizes_df = pd.read_csv('imageids_resampledsizes.csv')
# ImageIDs = imgid_sizes_df['ImageID'].values
# ResampledSizes = imgid_sizes_df['ResampledSize'].values


#%%


#%%

# %%
# model = UNet(spatial_dims=2,\
#             in_channels=2,\
#             out_channels=3,\
#             channels=(16, 32, 64, 128, 256),\
#             strides=(2, 2, 2, 2)).to(config.MAIN_DEVICE)

model = smp.Unet(
    encoder_name="resnet34",        
    encoder_weights="imagenet",     
    in_channels=3,                 
    classes=3,               
).to('cuda:0')

model = nn.DataParallel(model, device_ids=[0,1,2,3])
#%%


#%%
experiment_code = 'unet2dresnet34smp_diceloss'
save2ddir = '/data/blobfuse/hecktor2022/resampledCTPTGT/test/axial_data/pred2d'
savedir_pred2d = os.path.join(save2ddir, experiment_code)
os.makedirs(savedir_pred2d, exist_ok=True)


optimal_model_name = 'segmentation_ep=110.pth'
saved_models_dir = '/data/blobfuse/saved_models_hecktor/segmentation/'
optimal_model_dir = os.path.join(saved_models_dir, 'saved_models_'+experiment_code)
optimal_model_path = os.path.join(optimal_model_dir, optimal_model_name)
model.load_state_dict(torch.load(optimal_model_path))

#%%

#%%
for idx, (x, y) in enumerate(dataloader):

    # predict output
    prediction = predict(x, model, config.MAIN_DEVICE)
    # postprocessed_pt = postprocess_input(x, ResampledSizes[idx])
    
    # output preprocess
    ## ohe to labels (2D slice) 
    ## resize the predicted slice size to corresponding resampled size
    
    postprocessed_prediction = postprocess_prediction(prediction, ResampledSizes[idx])
    postprocessed_target = postprocess_target(y, ResampledSizes[idx])

    
    savename = ImageIDs[idx]+'.nii.gz'
    savepath = os.path.join(savedir_pred2d, savename)
    save_as_nifti(postprocessed_prediction, savepath)
    

    int1 = compute_intersection(postprocessed_prediction, postprocessed_target, 1)
    int2 = compute_intersection(postprocessed_prediction, postprocessed_target, 2)
    uni1 = compute_union(postprocessed_prediction, postprocessed_target, 1)
    uni2 = compute_union(postprocessed_prediction, postprocessed_target, 2)
    print('prediction:', ImageIDs[idx])
    print('Intersection(1):', int1)
    print('Intersection(2):', int2)
    print('Union(1):', uni1)
    print('Union(2):', uni2)
    print('\n\n')
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(0.7)
    ax[0].imshow(postprocessed_target)
    ax[1].imshow(postprocessed_prediction)
    # ax[2].imshow(postprocessed_pt)
    ax[0].set_title('GT')
    ax[1].set_title('Predicted')
    # ax[2].set_title('PT')
    plt.show()
    plt.close('all')
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

pred2d_paths = sorted(glob.glob(os.path.join(savedir_pred2d, patientid+'*.nii.gz')))

#%%
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
    
    pred3dsavepath = os.path.join(savedir_pred3d, ptid+'.nii.gz')
    nib.save(pred3d_img, pred3dsavepath)
    print('saving 3d: ' + ptid)

#%%
pred3d_paths = sorted(glob.glob(os.path.join(savedir_pred3d, '*.nii.gz')))
#%%


# intersection1 = []
# intersection2 = []
# union1 = []
# union2 = []

# PatientIDs = []

# for i in range(len(gt3dresampled_paths)):

#     gt_path = gt3dresampled_paths[i]
#     pred_path = pred3d_paths[i]

#     patientid = os.path.basename(gt_path)[:-7]
#     PatientIDs.append(patientid)
#     gt, vox_gt = utils.nib2numpy(gt_path)
#     pred, vox_pred = utils.nib2numpy(pred_path)

#     int1 = compute_intersection(pred, gt, 1)
#     int2 = compute_intersection(pred, gt, 2)
#     uni1 = compute_union(pred, gt, 1)
#     uni2 = compute_union(pred, gt, 2)

#     intersection1.append(int1)
#     intersection2.append(int2)
#     union1.append(uni1)
#     union2.append(uni2)
#     print('evaluation: ', i)

# #%%
# all_inference = np.column_stack((PatientIDs, intersection1, union1, intersection2, union2))
# all_inf_df = pd.DataFrame(data=all_inference, columns=['PatientID', 'Int1', 'Uni1', 'Int2', 'Uni2'])
# fname = 'before_resampling_inference_' + experiment_code + '.csv'
# all_inf_df.to_csv(fname)

#%%
save3dresdir = '/data/blobfuse/hecktor2022/resampledCTPTGT/test/pred3d_resampled'
savedir_pred3dresamp = os.path.join(save3dresdir, experiment_code)
os.makedirs(savedir_pred3dresamp, exist_ok=True)

# get all the original gt images
orig_gtdir = '/data/blobfuse/hecktor2022/labelsTr/test'
orig_gtpaths = sorted(glob.glob(os.path.join(orig_gtdir, patientid+'*.nii.gz')))

pred3d_paths = sorted(glob.glob(os.path.join(savedir_pred3d, patientid+'*.nii.gz')))

#%%
import SimpleITK as sitk
for i in range(len(pred3d_paths)):
    gt_orig_path = orig_gtpaths[i]
    gt_pred_path = pred3d_paths[i]
    filename = os.path.basename(gt_orig_path)

    gt_orig_image = sitk.ReadImage(gt_orig_path) 
    gt_pred_image = sitk.ReadImage(gt_pred_path) 
    gt_pred_resampled = sitk.Resample(gt_pred_image, gt_orig_image,interpolator=sitk.sitkNearestNeighbor)
    gtpredresampsavepath = os.path.join(savedir_pred3dresamp, filename)
    sitk.WriteImage(gt_pred_resampled, gtpredresampsavepath)
    print('resampled to orig:', i)

#%%
gtpath = '/data/blobfuse/hecktor2022/labelsTr/test/CHUM-001.nii.gz'
prpath = '/data/blobfuse/hecktor2022/resampledCTPTGT/test/pred3d_resampled/unet2dresnet34smp_diceloss/CHUM-001.nii.gz'

gt, voxgt = utils.nib2numpy(gtpath)
pr, voxpr = utils.nib2numpy(prpath)

#%%
for i in range(pr.shape[2]):
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(0.7)
    ax[0].imshow(gt[:,:,i])
    ax[1].imshow(pr[:,:,i])
    # ax[2].imshow(postprocessed_pt)
    ax[0].set_title('GT')
    ax[1].set_title('Predicted')
    # ax[2].set_title('PT')
    plt.show()
    plt.close('all')

#%%
pred3d_resampled_paths = sorted(glob.glob(os.path.join(savedir_pred3dresamp, '*.nii.gz')))

#%%
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

#%%
dice_agg_1 = 2*np.sum(np.array(intersection1))/np.sum(np.array(union1))
dice_agg_2 = 2*np.sum(np.array(intersection2))/np.sum(np.array(union2))
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