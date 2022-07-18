# %%
import numpy as np
import os
import SimpleITK as sitk
import pandas as pd
from pathlib import Path
import glob
import utils  
import SimpleITK as sitk
# %%D:\Projects\HECKTOR2022_data\hecktor2022\test
dir = '/data/blobfuse/hecktor2022/imagesTr'
ptdir = '/data/blobfuse/hecktor2022/resampledCTGT/pt'
dir1 = '/data/blobfuse/hecktor2022/labelsTr'
# ct_paths = sorted(list(Path(main_folder.joinpath('ct')).rglob("*.nii.gz")))
# pt_paths = sorted(list(Path(main_folder.joinpath('pt')).rglob("*.nii.gz")))
# imgdir = os.path.join(dir, 'test_images')
# lbldir = os.path.join(dir, 'test_labels')

ct_paths = sorted(glob.glob(os.path.join(dir ,"*CT.nii.gz")))
pt_paths = sorted(glob.glob(os.path.join(ptdir ,"*PT.nii.gz")))
gt_paths = sorted(glob.glob(os.path.join(dir1 ,"*.nii.gz")))
#%%
# resampled_ct_dir = main_folder + 'ct_resampled/' 
resampled_dir = '/data/blobfuse/hecktor2022/resampledCTGT/'
ct_resamp_dir = os.path.join(resampled_dir, 'ct')
gt_resamp_dir = os.path.join(resampled_dir, 'gt')
# %%
path_data = np.column_stack((ct_paths, pt_paths, gt_paths))#, gt_paths))
# %%
for dat in path_data:
    ctpath, ptpath, gtpath = dat
    ctimg = sitk.ReadImage(ctpath, imageIO="NiftiImageIO")
    ptimg = sitk.ReadImage(ptpath, imageIO="NiftiImageIO")  
    gtimg = sitk.ReadImage(gtpath, imageIO="NiftiImageIO")  
    resampled_ctimg = sitk.Resample(ctimg, ptimg, interpolator=sitk.sitkLinear, defaultPixelValue=-1024)
    resampled_gtimg = sitk.Resample(gtimg, ptimg, interpolator=sitk.sitkNearestNeighbor, defaultPixelValue=0)


    fileIDct = os.path.basename(ctpath)
    fileIDgt = os.path.basename(gtpath)
    ctsavename = os.path.join(ct_resamp_dir, fileIDct)
    gtsavename = os.path.join(gt_resamp_dir, fileIDgt)

    sitk.WriteImage(resampled_ctimg, ctsavename)
    sitk.WriteImage(resampled_gtimg, gtsavename)
    print(f'Done with patient: ', fileIDct)
    print(f'Done with patient: ', fileIDgt)

#%%
