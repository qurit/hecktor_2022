#%%
from scipy.signal import medfilt 
import pandas as pd 
import numpy as np 
import os 
import glob
import utils
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import nibabel as nib
#%%
main_dir = '/data/blobfuse/hecktor2022/resampledCTPTGT/'
ct_dir = os.path.join(main_dir, 'ct')
ct_paths = sorted(glob.glob(os.path.join(ct_dir, '*.nii.gz')))
save_dir = os.path.join(main_dir, 'ct_filtered')
os.makedirs(save_dir, exist_ok=True)
# %%
for path in ct_paths:
    fileID = os.path.basename(path)
    data, voxdim = utils.nib2numpy(path)
    data_filtered = median_filter(data, size=(5,5,5), mode='reflect')
    nifti_img = nib.Nifti1Image(data_filtered, affine=np.eye(4))
    nifti_img.header['pixdim'][1:4] = voxdim
    nib.save(nifti_img, os.path.join(save_dir, fileID))
    print("Done with fileID: ", fileID)
# %%
