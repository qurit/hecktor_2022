#%%
import utils 
import pandas as pd 
import numpy as np 
import shutil
import glob
import os
#%%
# fetch all images
data_dir_ctpt = '/data/blobfuse/hecktor2022/imagesTr'
data_dir_gt = '/data/blobfuse/hecktor2022/labelsTr'
ct_paths = sorted(glob.glob(os.path.join(data_dir_ctpt, '*CT.nii.gz')))
pt_paths = sorted(glob.glob(os.path.join(data_dir_ctpt, '*PT.nii.gz')))
gt_paths = sorted(glob.glob(os.path.join(data_dir_gt, '*.nii.gz')))

all_paths = np.column_stack((ct_paths, pt_paths, gt_paths))

#%%
# generate info file for 3D images

PatientID = []

CTSizeX = []
CTSizeY = []
CTSizeZ = []
CTVoxX = []
CTVoxY = []
CTVoxZ = []

PTSizeX = []
PTSizeY = []
PTSizeZ = []
PTVoxX = []
PTVoxY = []
PTVoxZ = []

GTSizeX = []
GTSizeY = []
GTSizeZ = []
GTVoxX = []
GTVoxY = []
GTVoxZ = []


# for path in ct_paths:

 
# %%
for data in all_paths:
    ctpath, ptpath, gtpath = data 
    
    ptid = os.path.basename(gtpath)[:-7]
    ctdata, ctvox = utils.nib2numpy(ctpath)
    ptdata, ptvox = utils.nib2numpy(ptpath)
    gtdata, gtvox = utils.nib2numpy(gtpath)

    ctx, cty, ctz = ctdata.shape
    ptx, pty, ptz = ptdata.shape
    gtx, gty, gtz = gtdata.shape

    ctvx, ctvy, ctvz = ctvox
    ptvx, ptvy, ptvz = ptvox
    gtvx, gtvy, gtvz = gtvox

    PatientID.append(ptid)

    CTSizeX.append(ctx)
    CTSizeY.append(cty)
    CTSizeZ.append(ctz)
    CTVoxX.append(ctvx)
    CTVoxY.append(ctvy)
    CTVoxZ.append(ctvz)

    PTSizeX.append(ptx)
    PTSizeY.append(pty)
    PTSizeZ.append(ptz)
    PTVoxX.append(ptvx)
    PTVoxY.append(ptvy)
    PTVoxZ.append(ptvz)

    GTSizeX.append(gtx)
    GTSizeY.append(gty)
    GTSizeZ.append(gtz)
    GTVoxX.append(gtvx)
    GTVoxY.append(gtvy)
    GTVoxZ.append(gtvz)

    print('Done with patient ID: ', ptid)


# %%
all_images_data = np.column_stack((PatientID, \
                                CTSizeX, CTSizeY, CTSizeZ,\
                                CTVoxX, CTVoxY, CTVoxZ,\
                                PTSizeX, PTSizeY, PTSizeZ,\
                                PTVoxX, PTVoxY, PTVoxZ, \
                                GTSizeX, GTSizeY, GTSizeZ, \
                                GTVoxX, GTVoxY, GTVoxZ
                                ))
# %%
all_images_df = pd.DataFrame(data = all_images_data,\
                            columns=['PatientID', \
                                'CTSizeX', 'CTSizeY', 'CTSizeZ',\
                                'CTVoxX', 'CTVoxY', 'CTVoxZ',\
                                'PTSizeX', 'PTSizeY', 'PTSizeZ',\
                                'PTVoxX', 'PTVoxY', 'PTVoxZ', \
                                'GTSizeX', 'GTSizeY', 'GTSizeZ', \
                                'GTVoxX', 'GTVoxY', 'GTVoxZ']
    
)
# %%
all_images_df.to_csv('images_data_info.csv', index=False)
# %%
import matplotlib.pyplot as plt
plt.hist(CTSizeX)
# %%
