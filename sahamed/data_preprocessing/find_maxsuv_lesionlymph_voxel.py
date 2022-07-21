#%%
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import utils
import pet_mask_box_generator as pmbg
import pandas as pd
import time
import os
# %%
gttrndir = '/data/blobfuse/hecktor2022/resampledCTPTGT/train/gt'
gttstdir = '/data/blobfuse/hecktor2022/resampledCTPTGT/test/gt'
pttrndir = '/data/blobfuse/hecktor2022/resampledCTPTGT/train/pt'
pttstdir = '/data/blobfuse/hecktor2022/resampledCTPTGT/test/pt'

gttrnpaths = sorted(glob.glob(os.path.join(gttrndir, '*.nii.gz')))
gttstpaths = sorted(glob.glob(os.path.join(gttstdir, '*.nii.gz')))
pttrnpaths = sorted(glob.glob(os.path.join(pttrndir, '*.nii.gz')))
pttstpaths = sorted(glob.glob(os.path.join(pttstdir, '*.nii.gz')))
# %%

gt_paths = gttrnpaths + gttstpaths
pt_paths = pttrnpaths + pttstpaths


#%%
# def make_zeros_except_given_coordinates()

patientIDs= []
lesion_suvmaxs = []
lymphnode_suvmaxs = []
# %%
for i in range(len(gt_paths)):
    gtpath = gt_paths[i]
    ptpath = pt_paths[i]
    gt, gtvox = utils.nib2numpy(gtpath)
    pt, ptvox = utils.nib2numpy(ptpath)
    patientid = os.path.basename(gtpath)[:-7]
    patientIDs.append(patientid)
    
    nzv_coords_x, nzv_coords_y, nzv_coords_z = np.nonzero(gt)
    lesion_voxels = []
    lymphnode_voxels = []
    for i in range(len(nzv_coords_x)):
        x_coord = nzv_coords_x[i]
        y_coord = nzv_coords_y[i]
        z_coord = nzv_coords_z[i]

        if gt[x_coord, y_coord, z_coord] == 1:
            # lesion
            lesion_voxels.append(pt[x_coord, y_coord, z_coord])
        else:
            # lymph node
            lymphnode_voxels.append(pt[x_coord, y_coord, z_coord])

    if len(lesion_voxels) != 0:
        lesion_suvmaxs.append(max(lesion_voxels))
    else:
        lesion_suvmaxs.append(np.nan)
    if len(lymphnode_voxels) != 0:
        lymphnode_suvmaxs.append(max(lymphnode_voxels))
    else:
        lymphnode_suvmaxs.append(np.nan)
    print("Done with ptid: ",patientid)
# %%



alldata = np.column_stack((patientIDs, lesion_suvmaxs, lymphnode_suvmaxs))
# %%
data_df = pd.DataFrame(data=alldata, columns=['PatientID', 'MaxLesionSUV', 'MaxLymphNodeSUV'])
data_df.to_csv('lesion_lymphnode_maxsuvs.csv')
# %%
fig, ax = plt.subplots(1,2, figsize=(12,7))
fig.patch.set_facecolor('white')
fig.patch.set_alpha(0.7)
ax[0].hist(lesion_suvmaxs,bins=20, edgecolor='black')
ax[1].hist(lymphnode_suvmaxs,bins=20, edgecolor='black')
ax[0].set_title('Lesion SUVmax', fontsize=15)
ax[1].set_title('Lymph Node SUVmax', fontsize=15)
fig.suptitle('PT SUVmax per patient for lesion & lymph-nodes', fontsize=20)
fig.savefig('lesion_lymphnode_suvmax_distribution.jpg')
plt.show()
# %%
