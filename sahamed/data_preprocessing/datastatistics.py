#%%
import pandas as pd 
import numpy as np 
import glob 
import os
import utils 



#%%
ptdirtrn = '/data/blobfuse/hecktor2022/resampledCTPTGT/train/pt'
ctdirtrn = '/data/blobfuse/hecktor2022/resampledCTPTGT/train/ct'
gtdirtrn = '/data/blobfuse/hecktor2022/resampledCTPTGT/train/gt'
ptdirtst = '/data/blobfuse/hecktor2022/resampledCTPTGT/test/pt'
ctdirtst = '/data/blobfuse/hecktor2022/resampledCTPTGT/test/ct'
gtdirtst = '/data/blobfuse/hecktor2022/resampledCTPTGT/test/gt'

# ptdirtrn = '/data/blobfuse/hecktor2022/'
# ctdirtrn = '/data/blobfuse/hecktor2022/train/ct'
# ptdirtst = '/data/blobfuse/hecktor2022/test/pt'
# ctdirtst = '/data/blobfuse/hecktor2022/test/ct'

pt_pathstrn =  sorted(glob.glob(os.path.join(ptdirtrn, '*.nii.gz')))
ct_pathstrn =  sorted(glob.glob(os.path.join(ctdirtrn, '*.nii.gz')))
gt_pathstrn =  sorted(glob.glob(os.path.join(gtdirtrn, '*.nii.gz')))

pt_pathstst =  sorted(glob.glob(os.path.join(ptdirtst, '*.nii.gz')))
ct_pathstst =  sorted(glob.glob(os.path.join(ctdirtst, '*.nii.gz')))
gt_pathstst =  sorted(glob.glob(os.path.join(gtdirtst, '*.nii.gz')))

# %%

pt_paths = pt_pathstrn + pt_pathstst
ct_paths = ct_pathstrn + ct_pathstst 
gt_paths = gt_pathstrn + gt_pathstst 

# %%
PatientIDs = []
CTmax = []
CTmin = []
PTmax = []
PTmin = []
CTSizeX = []
CTSizeY = []
CTSizeZ = []
PTSizeX = []
PTSizeY = []
PTSizeZ = []
GTSizeX = []
GTSizeY = []
GTSizeZ = []


CTSpacingX = []
CTSpacingY = []
CTSpacingZ = []
PTSpacingX = []
PTSpacingY = []
PTSpacingZ = []
GTSpacingX = []
GTSpacingY = []
GTSpacingZ = []

#%%
for i in range(len(pt_paths)):

    ptpath = pt_paths[i]
    ctpath = ct_paths[i]
    gtpath = gt_paths[i]

    PatientIDs.append(os.path.basename(gtpath)[:-7])

    imgpt, voxpt = utils.nib2numpy(ptpath)
    imgct, voxct = utils.nib2numpy(ctpath)
    imggt, voxgt = utils.nib2numpy(gtpath)

    CTSizeX.append(imgct.shape[0])
    CTSizeY.append(imgct.shape[1])
    CTSizeZ.append(imgct.shape[2])
    PTSizeX.append(imgpt.shape[0])
    PTSizeY.append(imgpt.shape[1])
    PTSizeZ.append(imgpt.shape[2])
    GTSizeX.append(imggt.shape[0])
    GTSizeY.append(imggt.shape[1])
    GTSizeZ.append(imggt.shape[2])

    CTSpacingX.append(voxct[0])
    CTSpacingY.append(voxct[1])
    CTSpacingZ.append(voxct[2])
    PTSpacingX.append(voxpt[0])
    PTSpacingY.append(voxpt[1])
    PTSpacingZ.append(voxpt[2])
    GTSpacingX.append(voxgt[0])
    GTSpacingY.append(voxgt[1])
    GTSpacingZ.append(voxgt[2])


    PTmax.append(imgpt.max())
    PTmin.append(imgpt.min())
    CTmax.append(imgct.max())
    CTmin.append(imgct.min())

    print("Done with ptid", os.path.basename(ptpath)[:-7])



# %%

allstats = np.column_stack((PatientIDs, CTmax, CTmin, PTmax, PTmin,CTSizeX, CTSizeY, CTSizeZ, PTSizeX, PTSizeY, PTSizeZ, GTSizeX, GTSizeY, GTSizeZ, CTSpacingX, CTSpacingY, CTSpacingZ, PTSpacingX, PTSpacingY, PTSpacingZ, GTSpacingX, GTSpacingY, GTSpacingZ))
stats_df = pd.DataFrame(data=allstats, columns = ['PatientIDs', 'CTmax', 'CTmin', 'PTmax', 'PTmin','CTSizeX', 'CTSizeY', 'CTSizeZ', 'PTSizeX', 'PTSizeY', 'PTSizeZ', 'GTSizeX', 'GTSizeY', 'GTSizeZ', 'CTSpacingX', 'CTSpacingY', 'CTSpacingZ', 'PTSpacingX', 'PTSpacingY', 'PTSpacingZ', 'GTSpacingX', 'GTSpacingY', 'GTSpacingZ'])
stats_df.to_csv('hecktor2022_images_stats_afterctfiltering.csv', index=False)


#%%
import matplotlib.pyplot as plt

#%%
def plot_ct_values_distribution(max_px_list, min_px_list, maintitle, savename):
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(0.7)

    ax[0].hist(max_px_list, bins=30, edgecolor='black')
    ax[1].hist(min_px_list,  bins=30, edgecolor='black')

    ax[0].set_xlabel('Hounsfield unit')
    ax[1].set_xlabel('Hounsfield unit')
    ax[0].set_title('Max HU values')
    ax[1].set_title('Min HU values')

    fig.suptitle(maintitle, fontsize=20)
    fig.savefig(savename)
    plt.show()

def plot_pt_values_distribution(max_px_list, min_px_list, maintitle, savename):
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(0.7)

    ax[0].hist(max_px_list, bins=30, edgecolor='black')
    ax[1].hist(min_px_list, bins=30, edgecolor='black')

    ax[0].set_xlabel('SUV')
    ax[1].set_xlabel('SUV')
    ax[0].set_title('Max SUV values')
    ax[1].set_title('Min SUV values')

    fig.suptitle(maintitle, fontsize=20)
    fig.savefig(savename)
    plt.show()

#%%
ctmax_list = stats_df['CTmax'].values.astype(float)
ctmin_list = stats_df['CTmin'].values.astype(float)
ptmax_list = stats_df['PTmax'].values.astype(float)
ptmin_list =  stats_df['PTmin'].values.astype(float) 

plot_ct_values_distribution(ctmax_list, ctmin_list,'HECKTOR 2022 CT intensity distribution', 'ctvalues_histogram.jpg')
plot_pt_values_distribution(ptmax_list, ptmin_list,'HECKTOR 2022 PT intensity distribution', 'ptvalues_histogram.jpg')
# %%

plt.hist(CTSizeX)
plt.show()
plt.hist(PTSizeX)
plt.show()
plt.hist(GTSizeX)
plt.show()
# %%
plt.hist(CTSizeY)
plt.show()
plt.hist(PTSizeY)
plt.show()
plt.hist(GTSizeY)
plt.show()
