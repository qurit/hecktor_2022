#%%
import pandas as pd 
import numpy as np 
import glob 
import os
import utils 



#%%
ptdir = '/data/blobfuse/hecktor2022/resampledCTPTGT/pt'
ctdir = '/data/blobfuse/hecktor2022/resampledCTPTGT/ct'
gtdir = '/data/blobfuse/hecktor2022/resampledCTPTGT/gt'


# ptdirtrn = '/data/blobfuse/hecktor2022/'
# ctdirtrn = '/data/blobfuse/hecktor2022/train/ct'
# ptdirtst = '/data/blobfuse/hecktor2022/test/pt'
# ctdirtst = '/data/blobfuse/hecktor2022/test/ct'

pt_paths =  sorted(glob.glob(os.path.join(ptdir, '*.nii.gz')))
ct_paths =  sorted(glob.glob(os.path.join(ctdir, '*.nii.gz')))
gt_paths =  sorted(glob.glob(os.path.join(gtdir, '*.nii.gz')))

# %%

# pt_paths = ptpathstrn + ptpathstst
# ct_paths = ctpathstrn + ctpathstst 
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


    PTmax.append(imgpt.max())
    PTmin.append(imgpt.min())
    CTmax.append(imgct.max())
    CTmin.append(imgct.min())

    print("Done with ptid", os.path.basename(ptpath)[:-7])



# %%

allstats = np.column_stack((PatientIDs, CTmax, CTmin, PTmax, PTmin,CTSizeX, CTSizeY, CTSizeZ, PTSizeX, PTSizeY, PTSizeZ, GTSizeX, GTSizeY, GTSizeZ))
stats_df = pd.DataFrame(data=allstats, columns = ['PatientIDs', 'CTmax', 'CTmin', 'PTmax', 'PTmin','CTSizeX', 'CTSizeY', 'CTSizeZ', 'PTSizeX', 'PTSizeY', 'PTSizeZ', 'GTSizeX', 'GTSizeY', 'GTSizeZ'])
stats_df.to_csv('hecktor2022_images_stats.csv', index=False)


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

plot_ct_values_distribution(ctmax_list, ctmin_list, 'HECKTOR 2022 CT intensity distribution', 'ctvalues_histogram.jpg')
plot_pt_values_distribution(ptmax_list, ptmin_list,'HECKTOR 2022 PT intensity distribution', 'ptvalues_histogram.jpg')
# %%
