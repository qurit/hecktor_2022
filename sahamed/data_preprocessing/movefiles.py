#%%
import numpy as np 
import os 
import glob 
import pandas as pd 
import shutil

#%%
maindir = '/data/blobfuse/hecktor2022/resampledCTPTGT'
mainctdir = os.path.join(maindir, 'ct_filtered')
mainptdir = os.path.join(maindir, 'pt')
maingtdir = os.path.join(maindir, 'gt')
ct_paths = sorted(glob.glob(os.path.join(mainctdir, '*.nii.gz')))
pt_paths = sorted(glob.glob(os.path.join(mainptdir, '*.nii.gz')))
gt_paths = sorted(glob.glob(os.path.join(maingtdir, '*.nii.gz')))

# %%
traindir = '/data/blobfuse/hecktor2022/resampledCTPTGT/train/'
testdir = '/data/blobfuse/hecktor2022/resampledCTPTGT/test/'

cttrndir = os.path.join(traindir, 'ct')
pttrndir = os.path.join(traindir, 'pt')
gttrndir = os.path.join(traindir, 'gt')
cttstdir = os.path.join(testdir, 'ct')
pttstdir = os.path.join(testdir, 'pt')
gttstdir = os.path.join(testdir, 'gt')

#%%
# get test patient IDs
test_df = pd.read_csv('ptinfo_test_split_0p20.csv')  
test_ids = test_df['PatientID'].values.tolist()

# %%
for i in range(len(gt_paths)):
    ctpath = ct_paths[i]
    ptpath = pt_paths[i]
    gtpath = gt_paths[i]

    ctid = os.path.basename(ctpath)
    ptid = os.path.basename(ptpath)
    gtid = os.path.basename(gtpath)

    gtid_removeex = gtid[:-7]

    if gtid_removeex in test_ids:
        # move everything to test folder
        shutil.move(ctpath, os.path.join(cttstdir, ctid))
        shutil.move(ptpath, os.path.join(pttstdir, ptid))
        shutil.move(gtpath, os.path.join(gttstdir, gtid))
    else:
        # move everything to train folder
        shutil.move(ctpath, os.path.join(cttrndir, ctid))
        shutil.move(ptpath, os.path.join(pttrndir, ptid))
        shutil.move(gtpath, os.path.join(gttrndir, gtid))
    print("Done with PatientID:", gtid_removeex)




# %% 

