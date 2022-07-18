#%%
import numpy as np 
import os 
import glob 
import pandas as pd 

#%%
mainptdir = '/data/blobfuse/hecktor2022/resampled_PT'
ptpaths = sorted(glob.glob(os.path.join(mainptdir, '*PT.nii.gz')))
# %%
trainptdir = '/data/blobfuse/hecktor2022/train/pt'
testptdir = '/data/blobfuse/hecktor2022/test/pt'
ptpaths_trn = sorted(glob.glob(os.path.join(trainptdir, '*PT.nii.gz')))
ptpaths_tst = sorted(glob.glob(os.path.join(testptdir, '*PT.nii.gz')))
# %%

for i in range(len(ptpaths_trn)):
    pth = ptpaths_trn[i]
    ptpaths_trn[i] = os.path.basename(pth)

for i in range(len(ptpaths_tst)):
    pth = ptpaths_tst[i]
    ptpaths_tst[i] = os.path.basename(pth)   
# %% 
import shutil
trnptdir = '/data/blobfuse/hecktor2022/train/pt_resampled'
tstptdir = '/data/blobfuse/hecktor2022/test/pt_resampled'

for path in ptpaths:
    fileID = os.path.basename(path)
    if fileID in ptpaths_trn:
        shutil.copy(path, os.path.join(trnptdir, fileID))
    else:
        shutil.copy(path, os.path.join(tstptdir, fileID))
    print("Done with fileID:", fileID)


# %%
trndf = pd.read_csv('ptinfo_train_split_0p80.csv')
tstdf = pd.read_csv('ptinfo_test_split_0p20.csv')

trnids = trndf['PatientID'].values.tolist()
tstids = tstdf['PatientID'].values.tolist()

# %%
ptdir = '/data/blobfuse/hecktor2022/resampled_PT'
ctdir = '/data/blobfuse/hecktor2022/imagesTr'
gtdir = '/data/blobfuse/hecktor2022/labelsTr'
pt_paths = sorted(glob.glob(os.path.join(ptdir, '*PT.nii.gz')))
ct_paths = sorted(glob.glob(os.path.join(ctdir, '*CT.nii.gz')))
gt_paths = sorted(glob.glob(os.path.join(gtdir, '*.nii.gz')))

ptsavetrn = '/data/blobfuse/hecktor2022/train/pt'
ctsavetrn = '/data/blobfuse/hecktor2022/train/ct'
gtsavetrn = '/data/blobfuse/hecktor2022/train/gt'
ptsavetst = '/data/blobfuse/hecktor2022/test/pt'
ctsavetst = '/data/blobfuse/hecktor2022/test/ct'
gtsavetst = '/data/blobfuse/hecktor2022/test/gt'
# %%
import shutil
for i in range(len(gt_paths)):
    gtpath = gt_paths[i]
    ptpath = pt_paths[i]
    ctpath = ct_paths[i]
    gtid = os.path.basename(gtpath)[:-7] # patientID
    ptid = os.path.basename(ptpath)
    ctid = os.path.basename(ctpath)
    
    if gtid in trnids:
        shutil.move(gtpath, os.path.join(gtsavetrn, gtid+'.nii.gz'))
        shutil.move(ptpath, os.path.join(ptsavetrn, ptid))
        shutil.move(ctpath, os.path.join(ctsavetrn, ctid))
    else:
        shutil.move(gtpath, os.path.join(gtsavetst, gtid+'.nii.gz'))
        shutil.move(ptpath, os.path.join(ptsavetst, ptid))
        shutil.move(ctpath, os.path.join(ctsavetst, ctid))

    print("Done with ptid", gtid)
# %%
