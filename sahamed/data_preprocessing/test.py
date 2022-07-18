#%%
import numpy as np 
import os 
import shutil
import glob 
#%%
srcdir = '/data/blobfuse/hecktor2022/imagesTr'
paths = sorted(glob.glob(os.path.join(srcdir, '*PT.nii.gz')))
# %%
dstdir = '/data/blobfuse/hecktor2022/resampledCTGT/pt'

for path in paths:
    fileID = os.path.basename(path)
    shutil.move(path, os.path.join(dstdir, fileID))
    print("done with fileID:", fileID)
# %%


#%%
