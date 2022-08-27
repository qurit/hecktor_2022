#%%
import numpy as np
import pandas as pd 
import os 
import shutil 
import glob

#%%
labelstr_main = '/data/blobfuse/hecktor2022/labelsTr'
labelstr_paths = sorted(glob.glob(os.path.join(labelstr_main, '*.nii.gz')))
labelstr_train = os.path.join(labelstr_main, 'train')
labelstr_test = os.path.join(labelstr_main, 'test')

#%%
test_list_file = './../data_preprocessing/ptinfo_test_split_0p20.csv'
test_df = pd.read_csv(test_list_file)
test_ptids = test_df['PatientID'].values
# %%
for path in labelstr_paths:
    filename = os.path.basename(path)
    patientid = filename[:-7]
    if patientid in test_ptids:
        shutil.move(path, os.path.join(labelstr_test, filename))
    else:
        shutil.move(path, os.path.join(labelstr_train, filename))
# %%
