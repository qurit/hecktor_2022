#%%

import utils
import numpy as np
import os
from glob import glob
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
import glob
from sklearn.model_selection import train_test_split
from stratifiedsampling import stratified_sample, stratified_sample_report
import shutil

# %%
ptinfofname = '/data/blobfuse/hecktor2022/hecktor2022_clinical_info_training.csv'
pt_info_df = pd.read_csv(ptinfofname)


# #%%
# pt_info_df['StratifyCol'] = pt_info_df['CenterID'].astype(str) + '_' + \
#                             pt_info_df['Gender'].astype(str) + '_' + \
#                             pt_info_df['Age'].astype(str) + '_' + \
#                             pt_info_df['Weight'].astype(str) 
# #%%
# train_data, test_data = train_test_split(pt_info_df, 
#     random_state=42,
#     shuffle = True,
#     train_size=0.8,
#     stratify=pt_info_df['StratifyCol'])

# %%
train_df = stratified_sample(pt_info_df, strata=['CenterID', 'Gender'], size=0.8)
train_df.to_csv('ptinfo_train_split_0p80.csv', index=False)

#%%
train_df = pd.read_csv('ptinfo_train_split_0p80.csv')
train_indices = train_df['index'].values

#%%
column_names = pt_info_df.columns.values.tolist()
test_df = pd.DataFrame(columns = column_names)
#%%
for index, row in pt_info_df.iterrows():
    if index in train_indices:
        pass 
    else:
        ptid = row['PatientID']
        task1 = row['Task 1']
        task2 = row['Task 2']
        centerid = row['CenterID']
        gender = row['Gender']
        age = row['Age']
        weight = row['Weight']
        tobacco = row['Tobacco']
        alcohol = row['Alcohol']
        performance_stat = row['Performance status']
        hpv_stat = row['HPV status (0=-, 1=+)']
        surgery = row['Surgery']
        chemotherapy = row['Chemotherapy'] 

        row_data = np.column_stack((ptid,task1,task2,centerid, gender, age, weight, tobacco, alcohol, performance_stat,hpv_stat, surgery,chemotherapy))

        row_df = pd.DataFrame(data = row_data, columns=column_names)
        test_df = pd.concat([test_df, row_df], ignore_index=True)
# %%
test_df.to_csv('ptinfo_test_split_0p20.csv', index=False)
# %%
test_df_ = pd.read_csv('ptinfo_test_split_0p20.csv')
# %%
# move data to train or test folders
images_dir = '/data/blobfuse/hecktor2022/imagesTr'
labels_dir = '/data/blobfuse/hecktor2022/labelsTr'
ct_paths = sorted(glob.glob(os.path.join(images_dir, '*CT.nii.gz'))) 
pt_paths = sorted(glob.glob(os.path.join(images_dir, '*PT.nii.gz'))) 
gt_paths = sorted(glob.glob(os.path.join(labels_dir, '*.nii.gz'))) 

train_dir = '/data/blobfuse/hecktor2022/train'
test_dir = '/data/blobfuse/hecktor2022/test'
train_ctdir = os.path.join(train_dir, 'ct') 
train_ptdir = os.path.join(train_dir, 'pt') 
train_gtdir = os.path.join(train_dir, 'gt') 
test_ctdir = os.path.join(test_dir, 'ct') 
test_ptdir = os.path.join(test_dir, 'pt') 
test_gtdir = os.path.join(test_dir, 'gt') 

# %%
test_ids = pd.read_csv('ptinfo_test_split_0p20.csv')['PatientID'].values.tolist()
# %%
for i in range(len(pt_paths)):
    fileid = os.path.basename(pt_paths[i])
    ptid = fileid[:-11]
    if ptid in test_ids:
        shutil.copy(ct_paths[i], os.path.join(test_ctdir, fileid))
        shutil.copy(pt_paths[i], os.path.join(test_ptdir, fileid))
        shutil.copy(gt_paths[i], os.path.join(test_gtdir, fileid))
    else:
        shutil.copy(ct_paths[i], os.path.join(train_ctdir, fileid))
        shutil.copy(pt_paths[i], os.path.join(train_ptdir, fileid))
        shutil.copy(gt_paths[i], os.path.join(train_gtdir, fileid))
    print("Done with ptid:" ,ptid)
    
# %%
