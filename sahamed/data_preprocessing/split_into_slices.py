#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 01:45:09 2021

@author: shadab
"""
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
###################################################################################################
###################################################################################################
########################   Loading 3D PET and MASK files  #########################################
###################################################################################################
###################################################################################################
#%%
print("Loading 3D PET, MASK and CT files\n")

# 3D niftii PET and MASK folder 
MAIN_DIR = '/data/blobfuse/hecktor2022/resampledCTPTGT/test'
AXIAL_DATA_DIR = os.path.join(MAIN_DIR, 'axial_data')

CT_DIR = os.path.join(MAIN_DIR, 'ct') 
PT_DIR = os.path.join(MAIN_DIR, 'pt') 
GT_DIR = os.path.join(MAIN_DIR, 'gt') 


CT_FG_SLICE_DIR = os.path.join(AXIAL_DATA_DIR, 'ct_fg') 
CT_BG_SLICE_DIR = os.path.join(AXIAL_DATA_DIR, 'ct_bg') 
PT_FG_SLICE_DIR = os.path.join(AXIAL_DATA_DIR, 'pt_fg') 
PT_BG_SLICE_DIR = os.path.join(AXIAL_DATA_DIR, 'pt_bg') 
GT_FG_SLICE_DIR = os.path.join(AXIAL_DATA_DIR, 'gt_fg') 
GT_BG_SLICE_DIR = os.path.join(AXIAL_DATA_DIR, 'gt_bg') 

CT_3D_paths = sorted(glob.glob(os.path.join(CT_DIR, '*.nii.gz')))
PT_3D_paths = sorted(glob.glob(os.path.join(PT_DIR, '*.nii.gz')))
GT_3D_paths = sorted(glob.glob(os.path.join(GT_DIR, '*.nii.gz')))

#%%
# info files for different cross-sectional views
axial_info_fname = 'axial_info.csv'
axial_info_file = open(os.path.join(AXIAL_DATA_DIR, axial_info_fname), 'w+')
axial_info_file.write("PatientID,NSlices,NFgSlices,NBgSlices\n")


#%%
###################################################################################################
###################################################################################################
############################## processing 3D niftii images of MASK ###############################
###################################################################################################
###################################################################################################

# processing 3D niftii images of MASK
# generates the MASK_info.csv, MASK_slices_count.csv files
# generates a dictionary storing information about the slices count
# (this dictionary will be imported later to save slices of 3D PET images)
# saves the lesion and non-lesion slices in the respective folders in .nii format

print("\nProcessing 3D GT/PT/CT files\n")

# initializing dictionary to store the info about which slices are lesion and non-lesion
# key = (patientID, caseID)
axial_slices_info_dict = {}

for i in range(len(GT_3D_paths)):
    ctpath = CT_3D_paths[i]
    ptpath = PT_3D_paths[i]
    gtpath = GT_3D_paths[i]

    # get header info about the image
    PID = os.path.basename(gtpath)[:-7]
    
    
#     PID, CID, Type = fileID.split('_')

#     # # get 3D array in numpy format 
    array3dct, voxdimct = utils.nib2numpy(ctpath) 
    array3dpt, voxdimpt = utils.nib2numpy(ptpath)
    array3dgt, voxdimgt = utils.nib2numpy(gtpath)
    
    
#     # # # print info about case details, image size, voxel dimension
#     # n_islands, tmtv_pixel, tmtv_mm3 = lv.find_TMTV_3D(path)
#     # mtv_pixel, mtv_mm3 = lv.find_lesionvolume_3D(path)
    
#     # data_info_file.write(PID + ',' + CID +\
#     #                      ',(' + str(array3d.shape[0]) + " " + str(array3d.shape[1]) + " " + str(array3d.shape[2]) + "),(" \
#     #                          + str(voxdim[0]) + " " + str(voxdim[1]) + " " + str(voxdim[2]) + "),"\
#     #                         + str(tmtv_pixel) + ',' + str(tmtv_mm3) + ',' 
#     #                         + str(n_islands) + ','\
#     #                         + str(mtv_pixel) + ',' + str(mtv_mm3) + "\n")
    
    
#     # # # print info about number of slices, number of slices with and without lesions and save it to dictionary
   
#     # # Generate axial slices
    n_axial_slices = utils.count_slices(gtpath, cross_section='axial')
    axial_fg_slices, axial_bg_slices = utils.select_slices(gtpath, cross_section='axial')
    axial_info_file.write(PID + ',' + str(int(n_axial_slices)) \
                           + ',' + str(len(axial_fg_slices)) + ',' + str(len(axial_bg_slices)) + '\n')
    axial_slices_info_dict[PID] = {'Lesion':axial_fg_slices, 'NonLesion': axial_bg_slices}
    
    
#     # # Generate coronal slices
#     # n_coronal_slices = utils.count_slices(path, cross_section='coronal')
#     # coronal_ls_slices, coronal_bg_slices = utils.select_slices(path, cross_section='coronal')
#     # coronal_info_file.write(Disease + ',' + Center + ',' + PID + ',' + CID +  ',' \
#     #                       + str(int(n_coronal_slices)) + ',' + str(len(coronal_ls_slices)) + ',' + str(len(coronal_bg_slices)) + '\n')
#     # coronal_slices_info_dict[(PID, CID)] = {'Lesion':coronal_ls_slices, 'NonLesion': coronal_bg_slices}
    
    
#     # # # Generate sagittal slices
#     # n_sagittal_slices = utils.count_slices(path, cross_section='sagittal')
#     # sagittal_ls_slices, sagittal_bg_slices = utils.select_slices(path, cross_section='sagittal')
#     # sagittal_info_file.write(Disease + ',' + Center + ',' + PID + ',' + CID +  ',' \
#     #                       + str(int(n_sagittal_slices)) + ',' + str(len(sagittal_ls_slices)) + ',' + str(len(sagittal_bg_slices)) + '\n')
#     # sagittal_slices_info_dict[(PID, CID)] = {'Lesion':sagittal_ls_slices, 'NonLesion': sagittal_bg_slices}
    
 
#     # # # save selected AXIAL slices into lesion folder this 3D niftii file
    utils.save_selected_slices_hecktor(array3dgt, PID, 'fg', axial_fg_slices, dest_folder=GT_FG_SLICE_DIR, cross_section='axial')
    utils.save_selected_slices_hecktor(array3dgt, PID, 'bg', axial_bg_slices, dest_folder=GT_BG_SLICE_DIR, cross_section='axial')
    utils.save_selected_slices_hecktor(array3dpt, PID, 'fg', axial_fg_slices, dest_folder=PT_FG_SLICE_DIR, cross_section='axial')
    utils.save_selected_slices_hecktor(array3dpt, PID, 'bg', axial_bg_slices, dest_folder=PT_BG_SLICE_DIR, cross_section='axial')
    utils.save_selected_slices_hecktor(array3dct, PID, 'fg', axial_fg_slices, dest_folder=CT_FG_SLICE_DIR, cross_section='axial')
    utils.save_selected_slices_hecktor(array3dct, PID, 'bg', axial_bg_slices, dest_folder=CT_BG_SLICE_DIR, cross_section='axial')
    
#     # # # save selected CORONAL slices into lesion folder this 3D niftii file
#     # utils.save_selected_slices(array3d, Disease, Center, PID, CID, coronal_ls_slices,\
#     #                             dest_folder=CORONAL_DIR.joinpath('ls_gt'), cross_section='coronal')
#     # # #utils.save_selected_slices(array3d, Disease, Center, PID, CID, coronal_bg_slices, dest_folder=CORONAL_DIR.joinpath('bg'), cross_section='coronal')
    
#     # # # # save selected SAGITTAL slices into lesion folder this 3D niftii file
#     # utils.save_selected_slices(array3d, Disease, Center, PID, CID, sagittal_ls_slices,\
#     #                             dest_folder=SAGITTAL_DIR.joinpath('ls_gt'), cross_section='sagittal')
#     # # #utils.save_selected_slices(array3d, Disease, Center, PID, CID, sagittal_bg_slices, dest_folder=SAGITTAL_DIR.joinpath('bg'), cross_section='sagittal')
    
    print('Done with patient {}'.format(PID)) 


# # closing the relevant MASK information files
# saving slices info dictionary, will be used while processing PET images since 
# the same slice ID needs to be used for lesion or non-lesion slices folders
# np.save(CORONAL_DIR.joinpath('coronal_slices_info_dict.npy'), coronal_slices_info_dict)
# np.save(SAGITTAL_DIR.joinpath('sagittal_slices_info_dict.npy'), sagittal_slices_info_dict)
axial_info_file.close()
np.save(os.path.join(AXIAL_DATA_DIR, 'axial_slices_info_dict.npy'), axial_slices_info_dict)

# # coronal_info_file.close()
# # sagittal_info_file.close()

#%%

###################################################################################################
###################################################################################################
##########################  processing 3D niftii images of PET/CT   ##################################
###################################################################################################
###################################################################################################
#%%
# processing 3D niftii images of MASK
# generates the PET_info.csv, PET_slices_count.csv files
# uses the slices count dictionary
# saves the lesion and non-lesion slices in the respective folders in .nii format
# print("\nProcessing 3D PET/CT files\n")




# for i in range(len(PT_3D_paths)):
#     ptpath = PT_3D_paths[i]
#     ctpath = CT_3D_paths[i]

#     fileID = os.path.basename(ptpath)[:-7]
    
#     PID, CID, Type = fileID.split('_')


#     # get 3D array in numpy format 
#     array3d_pt, voxdim_pt = utils.nib2numpy(ptpath) 
#     array3d_ct, voxdim_ct = utils.nib2numpy(ctpath) 
   
    
#     # get AXIAL lesion and non-lesion slices IDs from AXIAL the slice info dictionary
#     axial_fg_slices = axial_slices_info_dict[(PID, CID)]['Lesion']
#     axial_bg_slices = axial_slices_info_dict[(PID, CID)]['NonLesion']
    
#     # get CORONAL lesion and non-lesion slices IDs from CORONAL the slice info dictionary
#     # coronal_ls_slices = coronal_slices_info_dict[(PID, CID)]['Lesion']
#     # coronal_bg_slices = coronal_slices_info_dict[(PID, CID)]['NonLesion']
    
#     # # get CORONAL lesion and non-lesion slices IDs from CORONAL the slice info dictionary
#     # sagittal_ls_slices = sagittal_slices_info_dict[(PID, CID)]['Lesion']
#     # sagittal_bg_slices = sagittal_slices_info_dict[(PID, CID)]['NonLesion']



#     # # save selected AXIAL slices into lesion folder this 3D niftii file
#     utils.save_selected_slices(array3d_pt, PID, CID, 'fg', axial_fg_slices, dest_folder=PT_FG_SLICE_DIR, cross_section='axial')
#     utils.save_selected_slices(array3d_pt, PID, CID, 'bg', axial_bg_slices, dest_folder=PT_BG_SLICE_DIR, cross_section='axial')
#     utils.save_selected_slices(array3d_ct, PID, CID, 'fg', axial_fg_slices, dest_folder=CT_FG_SLICE_DIR, cross_section='axial')
#     utils.save_selected_slices(array3d_ct, PID, CID, 'bg', axial_bg_slices, dest_folder=CT_BG_SLICE_DIR, cross_section='axial')
#     # # # save selected CORONAL slices into lesion folder this 3D niftii file
#     # utils.save_selected_slices(array3d, Disease, Center, PID, CID, coronal_ls_slices,\
#     #                             dest_folder=CORONAL_DIR.joinpath('ls_pt'), cross_section='coronal')
#     # utils.save_selected_slices(array3d, Disease, Center, PID, CID, coronal_bg_slices,\
#     #                             dest_folder=CORONAL_DIR.joinpath('bg_pt'), cross_section='coronal')
    
#     # # # # save selected SAGITTAL slices into lesion folder this 3D niftii file
#     # utils.save_selected_slices(array3d, Disease, Center, PID, CID, sagittal_ls_slices,\
#     #                             dest_folder=SAGITTAL_DIR.joinpath('ls_pt'), cross_section='sagittal')
#     # utils.save_selected_slices(array3d, Disease, Center, PID, CID, sagittal_bg_slices,\
#     #                             dest_folder=SAGITTAL_DIR.joinpath('bg_pt'), cross_section='sagittal')
    
        
#     print('---PTCT: Done with patient {}, case {}'.format(PID, CID))
#%%

###################################################################################################
###################################################################################################
##########   Generating lesion slices bounding boxes on MASK foreground slices   ##################
###################################################################################################
###################################################################################################

# gets bounding boxes for lesion slices
#%%
# print("\nGenerating lesion slices bounding boxes on MASK foreground slices\n")



# # opening csv files for storing labels info, one file for both bg and ls
# axial_box_fname = 'axial_fg_boxes_test.csv'
# axial_box_fg_file = open(os.path.join(AXIAL_DATA_DIR, axial_box_fname), 'w+')
# # coronal_box_ls_file = open(CORONAL_DIR.joinpath('coronal_box_ls.csv'), 'w+')
# # sagittal_box_ls_file = open(SAGITTAL_DIR.joinpath('sagittal_box_ls.csv'), 'w+')

# axial_box_fg_file.write("ImageID,LabelName,XMin,XMax,YMin,YMax,ImageH,ImageW\n")
# # coronal_box_ls_file.write("ImageID,LabelName,XMin,XMax,YMin,YMax,ImageH,ImageW\n")
# # sagittal_box_ls_file.write("ImageID,LabelName,XMin,XMax,YMin,YMax,ImageH,ImageW\n")


# # bounding box for AXIAL foreground slices
# axial_gt_fg_paths =  sorted(glob.glob(os.path.join(GT_FG_SLICE_DIR, '*.nii.gz')))

# #%%
# for path in axial_gt_fg_paths:
#     imageID = os.path.basename(path)[0:-7]
#     labelname = 'tumor'
#     image_shape = utils.get_shape(path)
#     boxes = pmbg.get_lesion_boxes(path)
#     nlesions = len(boxes)
#     # pmbg.plot_lesion_boxes(path, boxes, saveimage=False)
#     for i in range(nlesions):
#         axial_box_fg_file.write(imageID + ',' + labelname + ',' +\
#                                 str(boxes[i][0]) + ',' + str(boxes[i][1]) + ',' +\
#                                 str(boxes[i][2]) + ',' + str(boxes[i][3]) + ',' +\
#                                 str(image_shape[0]) +',' + str(image_shape[1]) +'\n')
    
#     print("Done with patient ID: ", imageID)

# print('Done with all bounding boxes')
# axial_box_fg_file.close()        
        
#%%
# # # bounding box for CORONAL foreground slices
# coronal_gt_ls_niftii =  sorted(list(CORONAL_DIR.joinpath('ls_gt').rglob("*.nii")))

# for path in coronal_gt_ls_niftii:
#     imageID = os.path.basename(path)[0:-4]
#     labelname = 'tumor'
#     image_shape = utils.get_shape(path)
#     boxes = pmbg.get_lesion_boxes(path)
#     nlesions = len(boxes)
#     # pmbg.plot_lesion_boxes(path, boxes, saveimage=False)
#     for i in range(nlesions):
#         coronal_box_ls_file.write(imageID + ',' + labelname + ',' +\
#                                   str(boxes[i][0]) + ',' + str(boxes[i][1]) + ',' +\
#                                   str(boxes[i][2]) + ',' + str(boxes[i][3]) + ',' +\
#                                   str(image_shape[0]) +',' + str(image_shape[1]) +'\n')
 
# coronal_box_ls_file.close()


# # # bounding box for SAGITTAL foreground slices
# sagittal_gt_ls_niftii =  sorted(list(SAGITTAL_DIR.joinpath('ls_gt').rglob("*.nii")))

# for path in sagittal_gt_ls_niftii:
#     imageID = os.path.basename(path)[0:-4]
#     labelname = 'tumor'
#     image_shape = utils.get_shape(path)
#     boxes = pmbg.get_lesion_boxes(path)
#     nlesions = len(boxes)
#     # pmbg.plot_lesion_boxes(path, boxes, saveimage=False)
#     for i in range(nlesions):
#         sagittal_box_ls_file.write(imageID + ',' + labelname + ',' +\
#                                     str(boxes[i][0]) + ',' + str(boxes[i][1]) + ',' +\
#                                     str(boxes[i][2]) + ',' + str(boxes[i][3]) + ',' +\
#                                     str(image_shape[0]) +',' + str(image_shape[1]) +'\n')
        
# sagittal_box_ls_file.close()



###################################################################################################
###################################################################################################
###############  Plotting bounding boxes on MASK and PET foreground slices  #######################
###################################################################################################
###################################################################################################



# checking the boxes by plotting them on MASK and PET foreground slices
# print("\nPlotting bounding boxes on MASK and PET foreground slices side-by-side (this is a testing step to verify if boxes were correctly made\n")


# axial_box_ls_df = pd.read_csv(AXIAL_DIR.joinpath('axial_box_ls.csv'))
# # coronal_box_ls_df = pd.read_csv(CORONAL_DIR.joinpath('coronal_box_ls.csv'))
# # sagittal_box_ls_df = pd.read_csv(SAGITTAL_DIR.joinpath('sagittal_box_ls.csv'))


# # bounding box plotting for AXIAL MASK and PET slices
# axial_pt_ls_niftii =  sorted(list(AXIAL_DIR.joinpath('ls_pt').rglob("*.nii")))

# axial_paths = np.column_stack((axial_gt_ls_niftii, axial_pt_ls_niftii))

# for path in axial_paths:
#     path_gt, path_pt = path
#     imageID = os.path.basename(path_gt)[0:-4]
#     df = axial_box_ls_df[axial_box_ls_df['ImageID'] == imageID]
#     boxes = df['XMin,XMax,YMin,YMax'.split(',')].values
#     pmbg.plot_pet_mask_boxes(path_gt, boxes, path_pt, boxes, saveimage=True,\
#                              savefolder=AXIAL_DIR.joinpath('ls_gt_pt_box_img'),\
#                              cross_section = 'axial')
    
    
    
 
    
# # bounding box plotting for CORONAL MASK and PET slices
# coronal_pt_ls_niftii =  sorted(list(CORONAL_DIR.joinpath('ls_pt').rglob("*.nii")))
# coronal_paths = np.column_stack((coronal_gt_ls_niftii, coronal_pt_ls_niftii))

# for path in coronal_paths:
#     path_gt, path_pt = path
#     imageID = os.path.basename(path_gt)[0:-4]
#     df = coronal_box_ls_df[coronal_box_ls_df['ImageID'] == imageID]
#     boxes = df['XMin,XMax,YMin,YMax'.split(',')].values
#     pmbg.plot_pet_mask_boxes(path_gt, boxes, path_pt, boxes, saveimage=True, \
#                              savefolder=CORONAL_DIR.joinpath('ls_gt_pt_box_img'),\
#                              cross_section = 'coronal')
    

# # bounding box for SAGITTAL MASK and PET slices
# sagittal_pt_ls_niftii =  sorted(list(SAGITTAL_DIR.joinpath('ls_pt').rglob("*.nii")))
# sagittal_paths = np.column_stack((sagittal_gt_ls_niftii, sagittal_pt_ls_niftii))

# for path in sagittal_paths:
#     path_gt, path_pt = path
#     imageID = os.path.basename(path_gt)[0:-4]
#     df = sagittal_box_ls_df[sagittal_box_ls_df['ImageID'] == imageID]
#     boxes = df['XMin,XMax,YMin,YMax'.split(',')].values
#     pmbg.plot_pet_mask_boxes(path_gt, boxes, path_pt, boxes, saveimage=True,\
#                              savefolder=SAGITTAL_DIR.joinpath('ls_gt_pt_box_img'),\
#                              cross_section = 'sagittal')


###################################################################################################
###################################################################################################
##########  get bounding box distributions for foreground slices and plot histograms   ############
###################################################################################################
###################################################################################################


# get bounding box distributions for foreground slices and plot histograms
# get the mu and sigma distribution for boxes on foreground slices for X, Y, W, H of the box
# via Gaussian fit. I will later modify this if the distribution is NOT Gaussian

# AXIAL slice distribution 
# XCenter_ax, YCenter_ax, Wbox_ax, Hbox_ax = pmbg.get_lesion_boxes_distribution(axial_box_ls_df)
# axial_mu_x, axial_sigma_x, axial_mu_y, axial_sigma_y, axial_mu_w, axial_sigma_w, axial_mu_h, axial_sigma_h \
#     = pmbg.plot_lesion_boxes_distribution(XCenter_ax, YCenter_ax, Wbox_ax, Hbox_ax,\
#                                           cross_section='axial', slice_type='foreground',\
#                                           saveimage=True, savefolder=AXIAL_DIR)


# pmbg.plot_lesion_boxes_distribution(XCenter_ax, YCenter_ax, Wbox_ax, Hbox_ax, binsize=50, cross_section='axial', slice_type='foreground',saveimage=True, savefolder=AXIAL_DIR, savename='lesion_box_distribution.png')
# mu_ax_xcenter, sigma_ax_xcenter = scipy.stats.norm.fit(XCenter_ax)


# # CORONAL slice distribution
# XCenter_co, YCenter_co, Wbox_co, Hbox_co = pmbg.get_lesion_boxes_distribution(coronal_box_ls_df)
# coronal_mu_x, coronal_sigma_x, coronal_mu_y, coronal_sigma_y, coronal_mu_w, coronal_sigma_w, coronal_mu_h, coronal_sigma_h \
#     = pmbg.plot_lesion_boxes_distribution(XCenter_co, YCenter_co, Wbox_co, Hbox_co,\
#                                           cross_section='coronal', slice_type='foreground',\
#                                           saveimage=True, savefolder=CORONAL_DIR)


# # SAGITTAL slice distribtion
# XCenter_sg, YCenter_sg, Wbox_sg, Hbox_sg = pmbg.get_lesion_boxes_distribution(sagittal_box_ls_df)
# sagittal_mu_x, sagittal_sigma_x, sagittal_mu_y, sagittal_sigma_y, sagittal_mu_w, sagittal_sigma_w, sagittal_mu_h, sagittal_sigma_h \
#     = pmbg.plot_lesion_boxes_distribution(XCenter_sg, YCenter_sg, Wbox_sg, Hbox_sg,\
#                                           cross_section='sagittal', slice_type='foreground',\
#                                           saveimage=True, savefolder=SAGITTAL_DIR)




###################################################################################################
###################################################################################################
###############  Generating bounding boxes for background slices  ###########
###################################################################################################
###################################################################################################
# print("Generating bounding boxes for background slices\n")

# # opening csv files for storing labels info, one file for both bg and ls
# axial_box_bg_whole_file = open(AXIAL_DIR.joinpath('axial_box_bg_whole.csv'), 'w+')
# axial_box_bg_uniform_file = open(AXIAL_DIR.joinpath('axial_box_bg_uniform.csv'), 'w+')

# # coronal_box_bg_uniform_file = open(CORONAL_DIR.joinpath('coronal_box_bg_uniform.csv'), 'w+')
# # sagittal_box_bg_uniform_file = open(SAGITTAL_DIR.joinpath('sagittal_box_bg_uniform.csv'), 'w+')

# axial_box_bg_whole_file.write("ImageID,LabelName,XMin,XMax,YMin,YMax,ImageH,ImageW\n")
# axial_box_bg_uniform_file.write("ImageID,LabelName,XMin,XMax,YMin,YMax,ImageH,ImageW\n")

# # coronal_box_bg_uniform_file.write("ImageID,LabelName,XMin,XMax,YMin,YMax,ImageH,ImageW\n")
# # sagittal_box_bg_uniform_file.write("ImageID,LabelName,XMin,XMax,YMin,YMax,ImageH,ImageW\n")

# # bounding box for AXIAL background slices (uniform dist)
# axial_pt_bg_niftii =  sorted(list(AXIAL_DIR.joinpath('bg_pt').rglob("*.nii")))

# # print("Generating uniformly distributed bounding boxes for AXIAL background slices\n")

# for path in axial_pt_bg_niftii:
#     imageID = os.path.basename(path)[0:-4]
#     labelname = 'bg'
#     image_shape = utils.get_shape(path)
    
#     # generate whole box
#     boxes_whole = np.array([[0.05, 0.95, 0.05, 0.95]])
#     for i in range(len(boxes_whole)):
#         axial_box_bg_whole_file.write(imageID + ',' + labelname + ',' +\
#                                         str(boxes_whole[i][0]) + ',' + str(boxes_whole[i][1]) + ',' +\
#                                         str(boxes_whole[i][2]) + ',' + str(boxes_whole[i][3]) + ',' +\
#                                         str(image_shape[0]) +',' + str(image_shape[1]) +'\n')
       
    
#     # generate uniform boxes
#     boxes_uf = pmbg.get_background_boxes_coordinates(distribution_type='uniform')
#     # pmbg.plot_lesion_boxes(path, boxes, saveimage=False)
#     for i in range(len(boxes_uf)):
#         axial_box_bg_uniform_file.write(imageID + ',' + labelname + ',' +\
#                                         str(boxes_uf[i][0]) + ',' + str(boxes_uf[i][1]) + ',' +\
#                                         str(boxes_uf[i][2]) + ',' + str(boxes_uf[i][3]) + ',' +\
#                                         str(image_shape[0]) +',' + str(image_shape[1]) +'\n')
    
#     # boxes_ldist = pmbg.get_background_boxes_coordinates(distribution_type='from_lesion_boxes', lesion_distribution_file='')
    
#     # generate lesion distributed boxes
# axial_box_bg_whole_file.close()
# axial_box_bg_uniform_file.close()
# print("Done with AXIAL background slices\n")









# import pet_mask_box_generator as pmbg


# axial_box_bg_ldist_file = open(AXIAL_DIR.joinpath('axial_box_bg_ldist.csv'), 'w+')
# axial_box_bg_ldist_file.write("ImageID,LabelName,XMin,XMax,YMin,YMax,ImageH,ImageW\n")

# for path in axial_pt_bg_niftii:
#     imageID = os.path.basename(path)[0:-4]
#     labelname = 'bg'
#     image_shape = utils.get_shape(path)
    
   
#     # generate uniform boxes
#     boxes_ldist = pmbg.get_background_boxes_coordinates(distribution_type='ldist', lesion_dist_file=AXIAL_DIR.joinpath('axial_box_ls.csv') , binsize=100)
#     # pmbg.plot_lesion_boxes(path, boxes_ldist, saveimage=False)
#     for i in range(len(boxes_ldist)):
#         axial_box_bg_ldist_file.write(imageID + ',' + labelname + ',' +\
#                                         str(boxes_ldist[i][0]) + ',' + str(boxes_ldist[i][1]) + ',' +\
#                                         str(boxes_ldist[i][2]) + ',' + str(boxes_ldist[i][3]) + ',' +\
#                                         str(image_shape[0]) +',' + str(image_shape[1]) +'\n')
    
#     # boxes_ldist = pmbg.get_background_boxes_coordinates(distribution_type='from_lesion_boxes', lesion_distribution_file='')
    
#     # generate lesion distributed boxes
# axial_box_bg_ldist_file.close()
# print("Done with AXIAL background slices\n")

###################################################################################################
###################################################################################################
###############  Generating uniformly distributed bounding boxes for background slices  ###########
###################################################################################################
###################################################################################################

## create random boxes for background slices from the distribution U(0,1)

# print("Generating uniformly distributed bounding boxes for background slices\n")

# opening csv files for storing labels info, one file for both bg and ls
# axial_box_bg_uniform_file = open(AXIAL_DIR.joinpath('axial_box_bg_uniform.csv'), 'w+')
# coronal_box_bg_uniform_file = open(CORONAL_DIR.joinpath('coronal_box_bg_uniform.csv'), 'w+')
# sagittal_box_bg_uniform_file = open(SAGITTAL_DIR.joinpath('sagittal_box_bg_uniform.csv'), 'w+')

# axial_box_bg_uniform_file.write("ImageID,LabelName,XMin,XMax,YMin,YMax,ImageH,ImageW\n")
# coronal_box_bg_uniform_file.write("ImageID,LabelName,XMin,XMax,YMin,YMax,ImageH,ImageW\n")
# sagittal_box_bg_uniform_file.write("ImageID,LabelName,XMin,XMax,YMin,YMax,ImageH,ImageW\n")

# bounding box for AXIAL background slices (uniform dist)
# axial_pt_bg_niftii =  sorted(list(AXIAL_DIR.joinpath('bg_pt').rglob("*.nii")))

# print("Generating uniformly distributed bounding boxes for AXIAL background slices\n")

# for path in axial_pt_bg_niftii:
#     imageID = os.path.basename(path)[0:-4]
#     labelname = 'bg'
#     image_shape = utils.get_shape(path)
#     boxes = pmbg.get_background_boxes_coordinates(distribution_type='uniform')
#     # pmbg.plot_lesion_boxes(path, boxes, saveimage=False)
#     for i in range(len(boxes)):
#         axial_box_bg_uniform_file.write(imageID + ',' + labelname + ',' +\
#                                         str(boxes[i][0]) + ',' + str(boxes[i][1]) + ',' +\
#                                         str(boxes[i][2]) + ',' + str(boxes[i][3]) + ',' +\
#                                         str(image_shape[0]) +',' + str(image_shape[1]) +'\n')

# axial_box_bg_uniform_file.close()
# print("Done with AXIAL background slices\n")


# # bounding box for CORONAL background slices (uniform dist)
# coronal_pt_bg_niftii =  sorted(list(CORONAL_DIR.joinpath('bg_pt').rglob("*.nii")))


# print("Generating uniformly distributed bounding boxes for CORONAL background slices\n")

# for path in coronal_pt_bg_niftii:
#     imageID = os.path.basename(path)[0:-4]
#     labelname = 'bg'
#     image_shape = utils.get_shape(path)
#     boxes = pmbg.get_background_boxes_coordinates(distribution_type='uniform')
#     # pmbg.plot_lesion_boxes(path, boxes, saveimage=False)
#     for i in range(len(boxes)):
#         coronal_box_bg_uniform_file.write(imageID + ',' + labelname + ',' +\
#                                           str(boxes[i][0]) + ',' + str(boxes[i][1]) + ',' +\
#                                           str(boxes[i][2]) + ',' + str(boxes[i][3]) + ',' +\
#                                           str(image_shape[0]) +',' + str(image_shape[1]) +'\n')

# coronal_box_bg_uniform_file.close()
# print("Done with CORONAL background slices\n")

# # bounding box for SAGITTAL background slices (uniform dist)
# sagittal_pt_bg_niftii =  sorted(list(SAGITTAL_DIR.joinpath('bg_pt').rglob("*.nii")))

# print("Generating uniformly distributed bounding boxes for SAGITTAL background slices\n")

# for path in sagittal_pt_bg_niftii:
#     imageID = os.path.basename(path)[0:-4]
#     labelname = 'bg'
#     image_shape = utils.get_shape(path)
#     boxes = pmbg.get_background_boxes_coordinates(distribution_type='uniform')
#     # pmbg.plot_lesion_boxes(path, boxes, saveimage=False)
#     for i in range(len(boxes)):
#         sagittal_box_bg_uniform_file.write(imageID + ',' + labelname + ',' +\
#                                            str(boxes[i][0]) + ',' + str(boxes[i][1]) + ',' +\
#                                            str(boxes[i][2]) + ',' + str(boxes[i][3]) + ',' +\
#                                            str(image_shape[0]) +',' + str(image_shape[1]) +'\n')
 
# sagittal_box_bg_uniform_file.close()
# print("Done with SAGITTAL background slices\n")



###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################

# finding distributions and plotting histograms of the uniformly produced bounding boxes on 
# the background slices in the previous step

# axial_box_bg_uniform_df = pd.read_csv(AXIAL_DIR.joinpath('axial_box_bg_uniform.csv'))
# # coronal_box_bg_uniform_df = pd.read_csv(CORONAL_DIR.joinpath('coronal_box_bg_uniform.csv'))
# # sagittal_box_bg_uniform_df = pd.read_csv(SAGITTAL_DIR.joinpath('sagittal_box_bg_uniform.csv'))

# # AXIAL slice distribution 
# bg_XCenter_ax, bg_YCenter_ax, bg_Wbox_ax, bg_Hbox_ax = pmbg.get_lesion_boxes_distribution(axial_box_bg_uniform_df)
# bg_axial_mu_x, bg_axial_sigma_x, bg_axial_mu_y, bg_axial_sigma_y, bg_axial_mu_w, bg_axial_sigma_w, bg_axial_mu_h, bg_axial_sigma_h \
#     = pmbg.plot_lesion_boxes_distribution(bg_XCenter_ax, bg_YCenter_ax, bg_Wbox_ax, bg_Hbox_ax,\
#                                           cross_section='axial', slice_type='background',\
#                                           saveimage=True, savefolder=AXIAL_DIR)
# # mu_ax_xcenter, sigma_ax_xcenter = scipy.stats.norm.fit(XCenter_ax)
# pmbg.plot_lesion_boxes_distribution(bg_XCenter_ax, bg_YCenter_ax, bg_Wbox_ax, bg_Hbox_ax,cross_section='axial', slice_type='background', saveimage=True, savefolder=AXIAL_DIR)

# # CORONAL slice distribution
# bg_XCenter_co, bg_YCenter_co, bg_Wbox_co, bg_Hbox_co = pmbg.get_lesion_boxes_distribution(coronal_box_bg_uniform_df)
# bg_coronal_mu_x, bg_coronal_sigma_x, bg_coronal_mu_y, bg_coronal_sigma_y, bg_coronal_mu_w, bg_coronal_sigma_w, bg_coronal_mu_h, bg_coronal_sigma_h \
#     = pmbg.plot_lesion_boxes_distribution(bg_XCenter_co, bg_YCenter_co, bg_Wbox_co, bg_Hbox_co,\
#                                           cross_section='coronal', slice_type='background',\
#                                           saveimage=True, savefolder=CORONAL_DIR)


# # SAGITTAL slice distribtion
# bg_XCenter_sg, bg_YCenter_sg, bg_Wbox_sg, bg_Hbox_sg = pmbg.get_lesion_boxes_distribution(sagittal_box_bg_uniform_df)
# bg_sagittal_mu_x, bg_sagittal_sigma_x, bg_sagittal_mu_y, bg_sagittal_sigma_y, bg_sagittal_mu_w, bg_sagittal_sigma_w, bg_sagittal_mu_h, bg_sagittal_sigma_h \
#     = pmbg.plot_lesion_boxes_distribution(bg_XCenter_sg, bg_YCenter_sg, bg_Wbox_sg, bg_Hbox_sg,\
#                                           cross_section='sagittal', slice_type='background',\
#                                           saveimage=True, savefolder=SAGITTAL_DIR)





###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################

## create random boxes for background slices from the distribution U(a,b)

# print("Generating lesion distributed bounding boxes for background slices\n")

# # opening csv files for storing labels info, one file for both bg and ls
# axial_box_bg_lsdist_file = open(AXIAL_DIR.joinpath('axial_box_bg_lsdist.csv'), 'w+')
# # coronal_box_bg_lsdist_file = open(CORONAL_DIR.joinpath('coronal_box_bg_lsdist.csv'), 'w+')
# # sagittal_box_bg_lsdist_file = open(SAGITTAL_DIR.joinpath('sagittal_box_bg_lsdist.csv'), 'w+')

# axial_box_bg_lsdist_file.write("ImageID,LabelName,XMin,XMax,YMin,YMax,ImageH,ImageW\n")
# # coronal_box_bg_lsdist_file.write("ImageID,LabelName,XMin,XMax,YMin,YMax,ImageH,ImageW\n")
# # sagittal_box_bg_lsdist_file.write("ImageID,LabelName,XMin,XMax,YMin,YMax,ImageH,ImageW\n")

# # bounding box for AXIAL background slices (lsdist)
# print("Generating lesion distributed bounding boxes for AXIAL background slices\n")

# for path in axial_pt_bg_niftii:
#     imageID = os.path.basename(path)[0:-4]
#     labelname = 'bg'
#     image_shape = utils.get_shape(path)
#     boxes = pmbg.get_background_boxes_coordinates(distribution_type='from_lesion_boxes',\
#                                                   mu_x = axial_mu_x, sigma_x = axial_sigma_x,\
#                                                   mu_y = axial_mu_y, sigma_y = axial_sigma_y,\
#                                                   mu_w = axial_mu_w, sigma_w = axial_sigma_w,\
#                                                   mu_h = axial_mu_h, sigma_h = axial_sigma_h)
#     pmbg.plot_lesion_boxes(path, boxes, saveimage=False)
#     for i in range(len(boxes)):
#         axial_box_bg_lsdist_file.write(imageID + ',' + labelname + ',' +\
#                                        str(boxes[i][0]) + ',' + str(boxes[i][1]) + ',' +\
#                                        str(boxes[i][2]) + ',' + str(boxes[i][3]) + ',' +\
#                                        str(image_shape[0]) +',' + str(image_shape[1]) +'\n')

# axial_box_bg_lsdist_file.close()
# print("Done with AXIAL background slices\n")


# bounding box for CORONAL background slices (lsdist)

# print("Generating lesion distributed bounding boxes for CORONAL background slices\n")

# for path in coronal_pt_bg_niftii:
#     imageID = os.path.basename(path)[0:-4]
#     labelname = 'bg'
#     image_shape = utils.get_shape(path)
#     boxes = pmbg.get_background_boxes_coordinates(distribution_type='from_lesion_boxes',\
#                                                   mu_x = coronal_mu_x, sigma_x = coronal_sigma_x,\
#                                                   mu_y = coronal_mu_y, sigma_y = coronal_sigma_y,\
#                                                   mu_w = coronal_mu_w, sigma_w = coronal_sigma_w,\
#                                                   mu_h = coronal_mu_h, sigma_h = coronal_sigma_h)
#     # pmbg.plot_lesion_boxes(path, boxes, saveimage=False)
#     for i in range(len(boxes)):
#         coronal_box_bg_lsdist_file.write(imageID + ',' + labelname + ',' +\
#                                          str(boxes[i][0]) + ',' + str(boxes[i][1]) + ',' +\
#                                          str(boxes[i][2]) + ',' + str(boxes[i][3]) + ',' +\
#                                          str(image_shape[0]) +',' + str(image_shape[1]) +'\n')

# coronal_box_bg_lsdist_file.close()
# print("Done with CORONAL background slices\n")

# # bounding box for SAGITTAL background slices (lsdist)

# print("Generating uniformly distributed bounding boxes for SAGITTAL background slices\n")

# for path in sagittal_pt_bg_niftii:
#     imageID = os.path.basename(path)[0:-4]
#     labelname = 'bg'
#     image_shape = utils.get_shape(path)
#     boxes = pmbg.get_background_boxes_coordinates(distribution_type='from_lesion_boxes',\
#                                                   mu_x = sagittal_mu_x, sigma_x = sagittal_sigma_x,\
#                                                   mu_y = sagittal_mu_y, sigma_y = sagittal_sigma_y,\
#                                                   mu_w = sagittal_mu_w, sigma_w = sagittal_sigma_w,\
#                                                   mu_h = sagittal_mu_h, sigma_h = sagittal_sigma_h)
#     # pmbg.plot_lesion_boxes(path, boxes, saveimage=False)
#     for i in range(len(boxes)):
#         sagittal_box_bg_lsdist_file.write(imageID + ',' + labelname + ',' +\
#                                           str(boxes[i][0]) + ',' + str(boxes[i][1]) + ',' +\
#                                           str(boxes[i][2]) + ',' + str(boxes[i][3]) + ',' +\
#                                           str(image_shape[0]) +',' + str(image_shape[1]) +'\n')
 
# sagittal_box_bg_lsdist_file.close()
# print("Done with SAGITTAL background slices\n")



###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################

# finding distributions and plotting histograms of the bounding boxes produced from on 
# the background slices in the previous step

# axial_box_bg_lsdist_df = pd.read_csv(AXIAL_DIR.joinpath('axial_box_bg_lsdist.csv'))
# # coronal_box_bg_lsdist_df = pd.read_csv(CORONAL_DIR.joinpath('coronal_box_bg_lsdist.csv'))
# # sagittal_box_bg_lsdist_df = pd.read_csv(SAGITTAL_DIR.joinpath('sagittal_box_bg_lsdist.csv'))

# # AXIAL slice distribution 
# bglsdist_XCenter_ax, bglsdist_YCenter_ax, bglsdist_Wbox_ax, bglsdist_Hbox_ax = pmbg.get_lesion_boxes_distribution(axial_box_bg_lsdist_df)
# bglsdist_axial_mu_x, bglsdist_axial_sigma_x, bglsdist_axial_mu_y, bglsdist_axial_sigma_y, bglsdist_axial_mu_w, bglsdist_axial_sigma_w, bglsdist_axial_mu_h, bglsdist_axial_sigma_h \
#     = pmbg.plot_lesion_boxes_distribution(bglsdist_XCenter_ax, bglsdist_YCenter_ax, bglsdist_Wbox_ax, bglsdist_Hbox_ax,\
#                                           cross_section='axial', slice_type='background', saveimage=True, savefolder=AXIAL_DIR)
# # mu_ax_xcenter, sigma_ax_xcenter = scipy.stats.norm.fit(XCenter_ax)
# pmbg.plot_lesion_boxes_distribution(bglsdist_XCenter_ax, bglsdist_YCenter_ax, bglsdist_Wbox_ax, bglsdist_Hbox_ax, cross_section='axial', slice_type='background', saveimage=True, savefolder=AXIAL_DIR)

# # CORONAL slice distribution
# bglsdist_XCenter_co, bglsdist_YCenter_co, bglsdist_Wbox_co, bglsdist_Hbox_co = pmbg.get_lesion_boxes_distribution(coronal_box_bg_lsdist_df)
# bglsdist_coronal_mu_x, bglsdist_coronal_sigma_x, bglsdist_coronal_mu_y, bglsdist_coronal_sigma_y, bglsdist_coronal_mu_w, bglsdist_coronal_sigma_w, bglsdist_coronal_mu_h, bglsdist_coronal_sigma_h \
#     = pmbg.plot_lesion_boxes_distribution(bglsdist_XCenter_co, bglsdist_YCenter_co, bglsdist_Wbox_co, bglsdist_Hbox_co,\
#                                           cross_section='coronal', slice_type='background', saveimage=True, savefolder=CORONAL_DIR)


# # SAGITTAL slice distribtion
# bglsdist_XCenter_sg, bglsdist_YCenter_sg, bglsdist_Wbox_sg, bglsdist_Hbox_sg = pmbg.get_lesion_boxes_distribution(sagittal_box_bg_lsdist_df)
# bglsdist_sagittal_mu_x, bglsdist_sagittal_sigma_x, bglsdist_sagittal_mu_y, bglsdist_sagittal_sigma_y, bglsdist_sagittal_mu_w, bglsdist_sagittal_sigma_w, bglsdist_sagittal_mu_h, bglsdist_sagittal_sigma_h \
#     = pmbg.plot_lesion_boxes_distribution(bglsdist_XCenter_sg, bglsdist_YCenter_sg, bglsdist_Wbox_sg, bglsdist_Hbox_sg,\
#                                           cross_section='sagittal', slice_type='background', saveimage=True, savefolder=SAGITTAL_DIR)



###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################



  
# # checking the boxes by plotting them on PET foreground slices
# print("\nPlotting bounding boxes on PET foreground slices side-by-side (this is a testing step to verify if boxes were correctly made\n")


# # opening csv files for storing labels info, one file for both bg and ls
# # axial_box_ls_file = open(AXIAL_DIR.joinpath('axial_box_ls.csv'), 'r+')
# # coronal_box_ls_file = open(CORONAL_DIR.joinpath('coronal_box_ls.csv'), 'r+')
# # sagittal_box_ls_file = open(SAGITTAL_DIR.joinpath('sagittal_box_ls.csv'), 'r+')

# axial_box_ls_df = pd.read_csv(AXIAL_DIR.joinpath('axial_box_ls.csv'))
# coronal_box_ls_df = pd.read_csv(CORONAL_DIR.joinpath('coronal_box_ls.csv'))
# sagittal_box_ls_df = pd.read_csv(SAGITTAL_DIR.joinpath('sagittal_box_ls.csv'))



# # axial_box_ls_file.write("ImageID,LabelName,XMin,XMax,YMin,YMax,ImageH,ImageW\n")
# # coronal_box_ls_file.write("ImageID,LabelName,XMin,XMax,YMin,YMax,ImageH,ImageW\n")
# # sagittal_box_ls_file.write("ImageID,LabelName,XMin,XMax,YMin,YMax,ImageH,ImageW\n")


# # bounding box for AXIAL PET slices
# axial_pt_ls_niftii =  sorted(list(AXIAL_DIR.joinpath('ls_pt').rglob("*.nii")))



# for path in axial_pt_ls_niftii:
#     imageID = os.path.basename(path)[0:-4]
#     df = axial_box_ls_df[axial_box_ls_df['ImageID'] == imageID]
#     boxes = df['XMin,XMax,YMin,YMax'.split(',')].values
#     pmbg.plot_lesion_boxes(path, boxes)
    

#     # nlesions = len(boxes)
#     # pmbg.plot_lesion_boxes(path, boxes, saveimage=False)
#     # for i in range(nlesions):
#     #     axial_box_ls_file.write(imageID + ',' + 'tumor' + ',' + str(boxes[i][0]) + ',' + str(boxes[i][1]) + ',' + str(boxes[i][2]) + ',' + str(boxes[i][3]) + ',' + str(image_shape[0]) +',' + str(image_shape[1]) +'\n')
        
        
# # bounding box for CORONAL slices
# coronal_pt_ls_niftii =  sorted(list(CORONAL_DIR.joinpath('ls_pt').rglob("*.nii")))

# for path in coronal_pt_ls_niftii:
#     imageID = os.path.basename(path)[0:-4]
#     df = coronal_box_ls_df[coronal_box_ls_df['ImageID'] == imageID]
#     boxes = df['XMin,XMax,YMin,YMax'.split(',')].values
#     pmbg.plot_lesion_boxes(path, boxes)
    
#     # boxes = pmbg.get_lesion_boxes(path)
#     # nlesions = len(boxes)
#     # pmbg.plot_lesion_boxes(path, boxes, saveimage=False)
#     # for i in range(nlesions):
#     #     coronal_box_ls_file.write(imageID + ',' + 'tumor' + ',' + str(boxes[i][0]) + ',' + str(boxes[i][1]) + ',' + str(boxes[i][2]) + ',' + str(boxes[i][3]) + ',' + str(image_shape[0]) +',' + str(image_shape[1]) +'\n')
 

# # bounding box for SAGITTAL slices
# sagittal_pt_ls_niftii =  sorted(list(SAGITTAL_DIR.joinpath('ls_pt').rglob("*.nii")))

# for path in sagittal_pt_ls_niftii:
#     imageID = os.path.basename(path)[0:-4]
#     df = sagittal_box_ls_df[sagittal_box_ls_df['ImageID'] == imageID]
#     boxes = df['XMin,XMax,YMin,YMax'.split(',')].values
#     pmbg.plot_lesion_boxes(path, boxes)
    
    
    
    # image_shape = utils.get_shape(path)
    # boxes = pmbg.get_lesion_boxes(path)
    # nlesions = len(boxes)
    # pmbg.plot_lesion_boxes(path, boxes, saveimage=False)
    # for i in range(nlesions):
    #     sagittal_box_ls_file.write(imageID + ',' + 'tumor' + ',' + str(boxes[i][0]) + ',' + str(boxes[i][1]) + ',' + str(boxes[i][2]) + ',' + str(boxes[i][3]) + ',' + str(image_shape[0]) +',' + str(image_shape[1]) +'\n')
        
# axial_box_ls_file.close()
# coronal_box_ls_file.close()
# sagittal_box_ls_file.close()





# # gets bounding boxes for non-lesion slices (generated randomly in two ways)
# # gets ground truth images (lesion slice with box overlayed)
 
# print("\nGenerating non-lesion slices bounding boxes\n")





# ###################################################################################################
# ###################################################################################################
# ###################################################################################################
# ###################################################################################################
# ###################################################################################################


# # copy lesion bounding boxes annotation files from MASK to PET folder
# print("\nCopying lesion bounding boxes annotation files from MASK to PET folder\n")
# MASK_GTBOX = sorted(glob.glob(config.AX_LS_GTBOX_MASK +'*.txt'))

# for path in MASK_GTBOX:
#     shutil.copy(path, config.AX_LS_GTBOX_PET)


###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################


# # performing z-score normalization of non-zero pixels of 2D PET image slices 
# # The slices are in .nii format in the AX_LS_PET or AX_BG_PET folders
# # The normed slices are saved in the AX_LS_N_PET or AX_BG_N_PET folders
# print("\nPerforming z-score normalization on 2D PET slices\n")

# PET_2D_LS_niftii = sorted(glob.glob(config.AX_LS_PET+'*.nii'))
# PET_2D_BG_niftii = sorted(glob.glob(config.AX_BG_PET+'*.nii'))

# # saving the normed lesion slices as nii
# for path in PET_2D_LS_niftii:
#     utils.z_score_norm_nonzero_pixel(path, config.AX_LS_N_PET)

# # saving the normed non-lesion slices as nii
# for path in PET_2D_BG_niftii:
#     utils.z_score_norm_nonzero_pixel(path, config.AX_BG_N_PET)
    

###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################


# # converting normed PET slices to jpg format 
# # The slices are in .nii format in the AX_LS_N_PET or AX_BG_N_PET folders
# # The jpg slices are saved in the AX_LS_N_JPG_PET or AX_BG_N_JPG_PET folders
# print("\nConverting 2D PET slices to jpg format\n")

# PET_2D_LS_N_niftii = sorted(glob.glob(config.AX_LS_N_PET+'*.nii'))
# PET_2D_BG_N_niftii = sorted(glob.glob(config.AX_BG_N_PET+'*.nii'))

# # saving the normed lesion slices as jpg
# for path in PET_2D_LS_N_niftii:
#     utils.nii2jpg(path, config.AX_LS_N_JPG_PET)

# # saving the normed non-lesion slices as jpg
# for path in PET_2D_BG_N_niftii:
#     utils.nii2jpg(path, config.AX_BG_N_JPG_PET)


###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################

# saving PET images to final data (IMAGES_DIR) folder

# LS_IMGS = sorted(glob.glob(config.AX_LS_N_JPG_PET+'*.jpg'))
# BG_IMGS = sorted(glob.glob(config.AX_BG_N_JPG_PET+'*.jpg'))
# LS_LBLS = sorted(glob.glob(config.AX_LS_GTBOX_PET+'*.txt'))
# BG_LBLS = sorted(glob.glob(config.AX_BG_GTBOX_PET+'*.txt'))


# for path in LS_IMGS:
#     try:
#         shutil.copy(path, config.IMAGES_DIR)
#     except:
#         pass
    
# for path in BG_IMGS:
#     try:
#         shutil.copy(path, config.IMAGES_DIR)
#     except:
#         pass
    
# for path in LS_LBLS:
#     try:
#         shutil.copy(path, config.LABELS_DIR) 
#     except: 
#         pass   

# for path in BG_LBLS:
#     try:
#         shutil.copy(path, config.LABELS_DIR)
#     except:
#         pass


###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################



















    

# %%
