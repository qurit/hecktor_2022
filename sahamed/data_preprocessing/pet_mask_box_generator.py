#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 12:36:13 2021

@author: shadab
"""


import numpy as np
import matplotlib.pyplot as plt
import nrrd as nrrd
import glob
import os
import matplotlib.patches as patches
import matplotlib.image as mpimg
import nibabel as nib
import utils as utils
import cc3d
global disconnected_islands
from pathlib import Path
import matplotlib.gridspec as gridspec
import scipy
import pandas as pd


# from lmfit.models import SkewedGaussianModel
# from lmfit.models import GaussianModel


# get coordinates of all the lesions on 2D slice as a dict
def get_lesion_coordinates(mask):
    # works for multiple lesions on 2D slice
    mask_out, n_island2d = cc3d.connected_components(mask, connectivity=8, return_N = True) 
    
    nonzero_elements = np.nonzero(mask_out)
    lesion_stack = np.vstack((nonzero_elements, mask_out[nonzero_elements]))
    
    lesion_dict = {i+1 : [] for i in range(n_island2d)}
    
    # putting the n_island2d lesion coordinates in a dictionary where the
    # coordinates of each lesion is stores under a key referenced by lesion number
    for i in range(len(lesion_stack[0])):
        lesion_dict[lesion_stack[2][i]].append((lesion_stack[1][i], lesion_stack[0][i]))
    
    return lesion_dict



# get box coordinates for a particular lesion given the coordinates of the lesion
# Format: XMin, XMax, YMin, YMax normalized to image dimension
def get_lesion_boxes_coordinates(imagedim, lesion_coordinates):
    
    x_coords = [i[0] for i in lesion_coordinates]
    y_coords = [i[1] for i in lesion_coordinates]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    
    center_x = (x_min + x_max)/2
    center_y = (y_min + y_max)/2
    
    if x_max - x_min <= 1:
        width = 3
    else:
        width = (x_max - x_min) + 3
   
    if y_max - y_min <= 1:
        height = 3
    else:
        height = (y_max - y_min) + 3
    
    x_min_new = center_x - width/2
    y_min_new = center_y - height/2
    x_max_new = center_x + width/2
    y_max_new = center_y + height/2
    
    XMin = x_min_new/imagedim[1]
    XMax = x_max_new/imagedim[1]
    YMin = y_min_new/imagedim[0]
    YMax = y_max_new/imagedim[0]
    
    box_coords = np.array([XMin, XMax, YMin, YMax])
    return box_coords
    
    

# ground truth bounding box generation for a slice with lesions
def get_lesion_boxes(path):
    imageID = os.path.basename(path)[:-3]
    mask, voxdim = utils.nib2numpy(path)
    
    lesion_dict = get_lesion_coordinates(mask)
 
    nlesions = len(lesion_dict) 
    
    box_list = np.zeros((nlesions, 4))
        
    
    for i in range(nlesions):
         box_list[i][0], box_list[i][1],box_list[i][2],box_list[i][3] =  get_lesion_boxes_coordinates(mask.shape, lesion_dict[i+1])
    
    return box_list




# plots the boxes around all lesions given a 2D slice filepath and all the box list
# in the format XMin, XMax, YMin, YMax
def plot_lesion_boxes(path, box_list, CM='gray', saveimage=False, savefolder=''):
    mask, voxdim = utils.nib2numpy(path)
    fileID = os.path.basename(path)[:-4]
    #mask = mpimg.imread(path)
    fig, ax = plt.subplots()
    scale = 1
    im = ax.imshow(scale*mask, cmap=CM, origin = 'upper')
    # plt.colorbar(im)
    ax.axis('off')
 
    
    for i in range(len(box_list)):
       
        XMin, XMax, YMin, YMax = box_list[i]
        x_min = XMin*mask.shape[1]
        y_min = YMin*mask.shape[0]
        x_max = XMax*mask.shape[1]
        y_max = YMax*mask.shape[0]
        
        height = y_max - y_min
        width = x_max - x_min
        
        rect = patches.Rectangle((x_min, y_min), width, height, facecolor='none', edgecolor='red', lw=2)
        ax.add_patch(rect)
    
    if saveimage == True:
        os.makedirs(savefolder, exist_ok=True)
        plt.savefig(savefolder.joinpath(fileID + '.jpg'), bbox_inches='tight', pad_inches=0.05, dpi=1200)
    plt.show()


# plot the mask and pet images side by side with boxes around overlaid
def plot_pet_mask_boxes(path_gt, boxes_gt, path_pt, boxes_pt, saveimage=False, savefolder='', cross_section = 'axial'):
    mask_gt, voxdim_gt = utils.nib2numpy(path_gt)
    fileID_gt = os.path.basename(path_gt)[:-4]
    
    mask_pt, voxdim_pt = utils.nib2numpy(path_pt)
    fileID_pt = os.path.basename(path_pt)[:-4]
    
    # fig, ax = plt.subplots(1,2)
    # fig = plt.gcf()
    
    fig = plt.figure(1)
    gs1 = gridspec.GridSpec(1, 2)
    ax = [fig.add_subplot(ss) for ss in gs1]
    
    ax[0].imshow(mask_gt, cmap = 'gray', origin = 'upper')
    ax[1].imshow(mask_pt, cmap = 'gray', origin = 'upper')
    ax[0].axis('off')
    ax[1].axis('off')

    for i in range(len(boxes_gt)):
       
        XMin, XMax, YMin, YMax = boxes_gt[i]
        x_min = XMin*mask_gt.shape[1]
        y_min = YMin*mask_gt.shape[0]
        x_max = XMax*mask_gt.shape[1]
        y_max = YMax*mask_gt.shape[0]
        
        height = y_max - y_min
        width = x_max - x_min
        
        rect = patches.Rectangle((x_min, y_min), width, height, facecolor='none', edgecolor='red', lw=1)
        ax[0].add_patch(rect)
    
    for i in range(len(boxes_pt)):
       
        XMin, XMax, YMin, YMax = boxes_pt[i]
        x_min = XMin*mask_pt.shape[1]
        y_min = YMin*mask_pt.shape[0]
        x_max = XMax*mask_pt.shape[1]
        y_max = YMax*mask_pt.shape[0]
        
        height = y_max - y_min
        width = x_max - x_min
        
        rect = patches.Rectangle((x_min, y_min), width, height, facecolor='none', edgecolor='red', lw=1)
        ax[1].add_patch(rect)
    
    
    fig.suptitle(fileID_gt, fontsize=12)
    gs1.tight_layout(fig, rect=[0, 0.03, 1, 0.95], h_pad = 1, w_pad=0.6)  

    # if cross_section == 'axial':
    #     plt.subplots_adjust(top=0.95, wspace=0.05)
    # elif cross_section == 'coronal':
    #     plt.subplots_adjust(top=0.90, wspace=0.05)
    # else:
    #     plt.subplots_adjust(top=0.90, wspace=0.05)
    ax[0].set_title('MASK ground truth', fontsize=10)
    ax[1].set_title('PET ground truth', fontsize=10)
    
    if saveimage == True:
        os.makedirs(savefolder, exist_ok=True)
        plt.savefig(savefolder.joinpath(fileID_gt + '.jpg'), bbox_inches='tight', pad_inches=0.05, dpi=1200)
    plt.show()




# get distribution of coordinates (XCenter, YCenter, W, H) of lesion boxes
# and plot histogram
def get_lesion_boxes_distribution(df):
    # df = pd.read_csv(csvfilepath)
    all_boxes = df['XMin,XMax,YMin,YMax'.split(',')]
    XCenter = (all_boxes['XMax'] + all_boxes['XMin'])/2
    YCenter = (all_boxes['YMax'] + all_boxes['YMin'])/2
    Wbox = (all_boxes['XMax'] - all_boxes['XMin'])
    Hbox = (all_boxes['YMax'] - all_boxes['YMin'])

    return XCenter, YCenter, Wbox, Hbox


def plot_lesion_boxes_distribution(XCenter, YCenter, Wbox, Hbox, binsize=50, cross_section='axial', slice_type='foreground', saveimage=True, savefolder='', savename=''):
    fig, ax = plt.subplots(1,2)
    
    fig.suptitle('Tumor boxes histogram for ' + cross_section + ' ' + slice_type + ' slices')
    # plt.ylabel('Frequency')
    # plt.xlabel("Normalized box coordinates")
    
    _, bins_x, _ = ax[0].hist(XCenter, bins=binsize)
    _, bins_y, _ = ax[0].hist(YCenter, bins=binsize)
    ax[0].set_xlabel("Normalized center coordinates")
    ax[0].set_ylabel("Frequency")
    
    
    
    _, bins_w, _ = ax[1].hist(Wbox, bins=binsize)
    _, bins_h, _ = ax[1].hist(Hbox, bins=binsize)
    ax[1].set_xlabel("Normalized box dimensions")
    # ax[1].set_ylabel("Frequency")
    
    # try:
    #     mu_x, sigma_x, fit_x = get_gaussian_fit(XCenter, bins_x)
    #     mu_y, sigma_y, fit_y = get_gaussian_fit(YCenter, bins_y)
    #     mu_w, sigma_w, fit_w = get_skewedgaussian_fit(Wbox, bins_w)
    #     mu_h, sigma_h, fit_h = get_skewedgaussian_fit(Hbox, bins_h)
    # except:
    #     pass
    # ax[0].plot(bins_x, fit_x, color='dodgerblue')
    # ax[0].plot(bins_y, fit_y, color='chocolate')
    
    # ax[1].plot(bins_w, fit_w, color='dodgerblue')
    # ax[1].plot(bins_h, fit_h, color='chocolate')
    
    ax[0].legend([ 'X', 'Y'], loc='best', fontsize=6)
    ax[1].legend(['Width', 'Height'], loc='best', fontsize=6)
    
    #ax[0].legend(['X-fit ' + str((np.round(mu_x,2), np.round(sigma_x,2))), 'Y-fit ' + str((np.round(mu_y,2), np.round(sigma_y,2))), 'X', 'Y'], loc='best', fontsize=6)
    #ax[1].legend(['Width-fit '+ str((np.round(mu_w,2), np.round(sigma_w,2))), 'Height-fit ' + str((np.round(mu_h,2), np.round(sigma_h,2))),'Width', 'Height'], loc='best', fontsize=6)
    
    if saveimage == True:
        os.makedirs(savefolder, exist_ok=True)
        plt.savefig(savefolder.joinpath(savename), bbox_inches='tight', pad_inches=0.05, dpi=1200)
        print(savefolder.joinpath(savename))
    plt.show()
    
    # return mu_x, sigma_x, mu_y, sigma_y, mu_w, sigma_w, mu_h, sigma_h




def gaussian(data, A, mu, sigma):
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))


    
# def gauss(x, *p):
#     A, mu, sigma = p
#     return A*numpy.exp(-(x-mu)**2/(2.*sigma**2))


# # get a best fitting gaussian curve to the histogram data
# def get_gaussian_fit(data, bins):
#     mu, sigma = scipy.stats.norm.fit(data)
#     best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)
#     # plt.plot(bins, best_fit_line)
#     return mu, sigma, best_fit_line

# def get_gaussian_fit(function, bins, data, ):
#     model = GaussianModel()

#     params = model.guess(data, x=bins)
#     result = model.fit(data, params, x=bins)
#     return result
    
    
# def skewed_gaussian_fit(bins, data, amp = 10, loc = 0, sig = 1, gam = 0):
#     model = SkewedGaussianModel()

#     # set initial parameter values
#     params = model.make_params(amplitude=amp, center=loc, sigma=sig, gamma=gam)

#     # adjust parameters  to best fit data.
#     result = model.fit(data, params, x=bins)
#     best_fit = result.best_fit
#     return best_fit
    

# plot randomly generated boxes for the background PET slices
# generates the XCenter, YCenter, W and H (normalized to image width and height)
# either uniformly randomly between (0,1) or uniformly randomly between (a, b)
# where a and b are the limits obtained from the distribution of box parameters 
# from the foreground slices

def get_background_boxes_coordinates(distribution_type='uniform', lesion_dist_file='axial_box_ls_.csv', binsize=100): #, mu_x=0.0, sigma_x=1.0, mu_y=0.0, sigma_y=1.0, mu_w=0.0, sigma_w=1.0, mu_h=0.0, sigma_h=1.0):
    # mask2d, voxdim = utils.nib2numpy(path)
    epsilon = 0.02
    if distribution_type == 'uniform':
        XCenter = np.random.uniform(0.05,0.95)
        YCenter = np.random.uniform(0.05,0.95)
        if XCenter < 0.5:
            Wbox = np.random.uniform(epsilon,2*XCenter)
        else:
            Wbox = np.random.uniform(epsilon, 2*(1-XCenter))
        
        if YCenter < 0.5:
            Hbox = np.random.uniform(epsilon,1.99*YCenter)
        else:
            Hbox = np.random.uniform(epsilon,1.99*(1-YCenter))
        
    elif distribution_type == 'ldist':
        df = pd.read_csv(lesion_dist_file)
        XCenter_array, YCenter_array, Wbox_array, Hbox_array = get_lesion_boxes_distribution(df)
        
        
        # Making XCenter histogram
        histX, binsX = np.histogram(XCenter_array, bins=binsize)
        binX_midpoints = (binsX[:-1] + binsX[1:])/2
        cdfX = np.cumsum(histX)
        cdfX = cdfX/cdfX[-1]
        
        # Making YCenter histogram
        histY, binsY = np.histogram(YCenter_array, bins=binsize)
        binY_midpoints = (binsY[:-1] + binsY[1:])/2
        cdfY = np.cumsum(histY)
        cdfY = cdfY/cdfY[-1]
        
        # Making Wbox histogram
        histW, binsW = np.histogram(Wbox_array, bins=binsize)
        binW_midpoints = (binsW[:-1] + binsW[1:])/2
        cdfW = np.cumsum(histW)
        cdfW = cdfW/cdfW[-1]
        
        
        # Making Hbox histogram
        histH, binsH = np.histogram(Hbox_array, bins=binsize)
        binH_midpoints = (binsH[:-1] + binsH[1:])/2
        cdfH = np.cumsum(histH)
        cdfH = cdfH/cdfH[-1]
        
        # choosing XCenter
        randvalX = np.random.rand(1)
        value_binsX = np.searchsorted(cdfX, randvalX)
        XCenter = binX_midpoints[value_binsX][0]
        

        # choosing YCenter
        randvalY = np.random.rand(1)
        value_binsY = np.searchsorted(cdfY, randvalY)
        YCenter = binY_midpoints[value_binsY][0]

        # choosing Wbox
        randvalW = np.random.rand(1)
        value_binsW = np.searchsorted(cdfW, randvalW)
        Wbox = binW_midpoints[value_binsW][0]

        # choosing Hbox
        randvalH = np.random.rand(1)
        value_binsH = np.searchsorted(cdfH, randvalH)
        Hbox = binH_midpoints[value_binsH][0]
        

        
    else:
        pass
    
    XMin = XCenter - Wbox/2
    XMax = XCenter + Wbox/2
    YMin = YCenter - Hbox/2
    YMax = YCenter + Hbox/2
    
    return np.array([[XMin, XMax, YMin, YMax]])
    # return np.array([XCenter, YCenter, Wbox, Hbox])
    





# if __name__ == '__main__':
#     return
#     # pathmask = Path("C:/Users/sahamed/Desktop/Shadab/lymphoma_trial/processed_data/coronal/ls_gt/PMBCL_BC_00-13084_1_072_co.nii")
#     # boxes = get_lesion_boxes(pathmask)
#     # plot_lesion_boxes(pathmask, boxes, saveimage=True, savefolder = Path("C:/Users/sahamed/Desktop/Shadab/lymphoma_trial/processed_data/coronal/ls_gt_img/"))
    
#     # print(1)
#     # XCenter, YCenter, Wbox, Hbox = get_lesion_boxes_distribution(r'C:\Users\sahamed\Desktop\Shadab\lymphoma_trial\processed_data\axial\axial_box_ls.csv')
#     # plot_lesion_boxes_distribution(XCenter, YCenter, Wbox, Hbox)#r'C:\Users\sahamed\Desktop\Shadab\testingboxes\df.csv')
    
#     # XC = []
#     # YC = []
#     # W = []
#     # H = []
    
#     # XMin = []
#     # XMax = []
#     # YMin = []
#     # YMax = []
    
    
#     # # for i in range(130):    
#     # #     boxes = get_background_boxes_coordinates(distribution_type = 'from_lesion_boxes', distribution_boxes=np.array([XCenter, YCenter, Wbox, Hbox]))
#     # #     XC.append(boxes[0])
#     # #     YC.append(boxes[1])
#     # #     W.append(boxes[2])
#     #     H.append(boxes[3])
       
#     # plot_lesion_boxes_distribution(XC, YC, W, H)
    
