import torch
from torch import nn
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import random

import numpy as np

import matplotlib.cm as cm

import pickle

from denoising_assessment_project.global_vars import global_vars

from torchmetrics import PeakSignalNoiseRatio
import skimage.metrics as metrics
from skimage import io
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure

# 4a. NRMSE computing function 
# a function for calculating NRMSE for a pair of pictures (original and noised)
def NRMSE_compute(un_noised_images, noised_images): 
    error = torch.sub(un_noised_images, noised_images)
    SE = torch.pow(error,2)
    MSE = torch.mean(SE,dim=[3,4])
    RMSE = torch.pow(MSE,0.5)
    mean = torch.mean(un_noised_images, dim=[3,4])
    NRMSE = torch.div(RMSE, mean)
    return NRMSE

# 4b. RMSE 
# a function for calculating RMSE for a pair of pictures (original and noised)
def RMSE_compute(un_noised_images, noised_images): 
    error = un_noised_images - noised_images
    SE = torch.pow(error,2)
    MSE = torch.mean(SE,dim=[3,4])
    RMSE = torch.pow(MSE,0.5)
    return RMSE

# 4c. PSNR 
# function for calculating PSNR per pair of pictures (original and noised)
psnr = PeakSignalNoiseRatio()
# TODO: the torchmetrics's PeakSignalNoiseRatio function should be usable without for loops somehow, 
# there is the option to reduce the result at some selected dimensions (like with torch.mean(X, dim=[3,4]), 
# but then it requires specifying the data_range, which I do not understand what it is for). 
# I will look into it further, in the meantime, implementation using the for loops I wrote previosuly
def PSNR_compute(un_noised_images, noised_images):
    # computing the metric for each picture in each of the dim_0 x dim_1 data sets 
    dim_0_list = []
    for dim_0 in range(0, noised_images.shape[0]):
        dim_1_list = []
        for dim_1 in range(0, noised_images.shape[1]):
            dim_i_list = []
            for i in range(0, noised_images.shape[2]):
                # metric - un noised vs noised 
                a = psnr(un_noised_images[dim_0, dim_1, i, :, :], noised_images[dim_0, dim_1, i, :, :])
                dim_i_list.append(a)
            dim_i_tensor = torch.stack(dim_i_list)
            dim_1_list.append(dim_i_tensor)
        dim_1_tensor = torch.stack(dim_1_list)
        dim_0_list.append(dim_1_tensor)
    dim_0_tensor = torch.stack(dim_0_list)
    PSNR = dim_0_tensor
    # dim_0 - peaks, dim_1 - blobs, dim_3 - individual images
    return PSNR

# 4d. SSIM
def ssim(original_image, noised_image):
    # normalizing the pixel values 
    original_image = original_image/ original_image.max()
    noised_image = noised_image/ noised_image.max()
    
    mean_original = torch.mean(original_image)
    mean_noised = torch.mean(noised_image)
    sigma_original = torch.std(original_image)
    sigma_noised = torch.std(noised_image)
    
    C1 = 0.001
    C2 = 0.003
    
    SSIM = ((2*mean_original*mean_noised + C1)*(2*sigma_original*sigma_noised + C2))/((mean_noised**2 + mean_original**2 + C1)*(sigma_noised**2 + sigma_original**2 + C2))

    return SSIM

def SSIM_compute(un_noised_images, noised_images):
    # computing the metric for each picture in each of the dim_0 x dim_1 data sets 
    dim_0_list = []
    for dim_0 in range(0, noised_images.shape[0]):
        dim_1_list = []
        for dim_1 in range(0, noised_images.shape[1]):
            dim_i_list = []
            for i in range(0, noised_images.shape[2]):
                # metric - un noised vs noised 
                a = ssim(un_noised_images[dim_0, dim_1, i, :, :], noised_images[dim_0, dim_1, i, :, :])
                dim_i_list.append(a)
            dim_i_tensor = torch.stack(dim_i_list)
            dim_1_list.append(dim_i_tensor)
        dim_1_tensor = torch.stack(dim_1_list)
        dim_0_list.append(dim_1_tensor)
    dim_0_tensor = torch.stack(dim_0_list)
    SSIM = dim_0_tensor
    # dim_0 - peaks, dim_1 - blobs, dim_3 - individual images
    return SSIM

# 4e. MS_SSIM
# function for calculating MS_SSIM per pair of pictures 
ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(kernel_size=7,sigma = 1.5,data_range=1.0)

def ms_ssim(original_image, noised_image):
    # metric - original image vs one of the theree noised imaged 
    # adjusting the dimensions to those requested by the function (and analogically to the metrics notebook from Susan)
    original_image = original_image.unsqueeze(0).unsqueeze(0)
    noised_image = noised_image.unsqueeze(0).unsqueeze(0)

    # normalizing the pixel values 
    original_image = original_image/ original_image.max()
    noised_image = noised_image/ noised_image.max()

    # running the metric function
    MS_SSIM = ms_ssim(original_image, noised_image)
    
    return MS_SSIM

def MS_SSIM_compute(un_noised_images, noised_images):
    # computing the metric for each picture in each of the dim_0 x dim_1 data sets 
    dim_0_list = []
    for dim_0 in range(0, noised_images.shape[0]):
        dim_1_list = []
        for dim_1 in range(0, noised_images.shape[1]):
            dim_i_list = []
            for i in range(0, noised_images.shape[2]):
                # metric - un noised vs noised 
                a = ms_ssim(un_noised_images[dim_0, dim_1, i, :, :], noised_images[dim_0, dim_1, i, :, :])
                dim_i_list.append(a)
            dim_i_tensor = torch.stack(dim_i_list)
            dim_1_list.append(dim_i_tensor)
        dim_1_tensor = torch.stack(dim_1_list)
        dim_0_list.append(dim_1_tensor)
    dim_0_tensor = torch.stack(dim_0_list)
    MS_SSIM = dim_0_tensor
    # dim_0 - peaks, dim_1 - blobs, dim_3 - individual images
    return MS_SSIM
