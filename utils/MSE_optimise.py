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

def perform_percentile_based_image_normalisation(GT_images, p_low, p_high): 
    percentile_low = torch.quantile(GT_images, p_low, dim=[3,4])[:,:,:,0].unsqueeze(4).expand(GT_images)
    percentile_high = torch.quantile(GT_images, p_high, dim=[3,4])[:,:,:,1].unsqueeze(4).expand(GT_images)

    normalised_GT_images = (GT_images - percentile_low)/(percentile_high - percentile_low)

    return normalised_GT_images

# normalised_GT_images, noised_images - ought to be two extracted images from the array 
# return MSE per image
def MSE(normalised_GT_image, noised_image, alpha, beta):
    # how to make beta and alpha separate for each image? 
    alpha = alpha.unsqueeze(1).expand(noised_image)
    beta = beta.unsqueeze(1).expand(noised_image)

    # how to transofrm each image by its respective alpha and beta values 
    noised_image_t = torch.add(torch.mul(noised_image, alpha), beta)

    # standard MSE calculation 
    error = torch.sub(normalised_GT_image, noised_image)
    SE = torch.pow(error,2)
    MSE = torch.mean(SE)
    return MSE

# 4a. ALTERNATIVE NRMSE computing function 
# a function for calculating NRMSE for a pair of pictures (original and noised)
def ALTERNATIVE_NRMSE_compute(normalised_GT_images, noised_images):
    # 1. Normalise noised_images by tranforming them accoridng to 
    # the optimisation for MSE between the 'normalised_GT_images', 'noised_images': 
    # https://www.cl.cam.ac.uk/teaching/2021/LE49/probnn/3-3.pdf

    # 2. Calculate the MSE for the two normalised datasets 
    error = normalised_GT_images - normalised_noised_images
    SE = torch.pow(error,2)
    MSE = torch.mean(SE,dim=[3,4])
    RMSE = torch.pow(MSE,0.5)
    # as value normalisation has already been done:
    NRMSE = RMSE 
    return NRMSE