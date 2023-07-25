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

# TODO: CHECK X VS Y AND CONSISTENTCY THROUGHOUT
# 0 dim = y dim = height dim 
# 1 dim = x dim = width dim 

# random blob coordinates generation
def rand_x_y(image_dimensions):
    return [random.randint(0, image_dimensions[0]-1), random.randint(0, image_dimensions[1]-1)]

# 2. Generating images with a varying noise for peak values as specified in the 'noise_peak_values' tensor
# output: a vector of 4 tensors [noised_images, pre_noised_images, noised_images_indices_peaks, noised_images_indices_blobs]
# 2a. 
def varying_peak_values_apply(original_images, peak_values):
    # TODO: comment indicating what each dimension is
    assert original_images.ndim==4 
    assert peak_values.ndim==1
    
    # making an appropriate number of copies of each series of images with differet amount of blobs
    new_original_images = original_images.unsqueeze(0).expand(*peak_values.shape,*original_images.shape)
    peak_values = peak_values.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(new_original_images.shape)
    
    new_original_images = new_original_images * peak_values
    
    return new_original_images

# 2b.
def background_offset_apply(original_images, background_offset):
    new_original_images = original_images + background_offset
    return new_original_images

# 2c.
def gaussian_beam_apply(original_images):
    #1. guassian beam mask based on 256x256 coordinates
    x = np.linspace(-0.6, 0.6, 256)
    y = np.linspace(-0.6, 0.6, 256)
    xx, yy = np.meshgrid(x, y)
    zz = np.exp(-(xx**2 + yy**2))

    mask = torch.tensor(zz)
    mask = mask.expand(original_images.shape)
    #2. overlaying the mask somehow over the image 
    new_original_images = torch.mul(mask, original_images)
    return new_original_images

# 2d.
def noise_apply(original_images):
    noisy_images = torch.poisson(original_images) 
    return noisy_images
