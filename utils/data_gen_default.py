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

from denoising_assessment_project.utils import data_gen

# 1. - EVAL default: Generating series of images (of size 'repetition_number') with randomly located points in n different quantities as specified in the 'population_blobs'
# output: [tensor]; shape = [blobs_population_sizes.shape[0], series_size.shape[0], image_dimensions[0], image_dimensions[1]]
def varying_blob_quantities_img_series_generate(series_size, blobs_population_sizes, image_dimensions): 
    xinds = torch.arange(image_dimensions[1]).unsqueeze(0).expand(image_dimensions).to(global_vars.device)
    yinds = torch.arange(image_dimensions[0]).unsqueeze(1).expand(image_dimensions).to(global_vars.device)
    
    #spot size sigma
    sigma = 5

    # need to create xinds yinds pairs, squodged together across dimension 3
    coordinates_orig = torch.stack((xinds,yinds),2)
    
    distinct_blob_quantities = torch.zeros(blobs_population_sizes.shape[0],series_size,image_dimensions[0],image_dimensions[0])
    
    for idx in range(blobs_population_sizes.shape[0]):
        #make a container for coords per spot, number spots, rows, cols, number of indices
        coordinates = coordinates_orig.unsqueeze(0).expand(blobs_population_sizes[idx],*image_dimensions,2)
         
        for i in range(0, series_size):
            # generating random blob coordinates
            blobs_coordinates_i_list = []
            for blob in range(0,int(blobs_population_sizes[idx].item())): 
                blobs_coordinates_i_list.append(data_gen.rand_x_y(image_dimensions))
            blobs_coordinates_i_tensor = torch.tensor(blobs_coordinates_i_list).to(global_vars.device)

            # expand spot_pos so has same size as coords except for not being coords
            # this is creating xc and yc everywhere
            blobs_coordinates_i_tensor = blobs_coordinates_i_tensor.unsqueeze(1).unsqueeze(2).expand(*coordinates.shape)

            # subtract the two so you have x-xc and y-yc everywhere
            # sum across last dimension to get thing for exponent
            image = (torch.exp((-((blobs_coordinates_i_tensor - coordinates)**2).sum(3))/(2*sigma**2))).sum(0)
            # peak being almost the same as background 
            
            # append a single image 
            distinct_blob_quantities[idx,i,:,:] = image
        
    return distinct_blob_quantities

# 3. - EVAL default: Generating indices 
def indices_generate(noised_images):
    blobs_indices = global_vars.blobs_population_sizes.unsqueeze(0).expand(noised_images[:,:,0,0,0].shape)
    peaks_indices = global_vars.peak_values.unsqueeze(1).expand(noised_images[:,:,0,0,0].shape)
    indices = torch.stack((blobs_indices, peaks_indices))
    return indices