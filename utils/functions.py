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
                blobs_coordinates_i_list.append(rand_x_y(image_dimensions))
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

# 1. - EVAL type 2: Generating series (in the qunatity equal to 'series_quantity') of images, each of size max('my_range')+1, where each consecutive image in a series has got a single more randomly added blob relative to the previous image 
# output: [tensor]; shape = [series_quantity, blobs_population_sizes.shape[0], series_size.shape[0], image_dimensions[0], image_dimensions[1]]
def varying_blob_quantities_img_series_generate_plus_one(series_quantity, my_range, image_dimensions): 
    xinds = torch.arange(image_dimensions[1]).unsqueeze(0).expand(image_dimensions).to(global_vars.device)
    yinds = torch.arange(image_dimensions[0]).unsqueeze(1).expand(image_dimensions).to(global_vars.device)
    
    # spot size sigma
    sigma = 5

    # need to create xinds yinds pairs, squodged together across dimension 3
    coordinates_orig = torch.stack((xinds,yinds),2)

    # across dim 0 - different blob amounts - each image built based on the previous one
    # each series - differnet random configuration (each series independent of all other
    distinct_blob_quantities = torch.zeros(series_quantity,max(my_range).item()+1,image_dimensions[0],image_dimensions[0])
    
    for i in range(series_quantity):
        coordinates = torch.tensor(0)
        for idx in range(max(my_range)+1):
            coordinates_a = coordinates_orig.unsqueeze(0).expand(idx,*image_dimensions,2)
            single_coordinates = rand_x_y(image_dimensions)
            single_coordinates_tensor = torch.tensor(single_coordinates).to(global_vars.device).unsqueeze(0).expand(*image_dimensions,2)

            if idx == 0 or idx == 1:
                coordinates = single_coordinates_tensor
            elif idx == 2:
                coordinates = torch.cat((coordinates.unsqueeze(0), single_coordinates_tensor.unsqueeze(0)), dim=0)
            else: 
                coordinates = torch.cat((coordinates, single_coordinates_tensor.unsqueeze(0)))
            
            if idx == 0: 
                image = torch.zeros(*image_dimensions)
            else: 
                image = (torch.exp((-((coordinates - coordinates_a)**2).sum(3))/(2*sigma**2))).sum(0)
            
            # append a single image 
            distinct_blob_quantities[i,idx,:,:] = image
        
    return distinct_blob_quantities

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

# 3. - EVAL default: Generating indices 
def indices_generate(noised_images):
    blobs_indices = global_vars.blobs_population_sizes.unsqueeze(0).expand(noised_images[:,:,0,0,0].shape)
    peaks_indices = global_vars.peak_values.unsqueeze(1).expand(noised_images[:,:,0,0,0].shape)
    indices = torch.stack((blobs_indices, peaks_indices))
    return indices

# 3. - EVAL type 2: Generating indices
def indices_generate_plus_one(noised_images):
    blobs_indices = global_vars.blobs_population_sizes.unsqueeze(0).expand(noised_images[:,0,:,0,0].shape)
    peaks_indices = global_vars.peak_values.unsqueeze(1).expand(noised_images[:,0,:,0,0].shape)
    indices = torch.stack((blobs_indices, peaks_indices))
    return indices

# 5. a function computing for each set the mean and std of some metric scores for each set inputted 
# output: mean and std for the metric for each set 
def set_mean_std_compute(metric_output):
    mean = torch.mean(metric_output, dim=2)
    std = torch.std(metric_output, dim=2)
    stats = torch.stack((mean,std),dim=2)
    return stats

# torch.Size([1, 20, 51, 256, 256])
def set_mean_std_compute_t2(metric_output):
    mean = torch.mean(metric_output, dim=1)
    std = torch.std(metric_output, dim=1)
    stats = torch.stack((mean,std),dim=1)
    return stats

# 6. a function for plotting obtained statistics 
def plot_stats(metric_stats_data, indices, metric_name): 
    values = metric_stats_data[:,:,0]
    std = metric_stats_data[:,:,1]
    print(values.shape)

    # generating a plot with mean values for the metric for all datasets
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(indices[0,:,:], indices[1,:,:], values)
    ax.scatter(indices[0,:,:], indices[1,:,:], values+std, marker="*", c="red")
    ax.scatter(indices[0,:,:], indices[1,:,:],values-std, marker="*", c="red")
    ax.set_title('mean {}'.format(metric_name))
    ax.set_xlabel('number of blobs')
    ax.set_ylabel('value of peaks')
    ax.set_zlabel('mean value of {}'.format(metric_name))

    cmap = cm.get_cmap(name='rainbow')

    fig2 = plt.figure()
    for i in range(values.shape[1]):
        plt.errorbar(indices[1,:,0], values[:,i], yerr=std[:,i], color=cmap(i*15), fmt='o', capsize=5, markersize=2, elinewidth=1)
    plt.title('mean {} values - varying blob amounts [purple-3, red-288] \n background value = 10'.format(metric_name))
    plt.xlabel('peak value')
    plt.ylabel('mean {} value'.format(metric_name))

    fig3 = plt.figure()
    for i in range(values.shape[0]):
        plt.errorbar(indices[0,0,:], values[i,:], yerr=std[i,:], color=cmap(i*15), fmt='o', capsize=5, markersize=2, elinewidth=1)
    plt.title('mean {} values - varying peak values [purple-10, red-960] \n background value = 10'.format(metric_name))
    plt.xlabel('blob number')
    plt.ylabel('mean {} value'.format(metric_name))
    plt.show()
    return

# 6. EVAL t2 - a function for plotting obtained statistics 
def plot_stats_t2(metric_stats_data, indices, metric_name): 
    values = metric_stats_data[:,0,:]
    std = metric_stats_data[:,1,:]
    print(values.shape)

    cmap = cm.get_cmap(name='rainbow')

    fig = plt.figure()
    plt.errorbar(indices[0,0,:], values[0,:], yerr=std[0,:], color=cmap(15), fmt='o', capsize=5, markersize=2, elinewidth=1)
    plt.title('mean {} values - varying blob amounts [purple-3, red-288] \n background value = 10'.format(metric_name))
    plt.xlabel('peak value')
    plt.ylabel('mean {} value'.format(metric_name))
    plt.show()
    return

# TODO: DESCRIPTION 
def identify_BETTER_and_WORSE(noise_metrics_stats_data, metric_stats_data, metric): 
    values = metric_stats_data[:,:,0]
    std = metric_stats_data[:,:,1]
    noise_values = noise_metrics_stats_data[:,:,0]
    noise_std = noise_metrics_stats_data[:,:,1]
    if metric=='NRMSE' or metric=='RMSE':
        values_BETTER = torch.where(values < noise_values, values, 0)
        std_BETTER = torch.where(values < noise_values, std, 0)
        values_BETTER[values_BETTER==0] = np.nan
        std_BETTER[std_BETTER==0] = np.nan
        
        values_WORSE = torch.where(values >= noise_values, values, 0)
        std_WORSE = torch.where(values >= noise_values, std, 0)
        values_WORSE[values_WORSE==0] = np.nan
        std_WORSE[std_WORSE==0] = np.nan
    elif metric=='PSNR' or metric=='MS_SSIM' or metric=='SSIM': 
        values_BETTER = torch.where(values > noise_values, values, 0)
        std_BETTER = torch.where(values > noise_values, std, 0)
        values_BETTER[values_BETTER==0] = np.nan
        std_BETTER[std_BETTER==0] = np.nan
        
        values_WORSE = torch.where(values <= noise_values, values, 0)
        std_WORSE = torch.where(values <= noise_values, std, 0)
        values_WORSE[values_WORSE==0] = np.nan
        std_WORSE[std_WORSE==0] = np.nan
    else: 
        raise Exception("Unsupported metric")
    return [values_BETTER, std_BETTER, values_WORSE, std_WORSE]

# 6B. a function for plotting a network's metric statistics in comparison to GTvsNOISE statistics (or some other statistics for the same analysed space for the parameters of interest)
def plot_stats_NOISE_NETWORK_comparison(noise_metrics_stats_data, metric_stats_data, indices, metric_name):
    output = identify_BETTER_and_WORSE(noise_metrics_stats_data, metric_stats_data, metric_name)
    values_BETTER = output[0]
    std_BETTER = output[1]
    values_WORSE = output[2]
    std_WORSE = output[3]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(indices[0,:,:], indices[1,:,:], values_BETTER, c="green")
    ax.scatter(indices[0,:,:], indices[1,:,:], values_WORSE, c="red")
    ax.scatter(indices[0,:,:], indices[1,:,:], values_BETTER+std_BETTER, marker="*", c="lightgreen")
    ax.scatter(indices[0,:,:], indices[1,:,:],values_BETTER-std_BETTER, marker="*", c="lightgreen")
    ax.scatter(indices[0,:,:], indices[1,:,:], values_WORSE+std_WORSE, marker="*", c="pink")
    ax.scatter(indices[0,:,:], indices[1,:,:],values_WORSE-std_WORSE, marker="*", c="pink")
    ax.set_title('mean {}'.format(metric_name))
    ax.set_xlabel('number of blobs')
    ax.set_ylabel('value of peaks')
    ax.set_zlabel('mean value of {}'.format(metric_name))

    cmap = cm.get_cmap(name='rainbow')
    
    fig2 = plt.figure()
    for i in range(values_BETTER.shape[1]):
        plt.errorbar(indices[1,:,0], values_BETTER[:,i], yerr=std_BETTER[:,i], color=cmap(i*15), fmt='o', capsize=5, markersize=2, elinewidth=1)
    for i in range(values_WORSE.shape[1]):
        plt.errorbar(indices[1,:,0], values_WORSE[:,i], yerr=std_WORSE[:,i], color="gray", fmt='o', capsize=5, markersize=2, elinewidth=1)
    plt.title('mean {} values - varying blob amounts [purple-3, red-288] \n background value = 10'.format(metric_name))
    plt.xlabel('peak value')
    plt.ylabel('mean {} value'.format(metric_name))
    
    fig3 = plt.figure()
    for i in range(values_BETTER.shape[0]):
        plt.errorbar(indices[0,0,:], values_BETTER[i,:], yerr=std_BETTER[i,:], color=cmap(i*15), fmt='o', capsize=5, markersize=2, elinewidth=1)
    for i in range(values_WORSE.shape[0]):
        plt.errorbar(indices[0,0,:], values_WORSE[i,:], yerr=std_WORSE[i,:], color="gray", fmt='o', capsize=5, markersize=2, elinewidth=1)
    plt.title('mean {} values - varying peak values [purple-10, red-960] \n background value = 10'.format(metric_name))
    plt.xlabel('blob number')
    plt.ylabel('mean {} value'.format(metric_name))
    plt.show()
    return

# TODO: is it still needed? 
# 6. a function for plotting obtained statistics 
def plot_stats_single_blob(metric_stats_data, indices, metric_name): 
    cmap = cm.get_cmap(name='rainbow')

    values = metric_stats_data[:,:,0]
    std = metric_stats_data[:,:,1]
    
    fig = plt.figure()
    plt.errorbar(indices[1,:,0], values[:,19], yerr=std[:,19], color=cmap(i*15), fmt='o', capsize=5, markersize=2, elinewidth=1)
    plt.title('mean {} values - varying blob amounts [purple-3, red-288] \n background value = 10'.format(metric_name))
    plt.xlabel('peak value')
    plt.ylabel('mean {} value'.format(metric_name))
    return

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


# 4a. ALTERNATIVE NRMSE computing function 
# a function for calculating NRMSE for a pair of pictures (original and noised)
def NRMSE_compute_alternative(noised_images): 
    noised_images_50blobs = noised_images[:,:,51,:,:].expand(noised_images.shape)
    error = torch.sub(noised_images_50blobs, noised_images)
    SE = torch.pow(error,2)
    MSE = torch.mean(SE,dim=[3,4])
    RMSE = torch.pow(MSE,0.5)
    mean = torch.mean(un_noised_images, dim=[3,4])
    NRMSE = torch.div(RMSE, mean)
    return NRMSE

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

# 4b. RMSE 
# a function for calculating RMSE for a pair of pictures (original and noised)
def RMSE_compute(un_noised_images, noised_images): 
    error = un_noised_images - noised_images
    SE = torch.pow(error,2)
    MSE = torch.mean(SE,dim=[3,4])
    RMSE = torch.pow(MSE,0.5)
    return RMSE

# 4c. PSNR 
from torchmetrics import PeakSignalNoiseRatio
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
import skimage.metrics as metrics
from skimage import io

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
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure

# function for calculating MS_SSIM per pair of pictures 
ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(kernel_size=7,sigma = 1.5,data_range=1.0)

def my_ms_ssim(original_image, noised_image):
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
                a = my_ms_ssim(un_noised_images[dim_0, dim_1, i, :, :], noised_images[dim_0, dim_1, i, :, :])
                dim_i_list.append(a)
            dim_i_tensor = torch.stack(dim_i_list)
            dim_1_list.append(dim_i_tensor)
        dim_1_tensor = torch.stack(dim_1_list)
        dim_0_list.append(dim_1_tensor)
    dim_0_tensor = torch.stack(dim_0_list)
    MS_SSIM = dim_0_tensor
    # dim_0 - peaks, dim_1 - blobs, dim_3 - individual images
    return MS_SSIM
