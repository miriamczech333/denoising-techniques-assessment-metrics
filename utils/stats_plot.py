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

# 6. - EVAL default: a function for plotting obtained statistics 
def plot_stats(metric_stats_data, indices, metric_name): 
    values = metric_stats_data[:,:,0]
    std = metric_stats_data[:,:,1]

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
    bottom = indices[1,0,0]
    upper = indices[1,-1,0]
    for i in range(values.shape[1]):
        plt.errorbar(indices[1,:,0], values[:,i], yerr=std[:,i], color=cmap(i*15), fmt='o', capsize=5, markersize=2, elinewidth=1)
    plt.title('mean {} values - varying blob amounts [purple-{}, red-{}] \n background value = 10'.format(metric_name, bottom, upper))
    plt.xlabel('peak value')
    plt.ylabel('mean {} value'.format(metric_name))

    fig3 = plt.figure()
    bottom = indices[0,0,0]
    upper = indices[0,0,-1]
    for i in range(values.shape[0]):
        plt.errorbar(indices[0,0,:], values[i,:], yerr=std[i,:], color=cmap(i*15), fmt='o', capsize=5, markersize=2, elinewidth=1)
    plt.title('mean {} values - varying peak values [purple-{}, red-{}] \n background value = 10'.format(metric_name, bottom, upper))
    plt.xlabel('blob number')
    plt.ylabel('mean {} value'.format(metric_name))
    plt.show()
    return

# 6. - EVAL type 2: a function for plotting obtained statistics 
def plot_stats_t2(metric_stats_data, indices, metric_name): 
    values = metric_stats_data[:,0,:]
    std = metric_stats_data[:,1,:]

    cmap = cm.get_cmap(name='rainbow')
    
    fig = plt.figure()
    plt.errorbar(indices[0,0,:], values[0,:], yerr=std[0,:], color=cmap(15), fmt='o', capsize=5, markersize=2, elinewidth=1)
    plt.title('mean {} values - varying blob amounts [purple-3, red-288] \n background value = 10'.format(metric_name))
    plt.xlabel('peak value')
    plt.ylabel('mean {} value'.format(metric_name))
    plt.show()
    return

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
    bottom = indices[1,0,0]
    upper = indices[1,-1,0]
    for i in range(values_BETTER.shape[1]):
        plt.errorbar(indices[1,:,0], values_BETTER[:,i], yerr=std_BETTER[:,i], color=cmap(i*15), fmt='o', capsize=5, markersize=2, elinewidth=1)
    for i in range(values_WORSE.shape[1]):
        plt.errorbar(indices[1,:,0], values_WORSE[:,i], yerr=std_WORSE[:,i], color="gray", fmt='o', capsize=5, markersize=2, elinewidth=1)
    plt.title('mean {} values - varying blob amounts [purple-{}, red-{}] \n background value = 10'.format(metric_name, bottom, upper))
    plt.xlabel('peak value')
    plt.ylabel('mean {} value'.format(metric_name))
    
    fig3 = plt.figure()
    bottom = indices[0,0,0]
    upper = indices[0,0,-1]
    for i in range(values_BETTER.shape[0]):
        plt.errorbar(indices[0,0,:], values_BETTER[i,:], yerr=std_BETTER[i,:], color=cmap(i*15), fmt='o', capsize=5, markersize=2, elinewidth=1)
    for i in range(values_WORSE.shape[0]):
        plt.errorbar(indices[0,0,:], values_WORSE[i,:], yerr=std_WORSE[i,:], color="gray", fmt='o', capsize=5, markersize=2, elinewidth=1)
    plt.title('mean {} values - varying peak values [purple-{}, red-{}] \n background value = 10'.format(metric_name, bottom, upper))
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