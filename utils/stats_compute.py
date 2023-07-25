import torch
from torch import nn
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import random

import numpy as np

import matplotlib.cm as cm

import pickle

#from denoising_assessment_project.global_vars import global_vars

# 5. - EVAL default: a function computing for each set the mean and std of some metric scores for each set inputted 
# output: mean and std for the metric for each set 
def set_mean_std_compute(metric_output):
    mean = torch.mean(metric_output, dim=2)
    std = torch.std(metric_output, dim=2)
    stats = torch.stack((mean,std),dim=2)
    return stats

# 5. - EVAL type 2: a function computing for each set the mean and std of some metric scores for each set inputted 
# output: mean and std for the metric for each set 
def set_mean_std_compute_t2(metric_output):
    mean = torch.mean(metric_output, dim=1)
    std = torch.std(metric_output, dim=1)
    stats = torch.stack((mean,std),dim=1)
    return stats
