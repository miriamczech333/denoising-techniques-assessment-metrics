import torch

import numpy as np

from denoising_assessment_project.global_vars import global_vars

# TODO: DESCRIPTION 
def identify_BETTER_and_WORSE_NaN(noise_metrics_stats_data, metric_stats_data, metric): 
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

# TODO: description 
def identify_BETTER_and_WORSE_zero(noise_metrics_stats_data, metric_stats_data, metric): 
    values = metric_stats_data[:,:,0]
    std = metric_stats_data[:,:,1]
    noise_values = noise_metrics_stats_data[:,:,0]
    noise_std = noise_metrics_stats_data[:,:,1]
    if metric=='NRMSE' or metric=='RMSE':
        values_BETTER = torch.where(values < noise_values, values, 0)
        std_BETTER = torch.where(values < noise_values, std, 0)
        
        values_WORSE = torch.where(values >= noise_values, values, 0)
        std_WORSE = torch.where(values >= noise_values, std, 0)
    elif metric=='PSNR' or metric=='MS_SSIM' or metric=='SSIM': 
        values_BETTER = torch.where(values > noise_values, values, 0)
        std_BETTER = torch.where(values > noise_values, std, 0)
        
        values_WORSE = torch.where(values <= noise_values, values, 0)
        std_WORSE = torch.where(values <= noise_values, std, 0)
    else: 
        raise Exception("Unsupported metric")
    return [values_BETTER, std_BETTER, values_WORSE, std_WORSE]

