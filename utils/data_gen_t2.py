import torch

from denoising_assessment_project.global_vars import global_vars
from denoising_assessment_project.utils import data_gen

# 1. - EVAL type 2: Generating series (in the qunatity equal to 'series_quantity') of images, each of size max('my_range')+1, where each consecutive image in a series has got a single more randomly added blob relative to the previous image 
# output: [tensor]; shape = [series_quantity, blobs_population_sizes.shape[0], series_size.shape[0], image_dimensions[0], image_dimensions[1]]
def varying_blob_quantities_img_series_generate(series_quantity, my_range, image_dimensions): 
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
            single_coordinates = data_gen.rand_x_y(image_dimensions)
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

# 3. - EVAL type 2: Generating indices
def indices_generate(noised_images):
    blobs_indices = global_vars.blobs_population_sizes.unsqueeze(0).expand(noised_images[:,0,:,0,0].shape)
    peaks_indices = global_vars.peak_values.unsqueeze(1).expand(noised_images[:,0,:,0,0].shape)
    indices = torch.stack((blobs_indices, peaks_indices))
    return indices