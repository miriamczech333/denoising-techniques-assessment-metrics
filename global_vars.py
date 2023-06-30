import torch

def init():
    # -PARAMETERS FOR GENERATION-

    # image dimensions
    global image_dimensions
    #image_dimensions = torch.Size([256,256])

    # reps in a series 
    global series_size 
    #series_size = 50

    global blobs_population_sizes
    #blobs_population_sizes = torch.arange(3,300,15)
    global peak_values
    #peak_values = torch.arange(10,1000,50)

    # necessary setup: 
    global device
    torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
