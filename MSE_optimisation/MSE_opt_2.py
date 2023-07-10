import torch
import functions

# optimisation per what?
# maybe separately for each image and iterate over all images - that would simplify it alot 
# alpha, beta - two scalars, but used as a plane of scalars of dimensions 256x256
# - input them as scalars and only inside the functions for whose minimum value we're optimising unsqueeze and expand 

# 2 constant tensors: GT_imgs normalised, prediction_imgs
# 2 changing tensors with a series of scalars - one scalar per image (20x20x50 scalars)

# it needs to be the same size as noised_images
alpha_0 = torch.tensor([0.], requires_grad = True)
alpha = alpha_0

beta_0 = torch.tensor([0.], requires_grad = True)
beta = beta_0

optimizer = torch.optim.SGD([alpha_0], lr=0.1)
# no idea how many steps there should be 
steps = 2000
for i in range(steps):
  optimizer.zero_grad()
  f = MSE(normalised_GT_image, noised_image, alpha, beta)
  f.backward()
  optimizer.step()
  if i%100 == 0:
    print(f'At step {i+1:4} x={str([round(x[i].item(), 2) for i in range(x.numel())]):18}'\
          f' and the function value is {f.item(): 0.4f}.')

# 1. where should the optimiser be called from? 
# 2. How to store the optimised alpha and beta values for each image? 
# -> sth like 20x20x50x2 structure?
# 3. Only past generating a complete set of alphas and betas should we actually transform the values? 
# and then run RMSE on those transformed values? 
# 