
# source: https://www.cl.cam.ac.uk/teaching/2021/LE49/probnn/3-3.pdf

import torch
import torch.nn as nn 
import torch.optim as optim

class StraightLine(nn.Module): 
    def __init__(self):
        super().__init__()
        self.α = nn.Parameter(torch.tensor(1.0)) 
        self.β = nn.Parameter(torch.tensor(1.0))
    def forward(self, x):
        return self.α+ self.β ∗ x

class Y(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = StraightLine()
        self.σ = nn.Parameter(torch.tensor(1.0))
    def forward(self, x, y):
        pred = self.f(x)
        return −0.5∗torch.log(2∗np.pi∗self.σ∗∗2) − (y−pred)∗∗2/2/self.σ∗∗2
                                            
model = Y()
optimizer = optim.Adam(model.parameters()) 
epoch = 0

with Interruptable() as check_interrupted: 
    check_interrupted() 
    optimizer.zero_grad()
    loglik = model(x, y)
    e =− torch.mean(loglik) 
    e.backward() 
    optimizer.step()
    IPython.display.clear_output(wait=True)
    print(f ’epoch={epoch} loglik={−e.item():.3} ’)
    epoch += 1