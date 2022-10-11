import torch
import torch.nn as nn
import numpy as np
    


class Split_1D(nn.Module):
    def __init__(self):
        super(Split_1D, self).__init__()
        
    def forward(self, x):
        c = x.size(1)
        x1 = x[:,:c//2,:]
        x2 = x[:,c//2:,:]
        return x1, x2
    
    def reverse(self,z1,z2):
        z = torch.cat((z1,z2),dim=1)
        return z
