import torch
import torch.nn as nn
import numpy as np
from .net_1D import NN


class CouplingLayer_1D(nn.Module):
    def __init__(self,args, c):
        super(CouplingLayer_1D, self).__init__()

        self.coupling_bias = args.coupling_bias
        
        self.net = NN(args,c)

        
        self.to(args.device)


    def forward(self, x, C, logdet):
            xa, xb = self.split(x)
        
            s_and_t = self.net(xb, C)
            
            s, t = self.split(s_and_t)
            
            s = torch.sigmoid(s + 2.) + self.coupling_bias
            
            ya = s * xa + t
            
            y = torch.cat([ya, xb], dim=1)
            
            logdet = logdet + torch.log(s).sum((1,2))
                    
            return y, logdet
            
    def reverse(self, z, C, logdet):
            za, zb = self.split(z)
            
            s_and_t = self.net(zb, C)
            
            s, t = self.split(s_and_t)
            
            s = torch.sigmoid(s + 2.) + self.coupling_bias
            
            xa = (za - t) / s
            
            x = torch.cat([xa, zb], dim=1)
            
            logdet = logdet + torch.log(s).sum((1,2))
            
            return x, logdet
            
    def split(self, x):
        xa = x[:,:x.size(1)//2,:]
        xb = x[:,x.size(1)//2:,:]
        return xa, xb

