import torch
import torch.nn as nn
import numpy as np
    


class Squeeze_1D_contiguous(nn.Module):
    def __init__(self, factor):
        super(Squeeze_1D_contiguous, self).__init__()
        self.factor = factor
        
    def forward(self, x):
        b,c,e  = x.size()
        x = x.reshape(b, c*self.factor, e//self.factor)
        return x
    
    def reverse(self,z):
        b,c,e  = z.size()
        z = z.reshape(b, c//2, e*self.factor)
        
        return z
    
class Squeeze_1D(nn.Module):
    def __init__(self, factor):
        super(Squeeze_1D, self).__init__()
        self.factor = factor
        
    def forward(self, x):
        b,c,e  = x.size()
        
        x = x.reshape(b, c, e//self.factor, self.factor)
        x = x.permute(0,1,3,2)
        x = x.reshape(b, c*self.factor, e//self.factor)
        
        return x
    
    def reverse(self,z):
        b,c,e  = z.size()
        
        z = z.reshape(b, c//self.factor, self.factor, e)
        z = z.permute(0,1,3,2)
        z = z.reshape(b, c//self.factor, e*self.factor)
        
        return z
    
