import torch
import torch.nn as nn
import numpy as np

# device 
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    


class ActNorm_1D(nn.Module):
    def __init__(self, device):
        super(ActNorm_1D, self).__init__()
        self.initialized = False
        
    def initialize(self, x):
        if not self.training:
            self.initialized = True
            return
        with torch.no_grad():
            b,c,e = x.size()
            
            b_    = x.clone().mean(dim=(0,2))
            b_unsqueezed = b_.reshape(1,c,1).repeat(b,1,e)
            s_    = ((x.clone() - b_unsqueezed)**2).mean(dim=(0,2))
            
            b_    = -1 * b_
            logs_ = -1 * torch.log(torch.sqrt(s_)) 
            
            self.logs = torch.nn.Parameter(logs_)
            self.b = torch.nn.Parameter(b_)
            self.initialized = True
        
    def apply_bias(self, x, logdet):
            b,c,e = x.size()
            bias = self.b.reshape(1,c,1).repeat(b,1,e)
            x = x + bias
            return x, logdet
        
    def reverse_bias(self, x, logdet):
            b,c,e = x.size()
            bias = self.b.reshape(1,c,1).repeat(b,1,e)
            x =  x - bias
            return x, logdet
        
    def apply_scale(self,x, logdet):
            b,c,e = x.size()
            logs = self.logs.reshape(1,c,1).repeat(b,1,e)
            x = x * torch.exp(logs)
            logdet = logdet + e*torch.sum(self.logs)
            return x, logdet
        
    def reverse_scale(self,x, logdet):
            b,c,e = x.size()
            logs = self.logs.reshape(1,c,1).repeat(b,1,e)
            x = x * torch.exp(-logs)
            logdet = logdet + e*torch.sum(self.logs)
            return x, logdet
        
    def forward(self, x, logdet):
        if not self.initialized:
            self.initialize(x)
            
        x, logdet   = self.apply_bias(x,  logdet)
        x, logdet   = self.apply_scale(x, logdet)
            
        return x, logdet
        
    def reverse(self,x, logdet):
        x, logdet = self.reverse_scale(x, logdet)
        x, logdet = self.reverse_bias(x, logdet)
        
        return x, logdet
    