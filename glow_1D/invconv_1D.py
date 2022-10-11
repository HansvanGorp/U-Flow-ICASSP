import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

# device 
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    


class Invconv_1D(nn.Module):
    def __init__(self, args,c):
        super(Invconv_1D, self).__init__()
        self.W_size = [c, c, 1]
        W_init = np.random.normal(0, 1, self.W_size[:-1])
        W_init = np.linalg.qr(W_init)[0].astype(np.float32)
        W_init = W_init.reshape(self.W_size)
        self.W = nn.Parameter(torch.tensor(W_init))
        self.inv_w = None
        self.use_cached_inv_w = False

        self.to(args.device)
        
    def forward(self, x, logdet):
        b, c, e = x.size()
        x = F.conv1d(x, self.W)
        detW = torch.slogdet(self.W.squeeze())[1]
        logdet = logdet + e * detW
        return x, logdet
    
    def reverse(self,z, logdet):
        b, c, e = z.size()
        
        if (not self.training and self.use_cached_inv_w):
            if (self.inv_w is None):
                self.inv_w = torch.inverse(self.W.squeeze().double()).float().view(self.W_size)
            inv_w = self.inv_w
        else:
            inv_w = torch.inverse(self.W.squeeze().double()).float().view(self.W_size)
        z = F.conv1d(z, inv_w)
        
        detW = torch.slogdet(self.W.squeeze())[1]
        logdet = logdet + e* detW
        return z,logdet
