import torch
import torch.nn as nn
import numpy as np
from .actnorm_1D import ActNorm_1D
from .invconv_1D import Invconv_1D
from .coupling_1D import CouplingLayer_1D





class Flow_1D(nn.Module):
    def __init__(self, args,c,e):
        super(Flow_1D, self).__init__()
        self.actnorm = ActNorm_1D(args.device)
        self.invconv = Invconv_1D(args,c)
        self.coupling = CouplingLayer_1D(args,c)
        self.to(args.device)

    def forward(self, x, C, logdet):        
            x, logdet = self.actnorm(x, logdet)
            
            x, logdet = self.invconv(x, logdet)

            x, logdet = self.coupling(x, C, logdet)

            return x, logdet

    def reverse(self, z, C, logdet):
            z, logdet = self.coupling.reverse(z, C, logdet)
            
            z, logdet = self.invconv.reverse(z, logdet)

            z, logdet = self.actnorm.reverse(z, logdet)
            return z, logdet
