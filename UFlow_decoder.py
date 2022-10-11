# %% imports
# standard libraries
import torch
import torch.nn as nn
import numpy as np

# local imports
from glow_1D.flow_1D import Flow_1D
from glow_1D.squeeze_1D import Squeeze_1D
from glow_1D.split_1D import Split_1D

# %% create likelihood glow module
class decoder(nn.Module):
    def __init__(self, args):
        super(decoder, self).__init__() 
                
        #sizes
        self.input_size = args.input_size        
        self.factor = args.factor
        
        # initialize all the layers
        self.K = args.K
        self.L = args.L+1
        
        self.squeeze =  Squeeze_1D(self.factor)
        self.split = Split_1D()
        self.flow_levels = nn.ModuleList()
        
        # inital shapes
        c = 6
        e = 1792
        
        # go through all the levels
        for level in range(self.L):            
            flow_layers = nn.ModuleList()
            for k in range(self.K):
                 flow = Flow_1D(args,c,e)
                 flow_layers.append(flow)
        
            self.flow_levels.append(flow_layers)
            c = (c*self.factor)//2
            e = e//self.factor
        
        # push to device
        self.device = args.device
        self.to(self.device)
        
        # shapes
        self.consistent_sample = torch.randn(6,1792*6).to(self.device)
    
    # %% forward
    def forward(self, x, C, logdet=None):
        #check if there already is a logdet
        if logdet is None:
            logdet = torch.tensor(0.0,requires_grad=False,device=self.device,dtype=torch.float)
    
        # go through each level
        Z = []
        for l in range(self.L):
            if l != 0 : # do not squeeze the first time
                x = self.squeeze(x)
            # flow
            for k in range(self.K):
                x,logdet = self.flow_levels[l][k](x,C[l],logdet)
                
            x,z = self.split(x)
            
            Z.append(z)
            
        Z.append(x)
        
        # concatenate and reshape
        z = torch.cat(Z,axis=-1)
        z = z.reshape(z.size(0),z.size(1)*z.size(2))
        
        return z, logdet
    
    # %% negative log-likelihood
    def nll(self, z, logdet):
        # p_z
        p_z = 0.5*torch.sum(z**2,dim=-1)
                            
        # nll
        nll = (p_z-logdet)/self.input_size
        
        return nll, (p_z/self.input_size, logdet/self.input_size)
    
    # %% reverse
    def reverse(self, z, C, logdet=None):
        #check if there already is a logdet
        if logdet is None:
            logdet = torch.tensor(0.0,requires_grad=False,device=self.device,dtype=torch.float)
            
        # first reshape
        c = 3
        z = z.reshape(z.size(0),c,z.size(1)//c)
        
        # put back in list
        Z = []
        start = 0
        e = z.size(2)
        for l in range(self.L):
            e = e//self.factor
            end = start+e
            Z.append(z[:,:,start:end])
            start = end
        
        end = start +e
        Z.append(z[:,:,start:end])
        
        # go through each level
        z = Z[-1]
        for l in reversed(range(self.L)):
            z = self.split.reverse(z, Z[l])
            # flow
            for k in reversed(range(self.K)):
                z,logdet = self.flow_levels[l][k].reverse(z,C[l],logdet)
            
            if l != 0 : # do not squeeze the first time
                z = self.squeeze.reverse(z)
        x = z
        return x, logdet
    
    # %% create conistent sample
    def sample_conistently(self,C):
        with torch.no_grad():
            z = self.consistent_sample
            labels, logdet = self.reverse(z, C)
            
            # calculate extra stuff
            nll,_ = self.nll(z, logdet)
            labels = torch.argmax(labels,dim=1)
        return labels, nll
    
    # %% dequantize
    def dequantize(self, scoring):
        # get rid of uncscored
        scoring[scoring==-1] = 0
        
        # make onehot
        scoring = torch.nn.functional.one_hot(scoring,num_classes = 6).float()
        
        # dequantize
        u1 = torch.rand_like(scoring)-0.5
        u2 = torch.rand_like(scoring)-0.5
        e = (u1+u2)/2
        scoring = 0.25 + 0.5*scoring + 0.5*e
        
        # tranpose
        scoring = scoring.transpose(2,1)
        
        return scoring