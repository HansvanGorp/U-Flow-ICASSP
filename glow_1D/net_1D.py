import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .actnorm_1D import ActNorm_1D
    
# %%
class NN(nn.Module):
    def __init__(self, args,c):
        super(NN, self).__init__()
        
        channels_in = c//2 + args.conditional_channels
        channels = args.NN_channels
        channels_out = c
        
        self.conv1    = nn.Conv1d(channels_in,channels,kernel_size=7,stride=1,padding=3,bias=True)
        self.actnorm1 = ActNorm_1D(args.device)
                                
        self.conv2    = nn.Conv1d(channels,channels,kernel_size=7,stride=1,padding=3,bias=True)
        self.actnorm2 = ActNorm_1D(args.device)
        
        self.conv3    = nn.Conv1d(channels,channels_out,kernel_size=7,stride=1,padding=3,bias=True)        
        self.logs     = nn.Parameter(torch.zeros(1, channels_out, 1))
        
        # initializing
        with torch.no_grad():
            
            nn.init.normal_(self.conv1.weight, mean=0, std=0.01)
            nn.init.zeros_(self.conv1.bias)
            
            nn.init.normal_(self.conv2.weight, mean=0, std=0.01)
            nn.init.zeros_(self.conv2.bias) 
            
            nn.init.zeros_(self.conv3.weight) # last layer initialized with zeros
            nn.init.zeros_(self.conv3.bias)
            
        # to device
        self.to(args.device)
            
    def forward(self, x, C):
        b,c,e = x.size()
        
        x = torch.cat((x,C),dim=1)
        
        x = self.conv1(x)
        x, _ = self.actnorm1(x,0)
        x = F.relu(x)
        
        x = self.conv2(x)
        x, _ = self.actnorm2(x,0)
        x = F.relu(x)
        
        x = self.conv3(x)
        x = x * torch.exp(self.logs * 3).repeat(b,1,e)
        return x
    

