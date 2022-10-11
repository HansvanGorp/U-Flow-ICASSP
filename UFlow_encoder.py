# %% imports
# standard libraries
import torch
import torch.nn as nn
import numpy as np

# local imports

# %% create prenet
def create_preNet(channels_in, channels_out):
    layers = nn.Sequential(
        # conv 1
        nn.Conv1d(channels_in, channels_out, kernel_size = 7, padding = 3),
        nn.ReLU(),
        nn.MaxPool1d(4),
        
        # conv 2
        nn.Conv1d(channels_out, channels_out, kernel_size = 7, padding = 3),
        nn.ReLU(),
        nn.MaxPool1d(4),
        
        # conv 3
        nn.Conv1d(channels_out, channels_out, kernel_size = 7, padding = 3),
        nn.ReLU(),
        nn.MaxPool1d(4),
        
        # conv 4
        nn.Conv1d(channels_out, channels_out, kernel_size = 7, padding = 3),
        nn.ReLU(),
        nn.MaxPool1d(4),
        
        # conv 5
        nn.Conv1d(channels_out, channels_out, kernel_size = 15, padding = 0, stride = 15),
        )   
    
    return layers

# %% create squeezenet
def create_squeezeNet(channels_in, channels_out):
    layers = nn.Sequential(
        # conv 1
        nn.Conv1d(channels_in,  channels_out, kernel_size = 7, padding = 3),
        nn.ReLU(),
        nn.Dropout(),
        
        # conv 2
        nn.Conv1d(channels_out, channels_out, kernel_size = 7, padding = 3),
        nn.ReLU(),
        nn.Dropout(),
        
        # maxpool
        nn.MaxPool1d(2),
        )
    
    return layers

# %% create likelihood glow module
class encoder(nn.Module):
    def __init__(self, args):
        super(encoder, self).__init__()   
      
        # setup pre network
        channels_in  = 8
        channels_out = args.conditional_channels
        
        self.preNet = create_preNet(channels_in, channels_out)
        
        # set up squeezenet
        channels_in  = channels_out
        channels_out = channels_out
        self.squeezeNets = nn.ModuleList()
        for l in range(args.L):
           self.squeezeNets.append(create_squeezeNet(channels_in,channels_out))
        
        # push to device
        self.device = args.device
        self.to(self.device)
    
    # %% forward
    def forward(self, x):        
        # preNet
        x = self.preNet(x)
        
        # squeezeNets
        C = [x]
        for squeezeNet in self.squeezeNets:
            C.append(squeezeNet(C[-1]))
    
        return C