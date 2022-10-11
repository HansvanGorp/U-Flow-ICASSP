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
def create_encoders(channels_in, channels_out):
    layers = nn.Sequential(
        # conv 1
        nn.Conv1d(channels_in,  channels_out, kernel_size = 7, padding = 3),
        nn.ReLU(),
        nn.Dropout(),
        
        # conv 2
        nn.Conv1d(channels_out, channels_out, kernel_size = 7, padding = 3),
        nn.ReLU(),
        nn.Dropout(),
        )
    
    return layers

# %% create decoders
class decoder(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(decoder, self).__init__() 
        self.up_conv = nn.ConvTranspose1d(channels_in, channels_out, 2,stride=2) 
        
        self.layers = nn.Sequential(
            # conv 1
            nn.Conv1d(channels_in*2,  channels_out, kernel_size = 7, padding = 3),
            nn.ReLU(),
            nn.Dropout(),
            
            # conv 2
            nn.Conv1d(channels_out, channels_out, kernel_size = 7, padding = 3),
            nn.ReLU(),
            nn.Dropout(),
            )
        
    def forward(self, x_small, x_big):
        # upscale
        x_small_up = self.up_conv(x_small)
        
        #combined
        x = torch.cat((x_small_up,x_big),dim=1)
        
        # convs
        x = self.layers(x)
        
        return x

# %% create likelihood glow module
class UNet(nn.Module):
    def __init__(self, args):
        super(UNet, self).__init__()     
        self.L = args.L+1
        
        # setup pre network
        channels_in  = 8
        channels_out = args.epoch_encoder_channels
        
        self.preNet = create_preNet(channels_in, channels_out)
        
        # set up encoders
        channels_in  = channels_out
        channels_out = args.u_channels
        self.encoders = nn.ModuleList()
        for l in range(self.L):
             self.encoders.append(create_encoders(channels_in, channels_out))
             channels_in = channels_out
             
        # set up decoders
        self.decoders = nn.ModuleList()
        for l in range(self.L-1):
            self.decoders.append(decoder(channels_in, channels_out))
            
        # final predictor
        self.final_predictor = nn.Conv1d(channels_out,  5, kernel_size = 1, padding = 0)
        
        # push to device
        self.device = args.device
        self.to(self.device)
    
    # %% forward
    def forward(self, x):
        # prenet
        x = self.preNet(x)
        
        # encoders
        C = []
        for l in range(self.L):
            if l != 0 :
                x = nn.functional.max_pool1d(x,2)
            x = self.encoders[l](x)
            C.append(x)
            
        #decoders
        x_small = C.pop()
        for l in range(self.L-1):
            x_big = C.pop()
            x_small = self.decoders[l](x_small,x_big)
            
        # final
        logits = self.final_predictor(x_small)
        
        #reshape
        logits = logits.transpose(2,1)
    
        return logits