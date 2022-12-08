# %% imports
# standard libs
import numpy as np
from scipy import signal

# %% resampling
def resample(PSG,Fs, target_Fs):
    # check if sampling frequency is already correct
    if Fs == target_Fs:
        return PSG
    
    # calculate up-down
    up = target_Fs
    down = Fs
    while down%2==0 and down >2:
        up = up//2
        down = down//2
    
    
    # resample the signal using polyphase filtering
    resampled_PSG = signal.resample_poly(PSG, up, down, axis=1)
    
    return resampled_PSG

# %% scaling
def scale(PSG):
    s = np.percentile(np.abs(PSG),95,axis=1)
    s = np.repeat(np.expand_dims(s,1),PSG.shape[1],axis=1)
    scaled_PSG = np.sign(PSG)*np.log(np.abs(PSG)/s+1)
    return scaled_PSG
       
       