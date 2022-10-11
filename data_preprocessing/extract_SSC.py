EXCLUDED = ['ssc_4700_1-nsrr.edf', 'ssc_5589_1-nsrr.edf', 'ssc_6408_1-nsrr.edf', 'ssc_6865_1-nsrr.edf',
            'ssc_4682_1-nsrr.edf', 'ssc_4686_1-nsrr.edf', 'ssc_4691_1-nsrr.edf', 'ssc_4717_1-nsrr.edf',
            'ssc_4724_1-nsrr.edf', 'ssc_4745_1-nsrr.edf', 'ssc_4746_1-nsrr.edf', 'ssc_4799_1-nsrr.edf',
            'ssc_4802_1-nsrr.edf', 'ssc_4838_1-nsrr.edf', 'ssc_4847_1-nsrr.edf', 'ssc_4849_1-nsrr.edf',
            'ssc_4853_1-nsrr.edf', 'ssc_4888_1-nsrr.edf', 'ssc_4889_1-nsrr.edf', 'ssc_4895_1-nsrr.edf',
            'ssc_4915_1-nsrr.edf', 'ssc_5018_1-nsrr.edf', 'ssc_5050_1-nsrr.edf', 'ssc_5056_1-nsrr.edf',
            'ssc_5057_1-nsrr.edf', 'ssc_5064_1-nsrr.edf', 'ssc_5066_1-nsrr.edf', 'ssc_5069_1-nsrr.edf',
            'ssc_5098_1-nsrr.edf', 'ssc_5100_1-nsrr.edf', 'ssc_5113_1-nsrr.edf', 'ssc_5116_1-nsrr.edf',
            'ssc_5117_1-nsrr.edf', 'ssc_5138_1-nsrr.edf', 'ssc_5155_1-nsrr.edf', 'ssc_5183_1-nsrr.edf',
            'ssc_5187_1-nsrr.edf', 'ssc_5190_1-nsrr.edf', 'ssc_5202_1-nsrr.edf', 'ssc_5204_1-nsrr.edf',
            'ssc_5234_1-nsrr.edf', 'ssc_5248_1-nsrr.edf', 'ssc_5259_1-nsrr.edf', 'ssc_5270_1-nsrr.edf',
            'ssc_5318_1-nsrr.edf', 'ssc_5333_1-nsrr.edf', 'ssc_5419_1-nsrr.edf', 'ssc_5433_1-nsrr.edf',
            'ssc_5439_1-nsrr.edf', 'ssc_5441_1-nsrr.edf', 'ssc_5495_1-nsrr.edf', 'ssc_5525_1-nsrr.edf',
            'ssc_5526_1-nsrr.edf', 'ssc_5535_1-nsrr.edf', 'ssc_5536_1-nsrr.edf', 'ssc_5562_1-nsrr.edf',
            'ssc_5564_1-nsrr.edf', 'ssc_5573_1-nsrr.edf', 'ssc_5576_1-nsrr.edf', 'ssc_5616_1-nsrr.edf',
            'ssc_5711_1-nsrr.edf', 'ssc_5841_1-nsrr.edf', 'ssc_5851_1-nsrr.edf', 'ssc_5857_1-nsrr.edf',
            'ssc_5877_1-nsrr.edf', 'ssc_5879_1-nsrr.edf', 'ssc_5979_1-nsrr.edf', 'ssc_6035_1-nsrr.edf',
            'ssc_6036_1-nsrr.edf', 'ssc_6038_1-nsrr.edf', 'ssc_6042_1-nsrr.edf', 'ssc_6066_1-nsrr.edf',
            'ssc_6067_1-nsrr.edf', 'ssc_6070_1-nsrr.edf', 'ssc_6082_1-nsrr.edf', 'ssc_6087_1-nsrr.edf',
            'ssc_6095_1-nsrr.edf', 'ssc_6106_1-nsrr.edf', 'ssc_6112_1-nsrr.edf', 'ssc_6117_1-nsrr.edf',
            'ssc_6125_1-nsrr.edf', 'ssc_6135_1-nsrr.edf', 'ssc_6136_1-nsrr.edf', 'ssc_6143_1-nsrr.edf',
            'ssc_6199_1-nsrr.edf', 'ssc_6206_1-nsrr.edf', 'ssc_6241_1-nsrr.edf', 'ssc_6252_1-nsrr.edf',
            'ssc_6259_1-nsrr.edf', 'ssc_6260_1-nsrr.edf', 'ssc_6280_1-nsrr.edf', 'ssc_6281_1-nsrr.edf',
            'ssc_6284_1-nsrr.edf', 'ssc_6392_1-nsrr.edf', 'ssc_6840_1-nsrr.edf', 'ssc_6851_1-nsrr.edf',
            'ssc_6853_1-nsrr.edf', 'ssc_6870_1-nsrr.edf', 'ssc_6933_1-nsrr.edf', 'ssc_6938_1-nsrr.edf',
            'ssc_6939_1-nsrr.edf', 'ssc_7044_1-nsrr.edf', 'ssc_7079_1-nsrr.edf', 'ssc_7191_1-nsrr.edf',
            'ssc_7228_1-nsrr.edf', 'ssc_7235_1-nsrr.edf', 'ssc_7389_1-nsrr.edf', 'ssc_7450_1-nsrr.edf',
            'ssc_7463_1-nsrr.edf', 'ssc_7488_1-nsrr.edf', 'ssc_7636_1-nsrr.edf']
            
            

# %% imports
# standard libs
from glob import glob
from natsort import natsorted
from tqdm import tqdm
import numpy as np
import mne
from pathlib import Path
import matplotlib.pyplot as plt
import argparse

# local libs
from preprocess import scale, resample

# %% extract edf data
def extract_edf(file, max_epochs):
    # read edf
    edf_data = mne.io.read_raw_edf(file,verbose=False)
    
    # extract raw data 
    raw_data = edf_data.get_data()
    channels = edf_data.ch_names
    
    # extract all the channels of interest
    C3 = raw_data[channels.index('C3'),:]
    C4 = raw_data[channels.index('C4'),:]
    O1 = raw_data[channels.index('O1'),:]
    O2 = raw_data[channels.index('O2'),:]
    
    LOC = raw_data[channels.index('E1'),:]
    ROC = raw_data[channels.index('E2'),:]
    
    EMG = raw_data[channels.index('cchin'),:]
    
    ECG = raw_data[channels.index('ECG'),:]

    data = np.vstack((C3,C4,O1,O2,LOC,ROC,EMG,ECG))
    
    return data

# %% load scoring
def load_scoring(scoring_file):
    file = open(scoring_file,'r')
    lines = file.readlines()
    
    scoring = np.zeros((len(lines)))-1
    
    for i,line in enumerate(lines):
        if 'wake' in line:
            scoring[i] = 0
        elif 'N1' in line:
            scoring[i] = 1
        elif 'N2' in line:
            scoring[i] = 2
        elif 'N3' in line:
            scoring[i] = 3
        elif 'REM' in line:
            scoring[i] = 4
    return scoring


def find_start(scoring):
    start = 0
    while(scoring[start] == -1):
        start+=1
    return start


# %% fft
def plt_fft_segment(signal,Fs,name, min_t, max_t):
    min_sample = min_t*Fs
    max_sample = max_t*Fs
    
    segment = signal[min_sample:max_sample]
    
    F = np.abs(np.fft.fft(segment))
    PSD = 20*np.log(F)/np.log(10)
    
    Freq = (np.arange(len(F))/len(F))*Fs
    
    
    
    plt.figure()
    plt.plot(Freq,PSD)
    plt.grid()
    plt.title(name)
    plt.xlabel('freq [Hz]')
    plt.ylabel('PSD [-]')
    plt.xlim(0,Fs/2)
    
# %% test
if __name__ == "__main__":
    # user inputs
    parser = argparse.ArgumentParser(description='extract SSC data and preprocess')
    parser.add_argument('-data_loc'  , type=str)
    parser.add_argument('-target_loc', type=str)
    args = parser.parse_args()
    
    max_epochs = 1792
    
    # create train and val folder if necesarry
    Path(args.target_loc+"//train").mkdir(parents=True, exist_ok=True)
    Path(args.target_loc+"//val").mkdir(parents=True, exist_ok=True)
    
    file_names = natsorted(glob(f"{args.data_loc}//*.edf"))
    no_files = len(file_names)
    
    max_len = 0

    
    # loop over all files
    i = 0
    for file in tqdm(file_names):
        if file[41:] in EXCLUDED:
            continue
        
        scoring_file = file.replace('edf','eannot')
        scoring = load_scoring(scoring_file)
        
        start_epoch = find_start(scoring)
        
        end_epoch = len(scoring)-start_epoch
        len_here = end_epoch-start_epoch
        if len_here> max_len:
            max_len = len_here
        
        PSG = extract_edf(file, max_epochs)
        
        # adjust
        scoring_adjusted = scoring[start_epoch:]
        PSG_adjusted = PSG[:,start_epoch*30*256:]
        
        # zero-pad
        scoring_padded = np.zeros((max_epochs))-1
        PSG_padded = np.zeros((8,max_epochs*30*256))
        
        no_epochs_here = len(scoring_adjusted)
        
        scoring_padded[0:no_epochs_here] = scoring_adjusted
        PSG_padded[:,0:no_epochs_here*30*256] = PSG_adjusted[:,0:no_epochs_here*30*256]
        
        
        # %% preprocess
        # resample
        PSG = resample(PSG_padded,256,128)
        
        # scale
        PSG = scale(PSG)
        
        # cast data to correct type
        scoring = scoring_padded.reshape((1,1792))
        PSG = np.float32(PSG)
        scoring = np.int64(scoring)
        
        # save scoring and PSG as npz file
        if (i+1)<=0.2*len(file_names):
            split = "val"
        else:
            split = "train"
        
        save_name = f"{args.target_loc}//{split}//SSC_{i}.npz"
        np.savez(save_name, PSG = PSG, scoring = scoring)
        i+=1
    