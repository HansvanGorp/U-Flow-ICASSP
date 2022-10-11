# %% imports
# standard libs
from pathlib import Path
from glob import glob
from natsort import natsorted
from tqdm import tqdm
import numpy as np
import mne
import argparse

# local libs
from preprocess import scale

# %% extract label data            
def extract_labels(file,max_epochs,patient_id):
    # get the file as collumns
    with open(file) as f:
        collumns = zip(*[line for line in f])
    
    # extract only the usefull collumns
    scoring_here = []
    for i,collumn in enumerate(collumns):
        if i % 3 == 0:
            scoring = collumn[1:]
            scoring_as_array = np.asarray(scoring,dtype=int)
                         
            # fit all scorings into the same amount
            scoring_as_array_full_length = np.zeros(max_epochs,dtype=int)-1
            scoring_as_array_full_length[0:len(scoring_as_array)] = scoring_as_array
            scoring_as_array_full_length[scoring_as_array_full_length==5] = 4
            scoring_as_array_full_length[scoring_as_array_full_length==7] = -1
            scoring_here.append(scoring_as_array_full_length)
            
    # stack the scorings of the 6 different scorers
    scoring_here = np.vstack(scoring_here)

    # change scorer 2-6 based on cross correlation with scorer 1
    scoring_here_2 = np.zeros_like(scoring_here)-1
    
    # offset of 100
    scoring_here_2[0,100:] = scoring_here[0,:1792-100]
    
    for i in range(1,6):
        lag = np.argmax(np.correlate(scoring_here_2[0,:],scoring_here[i,:],'same')) - 1792//2
        scoring_here_2[i,lag:] = scoring_here[i,:1792-lag]
    
    # remove offset of 100
    scoring_here_3 = np.zeros_like(scoring_here)-1
    scoring_here_3[:,:1792-100] = scoring_here_2[:,100:]
    
    # remove any scoring where one of the scorers did not score
    if patient_id != 35:
        mask = np.any(scoring_here_3==-1,axis=0)
        scoring_here_3[:,mask] = -1
        
    else: #except when using patient 35
        scoring_here_test = np.vstack((scoring_here_3[:2],scoring_here_3[2+1:]))
        mask = np.any(scoring_here_test==-1,axis=0)
        scoring_here_3[:,mask] = -1
        
    # remove leading unscored
    lag = mask.argmin()
        
    scoring_here_4 = np.zeros_like(scoring_here)-1
    scoring_here_4[:,:1792-lag] = scoring_here_3[:,lag:]
        
    return scoring_here_4, lag

# %% extract edf data
def extract_edf(file,max_epochs, lag):
    # read edf
    edf_data = mne.io.read_raw_edf(file,verbose=False)
    
    # extract raw data 
    raw_data = edf_data.get_data()
    channels = edf_data.ch_names
    no_samples = raw_data.shape[1]
    
    # extract all the channels of interest
    C3 = raw_data[[channels.index(i) for i in channels if 'C3' in i][0],:]
    C4 = raw_data[[channels.index(i) for i in channels if 'C4' in i][0],:]
    O1 = raw_data[[channels.index(i) for i in channels if 'O1' in i][0],:]
    O2 = raw_data[[channels.index(i) for i in channels if 'O2' in i][0],:]
    
    LOC = raw_data[[channels.index(i) for i in channels if 'LOC' in i][0],:]
    ROC = raw_data[[channels.index(i) for i in channels if 'ROC' in i][0],:]
    
    EMG = raw_data[[channels.index(i) for i in channels if 'EMG' in i][0],:]
    
    ECG = raw_data[[channels.index(i) for i in channels if 'ECG' in i][0],:]

    data = np.vstack((C3,C4,O1,O2,LOC,ROC,EMG,ECG))
    
    # remove the epochs before start
    start = lag*30*128
    data_windowed = np.zeros((8,max_epochs*30*128))
    data_windowed[:,:(no_samples-start)] = data[:,start:]
    
    return data_windowed


# %% main
if __name__ == "__main__":
    # user inputs
    parser = argparse.ArgumentParser(description='extract IS-RC data and preprocess')
    parser.add_argument('-data_loc'  , type=str)
    parser.add_argument('-target_loc', type=str)
    args = parser.parse_args()
    
    max_epochs = 1792
    
    # create test folder if necesarry
    Path(args.target_loc+"//test").mkdir(parents=True, exist_ok=True)
    
    # get all edf files
    file_names = natsorted(glob(f"{args.data_loc}//*.edf"))
    no_files = len(file_names)
    
    # loop over all files
    for i,file in enumerate(tqdm(file_names)):  
        # get the label files
        STA_file = file.replace('edf','STA')
        scoring, lag = extract_labels(STA_file,max_epochs,i)
        
        # extract edf data
        PSG = extract_edf(file,max_epochs, lag)
        
        # scale
        PSG = scale(PSG)
        
        # cast data to correct type
        PSG = np.float32(PSG)
        scoring = np.int64(scoring)
        
        # save the data
        save_name = f"{args.target_loc}//test//IS-RC_{i}.npz"
        np.savez(save_name, PSG = PSG, scoring = scoring)

        
    
    
    