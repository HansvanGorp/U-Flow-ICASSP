# %% imports
# standard libs
from glob import glob
from natsort import natsorted
from tqdm import tqdm
import numpy as np
import h5py
import json
import argparse
from pathlib import Path

# local libs
from preprocess import scale, resample


# %% test
if __name__ == "__main__":
    # user inputs
    parser = argparse.ArgumentParser(description='extract DOD-O data and preprocess')
    parser.add_argument('-data_loc'  , type=str)
    parser.add_argument('-target_loc', type=str)
    args = parser.parse_args()
    max_epochs = 1792    
    
    # %%
    # create train folder if necesarry
    Path(args.target_loc+"//train").mkdir(parents=True, exist_ok=True)
    
    file_names = natsorted(glob(f"{args.data_loc}//PSG//*.h5"))
    no_files = len(file_names)
    
    
    # exlcude one corrupted file
    for i,file_name in enumerate(file_names):
        if "5d8e93a5-d26a-436d-bc2c-e9527dc92cbf.h5" in file_name:
            file_names.pop(i)
            
    # loop
    for i,file_name in enumerate(tqdm(file_names)):        
        f = h5py.File(file_name, 'r')
        
        # scoring
        scoring = []
        for j in range(1,6):
            scoring_name = file_name.replace("PSG",f"scorer_{j}").replace("h5","json")
            scoring.append(np.array(json.load(open(scoring_name))))
            
        scorings = np.vstack(scoring)
        
        # scorings mask
        mask = ((scorings==-1).sum(axis=0)==0)
        scorings = scorings[:,mask]
        
        scorings2 = np.zeros((5,max_epochs))-1
        scorings2[:,0:scorings.shape[1]] = scorings
        
        
        # PSG siognals
        C3   = f['signals']['eeg']['C3_M2']
        C4   = f['signals']['eeg']['C4_M1']
        O1   = f['signals']['eeg']['O1_M2']
        O2   = f['signals']['eeg']['O2_M1']
        
        EOG1 = f['signals']['eog']['EOG1']
        EOG2 = f['signals']['eog']['EOG2']
        
        EMG  = f['signals']['emg']['EMG']
        ECG  = f['signals']['emg']['ECG']
        
        data = np.vstack((C3,C4,O1,O2,EOG1,EOG2,EMG,ECG))
        PSG = np.zeros((8,max_epochs*30*250))
        PSG[:,0:data.shape[1]] = data
        
        # preprocess
        # resample
        PSG = resample(PSG,250,128)
        
        # scale
        PSG = scale(PSG)
        
        # cast data to correct type
        PSG = np.float32(PSG)
        scoring = np.int64(scorings2)
        
        # save data
        save_name = f"{args.target_loc}//train//DOD-O_{i}.npz"
        np.savez(save_name, PSG = PSG, scoring = scoring)
        
    
    
