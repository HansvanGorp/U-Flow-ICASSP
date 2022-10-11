ALTERNATIVE_EOG = [11, 13, 14, 17, 19, 20, 21, 23, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 41, 42, 43,
                   44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
                   69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90 ,91, 92, 93,
                   94, 95, 96, 97, 98, 99, 100]

# %% imports
# standard libs
from tqdm import tqdm
import numpy as np
import mne
from pathlib import Path
import argparse

# local libs
from preprocess import scale, resample


# %% extract edf data
def extract_edf(file):
    # read edf
    edf_data = mne.io.read_raw_edf(file,verbose=False)
    
    # extract raw data 
    raw_data = edf_data.get_data()
    channels = edf_data.ch_names
    
    # extract all the channels of interest
    C3 = raw_data[[channels.index(i) for i in channels if 'C3' in i][0],:]
    C4 = raw_data[[channels.index(i) for i in channels if 'C4' in i][0],:]
    O1 = raw_data[[channels.index(i) for i in channels if 'O1' in i][0],:]
    O2 = raw_data[[channels.index(i) for i in channels if 'O2' in i][0],:]
    
    try:
        LOC = raw_data[[channels.index(i) for i in channels if 'LOC' in i][0],:]
        ROC = raw_data[[channels.index(i) for i in channels if 'ROC' in i][0],:]
    except:
        LOC = raw_data[[channels.index(i) for i in channels if 'E1' in i][0],:]
        ROC = raw_data[[channels.index(i) for i in channels if 'E2' in i][0],:]
    
    try:
        EMG = raw_data[[channels.index(i) for i in channels if 'X1' in i][0],:]
        ECG = raw_data[[channels.index(i) for i in channels if 'X2' in i][0],:]
    except:
        EMG = raw_data[[channels.index(i) for i in channels if '24' in i][0],:]
        ECG = raw_data[[channels.index(i) for i in channels if '25' in i][0],:]

    data = np.vstack((C3,C4,O1,O2,LOC,ROC,EMG,ECG))
    
    return data

# %% load scoring
def load_scoring(scoring_file):
    file = open(scoring_file,'r')
    lines = file.readlines()
    scoring = []
    for line in lines:
        if line[0] != '\n':
            scoring.append(int(line[0]))
            
    scoring = np.array(scoring)
    scoring[scoring==5] = 4
    return scoring

# %% test
if __name__ == "__main__":
    # user inputs
    parser = argparse.ArgumentParser(description='extract ISRUC data and preprocess')
    parser.add_argument('-data_loc'  , type=str)
    parser.add_argument('-target_loc', type=str)
    args = parser.parse_args()
    max_epochs = 1792
    
    # create train folder if necesarry
    Path(args.target_loc+"//train").mkdir(parents=True, exist_ok=True)
    
    # get all edf files
    i = 0
    for ID in tqdm(range(1,111)):
        if ID>100:
            ID2 = ID-100
        else:
            ID2 = ID
        
        # %% rename edf
        p = Path(f"{args.data_loc}//{ID}//{ID2}//{ID2}.rec")
        if p.is_file():
            p.rename(p.with_suffix('.edf'))
        
        # %% load patient
        scoring_1 = load_scoring(f"{args.data_loc}//{ID}//{ID2}//{ID2}_1.txt")
        scoring_2 = load_scoring(f"{args.data_loc}//{ID}//{ID2}//{ID2}_2.txt")
        
        scoring = np.zeros((2,max_epochs))-1
        scoring[0,0:len(scoring_1)] = scoring_1
        scoring[1,0:len(scoring_2)] = scoring_2
        
        # %% PSG
        data = extract_edf(f"{args.data_loc}//{ID}//{ID2}//{ID2}.edf")
 
        PSG = np.zeros((8,max_epochs*30*200))
        PSG[:,0:data.shape[1]] = data
        
        # %% preprocess
        # resample
        PSG = resample(PSG,200,128)
        
        # scale
        PSG = scale(PSG)
        
        # cast data to correct type
        PSG = np.float32(PSG)
        scoring = np.int64(scoring)
        
        # save scoring and PSG as npz file
        save_name = f"{args.target_loc}//train//ISRUC_{i}.npz"
        np.savez(save_name, PSG = PSG, scoring = scoring)
        i+=1
