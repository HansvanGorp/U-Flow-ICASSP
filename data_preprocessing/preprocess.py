# %% imports
# standard libs
from glob import glob
from natsort import natsorted
from tqdm import tqdm
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

# %% majority vote
def get_majority_vote(scoring):
    no_scorers = scoring.shape[0]
    no_epochs  = scoring.shape[1]
    
    # if only one scorer than that is the majority vote
    if no_scorers == 1:
        return scoring[0,:]
    
    # else
    votes = np.zeros((6,no_epochs))
    
    #go over stages
    for i in range(6):
        votes[i,:] = (scoring == (i-1)).sum(axis=0)
    
    # majority vote
    majority_vote = votes.argmax(axis=0)-1
    
    return majority_vote


# %% if name main
if __name__ == "__main__":
    # %% parameters
   Root = "D://Datasets//U-Flow-Data"
   
   #Datasets = ["DOD-H", "DOD-O", "ISRUC", "IS-RC"]
   #Fs_sets  = [250    , 250    , 200    , 128] # sampling frequencies belonging to each dataset
   
   Datasets = ["SSC"]
   Fs_sets = [256]
   
   target_Fs = 128
   
   
   train_sets = ["DOD-H", "DOD-O", "ISRUC","SSC"] # which sets contain training data
   val_sets = ["SSC"] # which train sets also contain validation data
   val_percentage = 0.2 # percentage of validation set that is validation (vs training)
   
   # %% preprocessing
   for Dataset, Fs in zip(Datasets, Fs_sets):
       file_names = natsorted(glob(f"{Root}//intermediate//{Dataset}//*.npz"))
       for i,file_name in enumerate(tqdm(file_names)):
           
           # load file
           npz_file = np.load(file_name)
           PSG = npz_file['PSG']
           scoring = npz_file['scoring']
           
           # add dimension if SSC
           if Dataset == 'SSC':
               scoring = scoring.reshape((1,1792))
           
           # calculate majority vote
           majority_vote = get_majority_vote(scoring)
           
           # resample PSG to 128 Hz
           resampled_PSG = resample(PSG,Fs,target_Fs)
           
           # scale
           scaled_PSG = scale(resampled_PSG)
           
           # check under which type this data falls
           if Dataset in val_sets:
               if (i+1)<=val_percentage*len(file_names):
                   split = "val"
               else:
                   split = "train"
           elif Dataset in train_sets:
               split = "train"
           else:
               split = "test"
            
           # cast data to correct type
           scaled_PSG = np.float32(scaled_PSG)
           scoring = np.int64(scoring)
           majority_vote = np.int64(majority_vote)
            
           # save the data
           save_name = f"{Root}//preprocessed//{split}//{Dataset}_{i}.npz"
           np.savez(save_name, PSG = scaled_PSG, scoring = scoring, majority_vote = majority_vote)

       
       