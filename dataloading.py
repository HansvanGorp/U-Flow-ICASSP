"""
This file contains functions specifying the sleep datasets and the accompanying dataloaders

This file can be run in main to check whether files are present or not.

"""

# %% imports
from glob import glob
from natsort import natsorted
from tqdm import tqdm
import numpy as np
import argparse
import torch
from torch.utils.data import Dataset,DataLoader

# %% specifying the datasets
class sleep_set(Dataset):
    def __init__(self, loc, split):
        # saving local variables
        self.loc_split = loc 
        self.split = split
    
        # get all the filenames
        self.file_names = natsorted(glob(f"{loc}{split}//*.npz"))
        
        # calculate length
        self.length = len(self.file_names)
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # return a recording with a randomly selected scorer
        
        file_name = self.file_names[idx]
        
        npz_file = np.load(file_name)
        
        PSG           = torch.tensor(npz_file['PSG'])
        scoring       = torch.tensor(npz_file['scoring'])
        
        no_scorers = scoring.shape[0]
        
        selected_scorer = np.random.randint(0,no_scorers)
        
        scoring_selected = scoring[selected_scorer,:]
        
        mask = (scoring_selected!=-1)
        
        return PSG, scoring_selected, mask
    
    
class sleep_set_all(Dataset):
    def __init__(self, loc, split):
        # saving local variables
        self.loc_split = loc 
        self.split = split
        
        # get all the filenames
        self.file_names = natsorted(glob(f"{loc}{split}//*.npz"))
        
        # calculate length
        self.length = len(self.file_names)
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # return a recording with all scorers
        
        file_name = self.file_names[idx]
        
        npz_file = np.load(file_name)
        
        PSG           = torch.tensor(npz_file['PSG'])
        scoring       = torch.tensor(npz_file['scoring'])
        
        mask =(scoring[0,:]!=-1)
        
        return PSG, scoring, mask
    
# %% create dataloaders
def getDataLoader(args):
    # get the datasets
    train_set = sleep_set(args.data_loc, "train")
    val_set   = sleep_set(args.data_loc, "val")
    
    # check if set is non-empty
    if train_set.__len__()==0 or val_set.__len__()==0:
        return 0,0,args
        
    # create dataloaders
    dataloader_train = DataLoader(train_set, batch_size=args.batch_size,
                                  shuffle=True,drop_last=False,
                                  pin_memory=True)
    
    dataloader_val   = DataLoader(val_set,   batch_size=args.batch_size,
                                  shuffle=False,drop_last=False,
                                  pin_memory=True)
    
    
    # calculate some additional stuff from the data
    args.no_classes = 5
    
    args.no_examples = train_set.__len__()     
    args.batches_per_epoch = int(np.ceil(args.no_examples/args.batch_size))
    args.batch_modules = int(args.batches_per_epoch/args.no_samples_per_epoch)
    
    
    return dataloader_train, dataloader_val, args

def getDataLoaderTest(args):
    # get the datasets
    test_set  = sleep_set_all(args.data_loc, "test")
        
    # check if set is non-empty
    if test_set.__len__()==0:
        return 0,args
    
    # create dataloader
    dataloader_test  = DataLoader(test_set,  batch_size=args.batch_size,
                                  shuffle=False,drop_last=False,
                                  pin_memory=True)   
    
    return dataloader_test, args

# %% if name main
if __name__ == "__main__":   
    parser = argparse.ArgumentParser(description='test dataloaders')
    parser.add_argument('-batch_size',type=float,help='batch size',default=2)
    parser.add_argument('-no_samples_per_epoch',type=int,help='how much samples to show every epoch',default=4)
    parser.add_argument('-data_loc',type=str,help='location of data')
    args = parser.parse_args()
    
    
    dataloader_train, dataloader_val, args = getDataLoader(args)
    dataloader_test, args = getDataLoaderTest(args)
    
    # %% test loading all the data
    if dataloader_train!=0:
        print(f"train      loader found {dataloader_train.dataset.__len__()} recordings")
    else:
        print("train      loader found   0 recordings")
    
    if dataloader_val!=0:
        print(f"validition loader found {dataloader_val.dataset.__len__()} recordings")
    else:
        print("validition loader found   0 recordings")
    
    if dataloader_test!=0:
        print(f"test       loader found  {dataloader_test.dataset.__len__()} recordings")
    else:
        print("test       loader found   0 recordings")