# %% imports
# add parent directory to path
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

# standard libraries
from pathlib import Path
import torch
from tqdm import tqdm
import argparse

# local import
import dataloading
import Summerizing_Statistics_Functions as SSF

# %% seed
torch.random.manual_seed(0)

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test dataloaders')
    parser.add_argument('-batch_size',type=float,help='batch size',default=1)
    parser.add_argument('-no_samples_per_epoch',type=int,help='how much samples to show every epoch',default=4)
    parser.add_argument('-save_name',type=str,help='name to save results under',default="ground_truth")
    parser.add_argument('-data_loc',type=str,help='location of data')
    args = parser.parse_args()
    
    # %% create dataloader
    dataloader_test, args = dataloading.getDataLoaderTest(args)
    print('Loaded data')
    
    no_patients = dataloader_test.dataset.__len__()
    no_samples = 6
    
    # %% init stats
    stats = SSF.stats_object(no_samples)
    predictions = torch.zeros(no_patients,no_samples,1792)
    masks = torch.zeros(no_patients,1792).type(torch.bool)
    
    # %% go over the test set
    for i,(PSG, scoring, mask) in enumerate(tqdm(dataloader_test)):
        # remove batch dimension
        scoring = scoring[0,:,:]
        
        # add on the samled y to the predictions
        predictions[i,:,:] = scoring
        
        #if patient_id is 35 then one scoring is missing...
        if i == 35:
            # remove scoring 2 as it is missing
            scoring = torch.cat((scoring[:2],scoring[2+1:]))
        
        # get the stats for this index
        stats = SSF.create_stats_from_scoring(stats,scoring,i)
        
        # add on the mask
        masks[i,:] = mask[0,:]
        
    # %% save
    Path("results").mkdir(parents=True, exist_ok=True)
    
    torch.save(stats,f"results/{args.save_name}_statistics.pt")
    torch.save(predictions,f"results/{args.save_name}_predictions.tar")
    torch.save(masks,f"results/{args.save_name}_masks.tar")  