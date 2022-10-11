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
import UNet
import Summerizing_Statistics_Functions as SSF

# %% seed
torch.random.manual_seed(0)

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pretest U-Net factorized')
    parser.add_argument('-batch_size',type=float,help='batch size',default=1)
    parser.add_argument('-no_samples_per_epoch',type=int,help='how much samples to show every epoch',default=4)
    parser.add_argument('-save_name',type=str,help='name to save results under',default="U-Net")
    parser.add_argument('-data_loc',type=str,help='location of data')
    parser.add_argument('-device',type=str,help='device to use',default="cuda:0")
    args = parser.parse_args()
    
    # %% create dataloader 
    dataloader_test, args = dataloading.getDataLoaderTest(args)
    print('Loaded data')
    
    no_patients = dataloader_test.dataset.__len__()
    no_samples = 100
    
    # %% load Network
    args2        = torch.load(f"..//checkpoints//{args.save_name}//args.tar")
    state_dict  = torch.load(f"..//checkpoints//{args.save_name}//model.tar")
        
    Network_UNet = UNet.UNet(args2)    
    Network_UNet = Network_UNet.to(args.device)
    Network_UNet.load_state_dict(state_dict)
    print('Loaded Network')
    
    
    # %% init stats
    stats = SSF.stats_object(no_samples)
    predictions = torch.zeros(no_patients,no_samples,1792)
    masks = torch.zeros(no_patients,1792).type(torch.bool)
    
    # %% go over the test set
    Network_UNet.train()
    
    for i,(PSG, _, _) in enumerate(tqdm(dataloader_test)):
        # push to device
        PSG = PSG.to(args.device)
        
        # predict many times
        with torch.no_grad():
            logits = Network_UNet(PSG)
            for n in range(no_samples):
                eps = -torch.log(-torch.log(torch.rand_like(logits)))
                predictions[i,n,:] = torch.argmax(logits+eps,dim=-1)
                
        
        stats = SSF.create_stats_from_scoring(stats,predictions[i,:,:],i)
        
        
    # %% save
    Path("results").mkdir(parents=True, exist_ok=True)
    
    torch.save(stats,f"results/{args.save_name}_statistics.pt")
    torch.save(predictions,f"results/{args.save_name}_predictions.tar")