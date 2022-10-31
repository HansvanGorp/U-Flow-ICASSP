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
import UFlow_encoder
import UFlow_decoder
import Train_UFlow
import Summerizing_Statistics_Functions as SSF

# %% seed
torch.random.manual_seed(0)

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pretest U-Flow')
    parser.add_argument('-batch_size',type=float,help='batch size',default=1)
    parser.add_argument('-no_samples_per_epoch',type=int,help='how much samples to show every epoch',default=4)
    parser.add_argument('-save_name',type=str,help='name to save results under',default="U-Flow")
    parser.add_argument('-data_loc',type=str,help='location of data')
    parser.add_argument('-device',type=str,help='device to use',default="cuda:0")
    args = parser.parse_args()
    
    # %% create dataloader 
    dataloader_test, args = dataloading.getDataLoaderTest(args)
    print('Loaded data')
    
    no_patients = dataloader_test.dataset.__len__()
    no_samples = 100
    
    # %% load statedicts
    args2        = torch.load(f"..//checkpoints//{args.save_name}//args.tar")
    state_dict_encoder  = torch.load(f"..//checkpoints//{args.save_name}//encoder.tar")
    state_dict_decoder  = torch.load(f"..//checkpoints//{args.save_name}//decoder.tar")
    
    # %% load Uflow encoder
    Network_UFlow_encoder = UFlow_encoder.encoder(args2)    
    Network_UFlow_encoder = Network_UFlow_encoder.to(args.device)
    Network_UFlow_encoder.load_state_dict(state_dict_encoder)
    print('Loaded UFlow encoder')
    
    # %% load Uflow decoder
    Network_UFlow_decoder = UFlow_decoder.decoder(args2)    
    Network_UFlow_decoder = Network_UFlow_decoder.to(args.device)
    
    # init actnorm
    Train_UFlow.init_actnorm(Network_UFlow_encoder,Network_UFlow_decoder,dataloader_test,args)
    
    # load statedict
    Network_UFlow_decoder.load_state_dict(state_dict_decoder)
    
    print('Loaded UFlow decoder')
    
    # %% init stats
    stats = SSF.stats_object(no_samples)
    predictions = torch.zeros(no_patients,no_samples,1792)
    masks = torch.zeros(no_patients,1792).type(torch.bool)
    
    # %% go over the test set
    Network_UFlow_encoder.train()
    Network_UFlow_decoder.train()
    
    for i,(PSG, _, _) in enumerate(tqdm(dataloader_test)):
        # push to device
        PSG = PSG.to(args.device)
        
        # get context
        C = Network_UFlow_encoder(PSG)
        
        # predict many times
        for n in range(no_samples):
            with torch.no_grad():
                # sample z
                z =  torch.randn(1,1792*6).to(args.device)
                
                # reverse trhough network
                labels, logdet = Network_UFlow_decoder.reverse(z, C)
        
                # use argmax
                scoring = torch.argmax(labels,dim=1)
                
                # get rid of 5
                scoring[scoring==5]=0
                
                #save
                predictions[i,n,:] = scoring[0,:]
        
        stats = SSF.create_stats_from_scoring(stats,predictions[i,:,:],i)
        
        
    # %% save
    Path("results").mkdir(parents=True, exist_ok=True)
    
    torch.save(stats,f"results/{args.save_name}_statistics.pt")
    torch.save(predictions,f"results/{args.save_name}_predictions.tar")