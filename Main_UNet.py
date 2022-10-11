# %% imports
# standard libraries
import torch
import argparse
from pathlib import Path

# local import
import dataloading
import UNet
import Train_UNet

# %% seed
torch.random.manual_seed(0)

# %%
if __name__ == "__main__":
    # %% user defined input
    parser = argparse.ArgumentParser(description='train Unet baseline')
    
    # network architecture
    parser.add_argument('-epoch_encoder_channels',type=int,default=32)
    parser.add_argument('-u_channels',type=int,default=100)
    parser.add_argument('-L',type=int,default=8)
    
    
    # # training parameters
    parser.add_argument('-lr',type=float,default=1e-4)
    
    # # dataloader parameters
    parser.add_argument('-batch_size',type=float,default=2)
    parser.add_argument('-epochs',type=int,default=200)
    
    # # dataset
    parser.add_argument('-no_samples_per_epoch',type=int,default=4)
    parser.add_argument('-data_loc',type=str)
    
    # # save name
    parser.add_argument('-save_name',type=str)
    
    # device
    parser.add_argument('-device',type=str,default="cuda:0")
    
    # parse    
    args = parser.parse_args()
    
    # %% create dataloader
    dataloader_train, dataloader_val, args = dataloading.getDataLoader(args)
    print('Loaded data')
    
    # %% load Network
    Network_UNet = UNet.UNet(args)    
    Network_UNet = Network_UNet.to(args.device)
    print('Loaded Likelihood Network')
    
    # %% create optimizers
    optimizer = torch.optim.Adam(Network_UNet.parameters(), lr=args.lr)
    print('Created optimizers')
    
    # %% create relevant callback folder
    Path(f"checkpoints/{args.save_name}").mkdir(parents=True, exist_ok=True)
    
    # %% LR scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.1, last_epoch=- 1, verbose=False)
        
    # %% train
    print('Start Training\n')
    Train_UNet.Train(args, Network_UNet, optimizer, dataloader_val, dataloader_train, scheduler)
    print('\n\nDone!\n')
    