# %% imports
# standard libraries
import torch
import argparse
from pathlib import Path

# local import
import dataloading
import Train_UFlow
import UFlow_encoder
import UFlow_decoder

# %% seed
torch.random.manual_seed(0)

# %%
if __name__ == "__main__":
    # %% user defined input
    parser = argparse.ArgumentParser(description='train branched glow for features extracted from PPG')
    
    # network architecture
    parser.add_argument('-input_size',type=int,default=1792*6)
    
    parser.add_argument('-factor',type=int,help='squeezing factor',default=2)
    parser.add_argument('-K',type=int,help='no. of steps of flow',default=6)
    parser.add_argument('-L',type=int,help='no. of squeezes',default=8)
    
    # # coupling
    parser.add_argument('-conditional_channels',type=int,help='no. channels in conditional net',default=32)
    
    parser.add_argument('-NN_channels',type=int,help='no. channels in coupling net',default=64)
    parser.add_argument('-coupling_bias',type=float,help='how much coupling bias to use',default=0.01)
    
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
    
    # %% load Uflow encoder
    Network_Uflow_encoder = UFlow_encoder.encoder(args)    
    Network_Uflow_encoder = Network_Uflow_encoder.to(args.device)
    print('Loaded Uflow encoder')
    
    # %% load Uflow decoder
    Network_Uflow_decoder = UFlow_decoder.decoder(args)    
    Network_Uflow_decoder = Network_Uflow_decoder.to(args.device)
    print('Loaded Uflow decoder')
        
    # %% actnorm init
    Train_UFlow.init_actnorm(Network_Uflow_encoder,Network_Uflow_decoder,dataloader_train,args)
    print('Initialized actnorm layers')
    
    # %% create optimizers
    params = list(Network_Uflow_encoder.parameters())+ list(Network_Uflow_decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    print('Created optimizer')
    
    # %% create relevant callback folder
    Path(f"checkpoints/{args.save_name}").mkdir(parents=True, exist_ok=True)
    
    # %% LR scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.1, last_epoch=- 1, verbose=False)
        
    # %% train
    print('Start Training\n')
    Train_UFlow.Train(args, Network_Uflow_encoder, Network_Uflow_decoder, optimizer, dataloader_val, dataloader_train, scheduler)
    print('\n\nDone!\n')
    