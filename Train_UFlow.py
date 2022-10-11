# %% imports
# standard libraries
import torch
import time
from datetime import datetime

# local imports
import Prints

# %% init actnorm
def init_actnorm(Network_Uflow_encoder,Network_Uflow_decoder,dataloader,args):
                 
    Network_Uflow_encoder.train()
    Network_Uflow_decoder.train()
    
    for i, (PSG, scoring, mask) in enumerate(dataloader):
        PSG     = PSG.to(args.device)
        scoring = scoring.to(args.device)
        if scoring.ndim==3:
            scoring = scoring[:,0,:]
        scoring = Network_Uflow_decoder.dequantize(scoring)
        break
    
    with torch.no_grad():
        C = Network_Uflow_encoder(PSG)
        z, logdet = Network_Uflow_decoder(scoring, C)

# %% go over a full epoch of a dataset
def go_over_epoch(optimizer,Network_Uflow_encoder, Network_Uflow_decoder,dataloader,args,epoch,global_step,start_time,dataloader_val,train=True):
    torch.cuda.empty_cache()
    
    # check if trian or eval mode
    if train == True:
        Network_Uflow_encoder.train()
        Network_Uflow_decoder.train()
    else:
        Network_Uflow_encoder.eval()
        Network_Uflow_decoder.eval()

    # loop over an entire epoch of data
    for i,(PSG, scoring, mask) in enumerate(dataloader):
        if train == True:
            # intermediate print
            Prints.print_intermediate(epoch,i,start_time,args)
            global_step += 1
        
        # push to device
        PSG     = PSG.to(args.device)
        scoring = scoring.to(args.device)
        scoring = Network_Uflow_decoder.dequantize(scoring)
        
        # zero grad
        optimizer.zero_grad()
        
        # Network i/o
        if train == True:
                C = Network_Uflow_encoder(PSG)
                z, logdet = Network_Uflow_decoder(scoring, C)
        else:
            with torch.no_grad():
                C = Network_Uflow_encoder(PSG)
                z, logdet = Network_Uflow_decoder(scoring, C)                
                
                
        #  losses
        nll, (p_z, logdet) = Network_Uflow_decoder.nll(z, logdet)
         
        # take averages
        loss = torch.mean(nll)
        
        if train == True:
            loss.backward()
            optimizer.step()

            
    return global_step

# %% enitre training loop
def Train(args, Network_Uflow_encoder, Network_Uflow_decoder, optimizer, dataloader_val, dataloader_train, scheduler):
    # initial saves
    global_step = 0
    
    start_time = time.time()
    
    # save initial checkpoint and arguments
    torch.save(args, f"checkpoints/{args.save_name}/args.tar")
    torch.save(Network_Uflow_encoder.state_dict(), f"checkpoints/{args.save_name}/encoder.tar")
    torch.save(Network_Uflow_decoder.state_dict(), f"checkpoints/{args.save_name}/decoder.tar")
    
    # initial test
    global_step = go_over_epoch(optimizer, Network_Uflow_encoder, Network_Uflow_decoder, dataloader_val,args,0,global_step,start_time,dataloader_val,train=False)
    
    # %% go over an epoch
    for epoch in range(args.epochs):
        Prints.print_epoch()    
            
        # train
        global_step = go_over_epoch(optimizer,Network_Uflow_encoder, Network_Uflow_decoder,dataloader_train,args,epoch,global_step,start_time,dataloader_val,train=True)
        
        # validation
        global_step = go_over_epoch(optimizer,Network_Uflow_encoder, Network_Uflow_decoder,dataloader_val,args,epoch,global_step,start_time,dataloader_val,train=False)
        
        # save checkpoint
        torch.save(Network_Uflow_encoder.state_dict(), f"checkpoints/{args.save_name}/encoder.tar")
        torch.save(Network_Uflow_decoder.state_dict(), f"checkpoints/{args.save_name}/decoder.tar")
        
        # step
        scheduler.step()
    
    return global_step