# %% imports
# standard libraries
import torch
import time

# local imports
import Prints


# %% CE loss function
def CE_loss(logits,scoring):
    #unscored regions should be wake
    scoring_adjusted = scoring.clone()
    scoring_adjusted[scoring == -1] = 0
    
    # adjust the logits dimensions
    logits_adjusted = logits.transpose(2,1)
    
    #calculate cross entropy loss
    loss = torch.nn.functional.cross_entropy(logits_adjusted, scoring_adjusted)
    
    return loss

# %% go over a full epoch of a dataset
def go_over_epoch(optimizer,Network_Unet,dataloader,args,epoch,global_step,start_time,dataloader_val,train=True):
    torch.cuda.empty_cache()
    
    # loop over an entire epoch of data
    for i,(PSG, scoring, mask) in enumerate(dataloader):
        if train == True:
            # intermediate print
            Prints.print_intermediate(epoch,i,start_time,args)
            global_step += 1
        
        # push to device
        PSG = PSG.to(args.device)
        scoring = scoring.to(args.device)
        mask = mask.to(args.device)
        
        # Network i/o
        if train == True:
                optimizer.zero_grad()
                logits = Network_Unet(PSG)
        else:
            with torch.no_grad():
                logits = Network_Unet(PSG)
        
        # add all together
        loss = CE_loss(logits,scoring)
        
        if train == True:
            loss.backward()
            optimizer.step()
            
    return global_step

    
# %% enitre training loop
def Train(args,Network_Unet, optimizer, dataloader_val, dataloader_train, scheduler):    
    # initial saves
    global_step = 0
    
    start_time = time.time()
    
    # save initial checkpoint and arguments
    torch.save(args, f"checkpoints/{args.save_name}/args.tar")
    torch.save(Network_Unet.state_dict(), f"checkpoints/{args.save_name}/model.tar")
    
    # initial test
    Network_Unet.eval()
    global_step = go_over_epoch(optimizer,Network_Unet,dataloader_val,args,0,global_step,start_time,dataloader_val,train=False)
    
    # %% go over an epoch
    for epoch in range(args.epochs):
        Prints.print_epoch()    
            
        # train
        Network_Unet.train()
        global_step = go_over_epoch(optimizer,Network_Unet,dataloader_train,args,epoch,global_step,start_time,dataloader_val,train=True)
        
        # validation
        Network_Unet.eval()
        global_step = go_over_epoch(optimizer,Network_Unet,dataloader_val,args,epoch,global_step,start_time,dataloader_val,train=False)
        
        # save checkpoint
        torch.save(Network_Unet.state_dict(), f"checkpoints/{args.save_name}/model.tar")
            
        # step
        scheduler.step()
        
    return global_step