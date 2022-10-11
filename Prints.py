# %% imports
# standard libraries
import numpy as np
import time


# %% prints
def print_epoch():
    print("")
    print("----------------------------------------------")
    print("| epoch | perc. |  time taken |    time left |")
    
def print_intermediate(epoch_id,batch_id,start_time,args):
    percentage = int(100 * (batch_id+1) / args.batches_per_epoch)
    
    epoch = epoch_id+1
    time_taken = time.time()-start_time
    
    hours_taken = int(np.floor(time_taken/3600))
    time_inter = time_taken%3600 
    minutes_taken = int(np.floor(time_inter/60))
    seconds_taken = int(time_inter%60)
    
    epochs_left = args.epochs - epoch_id - 1
    batches_left_here = args.batches_per_epoch-batch_id -1
    total_no_batches_left = epochs_left*args.batches_per_epoch + batches_left_here
    
    total_no_batches_seen = batch_id+1+epoch_id*args.batches_per_epoch
    
    time_per_batch = time_taken/total_no_batches_seen
    time_left = time_per_batch*total_no_batches_left
    
    hours_left = int(np.floor(time_left/3600))
    time_inter = time_left%3600 
    minutes_left = int(np.floor(time_inter/60))
    seconds_left = int(time_inter%60) 
    
    
    print(f"| {epoch:>5} | {percentage:>4}% | {hours_taken:>2}h{minutes_taken:>3}m{seconds_taken:>3}s | {hours_left:>3}h{minutes_left:>3}m{seconds_left:>3}s |", end="\r")
 