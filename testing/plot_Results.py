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
import argparse
import matplotlib.pyplot as plt
import numpy as np

# local import
from Compare import get_majority_vote

# %% seed
torch.random.manual_seed(0)

    
    
# %% plot a single hypnogram
def plot_hypnogram(y):
    y = y.float().cpu()
    
    # create time variable
    time = (torch.arange(y.size(0))*30)/3600
    
    # rearange order
    y2 = torch.zeros_like(y)
    y2[y==-1] = 4
    y2[y==0]  = 4
    y2[y==1]  = 2
    y2[y==2]  = 1
    y2[y==3]  = 0
    y2[y==4]  = 3
    
    # create REM line
    y2_R = y2.clone()
    t_R = time.clone()
    
    y2_R[y2!=3] = np.nan
    t_R[y2!=3] = np.nan
    
    # plot
    plt.plot(time,y2,c='k')
    plt.plot(t_R,y2_R,c='r',linewidth=4)
    plt.grid()
    plt.yticks([0,1,2,3,4],['N3','N2','N1','REM','Wake'])
    plt.ylim(-0.2,4.2)
    
# %% main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='plot results')
    parser.add_argument('-name',type=str,help='name of method to plot')
    parser.add_argument('-patient_id',type=int,help='id of recording to plot',default=34)
    args = parser.parse_args()
    
    # load preds
    predictions = torch.load(f"results/{args.name}_predictions.tar")
    predictions[predictions==-1] = 0
    
    # load masks
    masks = torch.load("results/ground_truth_masks.tar")
    
    # %% plot!
    min_t = 0
    max_t = 7.51
    plt.figure(figsize=(5,7.5))
    
    for i in range(6):        
        
        plt.subplot(8,1,i+1)        
        plot_hypnogram(predictions[args.patient_id,i,:].cpu())
        plt.ylabel('')
        plt.grid()
        
        if i == 0:
            plt.title(f"hypnograms for {args.name}")
        
        if i == 5:
             plt.xlabel("hours of sleep")
        else:
             plt.xticks([0,2,4,6,8],[])
        
        plt.xlim(min_t,max_t)
        
    # majority vote
    ax = plt.subplot(8,1,8)
    majority_vote = get_majority_vote(predictions, weighted_by_kappa = True, masks = masks)
    plot_hypnogram(majority_vote[args.patient_id,:].cpu())
    plt.ylabel('')
    plt.xlim(min_t,max_t)
    plt.grid()
    plt.xlabel("hours of sleep")
    plt.title(f"majority vote for {args.name}")

                
    # save and close
    Path("figures").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"figures//{args.name}.png",dpi=300,bbox_inches='tight')
    plt.close()
    
    
    