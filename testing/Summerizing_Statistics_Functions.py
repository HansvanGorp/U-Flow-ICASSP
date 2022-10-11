# %% imports
# standard libraries
import torch
import torch.nn as nn 

# local import

# %% initialize stats
class stats_object(nn.Module):
    def __init__(self,no_samples, no_patients = 70):
        super(stats_object, self).__init__()    
        self.TST_mean = torch.zeros(no_patients)
        self.TST_std  = torch.zeros(no_patients)
        
        self.TIS = torch.zeros(no_patients, no_samples, 5)
        self.TIS_means  = torch.zeros(no_patients,5)
        self.TIS_stds   = torch.zeros(no_patients,5)
        
        self.RA_mean = torch.zeros(no_patients)
        self.RA_std  = torch.zeros(no_patients)
        
        self.NA_mean = torch.zeros(no_patients)
        self.NA_std  = torch.zeros(no_patients)
        
    def forward(self,x):
        return x

# %% Total Sleep Time (TST)
def get_TST(labels):
    labels2 = labels.clone()
    labels2[labels==0]=-1
    TST = (labels2!=-1).sum(dim=1)/2
    return TST

# %% Time in Stage(TIS)
def get_TIS(labels):
    TIS = torch.zeros(labels.size(0),5)
    for i in range(5):
        TIS[:,i] = (labels==i).sum(dim=1)/2
    return TIS

# %% split into bouts
def split_into_bouts(labels):    
    bouts = []
    
    for i in range(labels.size(0)):
        labels_here = labels[i,:]
        bouts_here = torch.tensor_split(labels_here, torch.where(torch.diff(labels_here) != 0)[0]+1)
        bouts.append(bouts_here)
        
    return bouts

# %% REM and NREM no_awakenings
def get_RA(bouts):
    no_awakenings = torch.zeros(len(bouts))
    for i,bout in enumerate(bouts):
        for j,b in enumerate(bout):
            if j != 0:
                if bout[j][0] == 0 and bout[j-1][0] == 4:
                    no_awakenings[i]+=1
    return no_awakenings

def get_NA(bouts):
    no_awakenings = torch.zeros(len(bouts))
    for i,bout in enumerate(bouts):
        for j,b in enumerate(bout):
            if j != 0:
                if bout[j][0] == 0 and bout[j-1][0] != 4:
                    no_awakenings[i]+=1
    return no_awakenings

# %% get all stats from a scoring
def create_stats_from_scoring(stats,labels,i):
    # TST
    TST = get_TST(labels)
    stats.TST_mean[i] = TST.mean()
    stats.TST_std[i]  = TST.std()
    
    # time in stage
    TIS = get_TIS(labels)
    stats.TIS_means[i] = TIS.mean(dim=0)
    stats.TIS_stds[i]  = TIS.std(dim=0)
    
    # get bouts
    bouts = split_into_bouts(labels)
    
    # REM arousals
    RA = get_RA(bouts)
    stats.RA_mean[i] = RA.mean()
    stats.RA_std[i]  = RA.std()
    
    # NREM arousals
    NA = get_NA(bouts)
    stats.NA_mean[i] = NA.mean()
    stats.NA_std[i]  = NA.std()
    
    return stats