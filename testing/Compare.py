# %% imports
# standard libraries
import torch
from sklearn.metrics import cohen_kappa_score
import argparse

# %% seed
torch.random.manual_seed(0)

# %% KL divergence
def calculate_KL(mu1, sigma1, mu2, sigma2):
    for i,sigma in enumerate(sigma1):
        if sigma < 1e-3:
            sigma1[i] += 1e-2
            
    for i,sigma in enumerate(sigma2):
        if sigma < 1e-3:
            sigma2[i] += 1e-2
    
    KL = torch.log(sigma2/sigma1) + (sigma1**2 + (mu1-mu2)**2)/(2*sigma2**2) - 1/2
    return KL

def print_KL_divergence(gt_stats,cp_stats):
    KL_TST = calculate_KL(gt_stats.TST_mean, gt_stats.TST_std, cp_stats.TST_mean, cp_stats.TST_std)
    KL_N1  = calculate_KL(gt_stats.TIS_means[:,1], gt_stats.TIS_stds[:,1], cp_stats.TIS_means[:,1], cp_stats.TIS_stds[:,1])
    KL_N2  = calculate_KL(gt_stats.TIS_means[:,2], gt_stats.TIS_stds[:,2], cp_stats.TIS_means[:,2], cp_stats.TIS_stds[:,2])
    KL_N3  = calculate_KL(gt_stats.TIS_means[:,3], gt_stats.TIS_stds[:,3], cp_stats.TIS_means[:,3], cp_stats.TIS_stds[:,3])
    KL_REM = calculate_KL(gt_stats.TIS_means[:,4], gt_stats.TIS_stds[:,4], cp_stats.TIS_means[:,4], cp_stats.TIS_stds[:,4])
    
    KL_RA  = calculate_KL(gt_stats.RA_mean, gt_stats.RA_std, cp_stats.RA_mean, cp_stats.RA_std)
    KL_NA  = calculate_KL(gt_stats.NA_mean, gt_stats.NA_std, cp_stats.NA_mean, cp_stats.NA_std)

    print("\nKL divergences for overnight sleep statistics:")
    print(f"TST              = {KL_TST.mean():.4}")
    print(f"Time in N1       = {KL_N1.mean():.4}")
    print(f"Time in N2       = {KL_N2.mean():.4}")
    print(f"Time in N3       = {KL_N3.mean():.4}")
    print(f"Time in REM      = {KL_REM.mean():.4}")
    
    print(f"REM  awakenings  = {KL_RA.mean():.4}")
    print(f"NREM awakenings  = {KL_NA.mean():.4}")

# %% get majority vote
def get_majority_vote(predictions, weighted_by_kappa = False, masks = None, remove_ties = False):
    
    # do not weigh by kappa
    if weighted_by_kappa == False:
        votes = torch.zeros(predictions.size(0),6,predictions.size(2))
        
        #go over stages
        for i in range(6):
            votes[:,i,:] = (predictions == (i-1)).sum(dim=1)
        
        # majority vote
        majority_vote = votes.argmax(dim=1)-1
        
        if remove_ties == False:
            return majority_vote
        else:
            # if remove ties is true, make all the ties equal to -2
            max_votes,_ = votes.max(dim=1)
            splits = (votes == max_votes.unsqueeze(1).repeat(1,6,1))
            splits = splits.sum(dim=1)>1
            majority_vote[splits] = -2
            return majority_vote
    
    # do weigh by kappa
    else:
        # first get the majority vote not weighted by kappa
        majority_vote = get_majority_vote(predictions, weighted_by_kappa = False, masks = None, remove_ties = True)
        votes = torch.zeros(predictions.size(0),6,predictions.size(2))
        
        # loop over each night
        for n in range(predictions.size(0)):
            majority_vote_here_masked = majority_vote[n,masks[n,:]]
            
            # calculate kappa for each scorer seperately
            kappas = torch.zeros(predictions.size(1))
            for s in range(predictions.size(1)):
                scorer_vote_here_masked = predictions[n,s,masks[n,:]]
                kappas[s] = cohen_kappa_score(majority_vote_here_masked,scorer_vote_here_masked)
                
                # create the weighted majority vote from the kappas and the votes
                for i in range(6):
                    votes[n,i,:] += (predictions[n,s,:] == (i-1))*kappas[s]
            
        # get weighted majority vote
        majority_vote_weighted = votes.argmax(dim=1)-1
        return majority_vote_weighted
    
# %% majority vote stats
def print_majority_vote_stats(gt_predictions,cp_predictions, masks):
    # majority votes
    gt_majority = get_majority_vote(gt_predictions, weighted_by_kappa = True, masks = masks)
    cp_majority = get_majority_vote(cp_predictions, weighted_by_kappa = True, masks = masks)
    
    # accuracy and kappa
    gt_majority_masked = gt_majority[masks]
    cp_majority_masked = cp_majority[masks]
    
    acc = 100*(gt_majority_masked==cp_majority_masked).sum()/gt_majority_masked.size(0)
    kappa = cohen_kappa_score(gt_majority_masked,cp_majority_masked)
    
    print("\nMajority voting results:")
    print(F"accuracy = {acc:.4}")
    print(F"kappa    = {kappa:.3}\n")
    
    
# %% main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='compare results')
    parser.add_argument('-name',type=str,help='name to compare against')
    args = parser.parse_args()
    # load preds
    gt_predictions = torch.load("results/ground_truth_predictions.tar")   
    cp_predictions = torch.load(f"results/{args.name}_predictions.tar")   

    # load stats
    gt_stats = torch.load("results/ground_truth_statistics.pt")
    cp_stats = torch.load(f"results/{args.name}_statistics.pt")
    
    # load masks
    masks = torch.load("results/ground_truth_masks.tar")
    
    # summerizing stats comparison
    print_KL_divergence(gt_stats,cp_stats)
    
    # predictions comparison
    print_majority_vote_stats(gt_predictions,cp_predictions,masks)
    
    