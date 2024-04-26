import torch
import numpy as np
import torch.nn.functional as F
from scipy.stats import norm


def kl_loss_compute(pred, soft_targets, reduce=True):

    kl = F.kl_div(F.log_softmax(pred, dim=1),F.softmax(soft_targets, dim=1),reduction='none')

    if reduce:
        return torch.mean(torch.sum(kl, dim=1))
    else:
        return torch.sum(kl, 1)

def mutual_information_compute(y_1, y_2):
    y_1 = F.softmax(y_1, dim=-1)
    y_2 = F.softmax(y_2, dim=-1)
    return torch.sum((torch.log2((y_1 + y_2)/2) - (torch.log2(y_1) + torch.log2(y_2)) / 2), dim=1)


def loss_drop_gaussian(y_1, t, forget_rate, ind, noise_or_not, after_warmup=False):
    loss_pick= F.cross_entropy(y_1, t.long(), reduction='none')
    loss_pick = loss_pick.cpu()

    ind_sorted = np.argsort(loss_pick.data)
    loss_sorted = loss_pick[ind_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_sorted))

    pure_ratio = np.sum(noise_or_not[ind[ind_sorted[:num_remember]]])/float(num_remember)

    ind_update = ind_sorted[:num_remember]
    
    if after_warmup==False:  
        loss = torch.mean(loss_pick[ind_update])
    
    else:
        # gaussian distribution
        x = torch.linspace(-5, 5, loss_pick[ind_update].shape[0])
        gaussian_weights = norm.pdf(x, loc=0, scale=10) 
        gaussian_weights = np.max(gaussian_weights) - gaussian_weights 
        gaussian_weights = 1 + gaussian_weights 
        gaussian_weights = torch.tensor(gaussian_weights)
        weighted_losses = loss_pick[ind_update] * gaussian_weights
        loss = torch.mean(weighted_losses)


    ind_difference = np.setdiff1d(ind_sorted, ind_update)
    ind_noisy = ind_sorted[ind_difference]
    return loss, pure_ratio, ind_update, ind_noisy 




class one_hot_CrossEntropy(torch.nn.Module):

    def __init__(self):
        super(one_hot_CrossEntropy, self).__init__()

    def forward(self, x, y):
        P_i = torch.nn.functional.softmax(x, dim=1)
        loss = y * torch.log(P_i + 0.0000001)
        loss = -torch.mean(torch.sum(loss, dim=1), dim=0)
        return loss
