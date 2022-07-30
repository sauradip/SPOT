import torch
import numpy as np


def TemporalHalf(snip):

    B,C,T  = snip.size()

    t_half = int(T//2)
    half_snip = torch.zeros_like(snip)
    if np.random.rand() < 0.5:
        half_snip[:,:,0:t_half] = snip[:,:,0:t_half]
    else:
        half_snip[:,:,t_half:T] = snip[:,:,t_half:T]
    
    return half_snip

def TemporalReverse(snip):

    B,C,T = snip.size()
    reverse_snip = torch.flip(snip,[2])
    # reverse_snip = snip[:,:,::-1]

    return reverse_snip

def TemporalCutOut(snip):
    prob = 1.0
    B,C,T = snip.size()

    drops = np.random.rand(T) < prob
    drops[0] = False
    aug_idx = list(range(T))
    for i, drop in enumerate(drops):
        if drop:
            aug_idx[i] = aug_idx[i-1]
    cut_snip = snip[:,:,aug_idx]

    return cut_snip

def RandAugment(snip):

    rnd = np.random.rand()
    if rnd < 0.25:
        return snip
    elif rnd < 0.5:
        return TemporalHalf(snip)
    # elif rnd < 0.75:
    #     return TemporalReverse(snip)
    else:
        return TemporalCutOut(snip)


## from Learning Reprsentational Invariances for Data-Efficient Action Recognition ##
