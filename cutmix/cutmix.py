import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init

import numpy as np
import torch


def cut_mask(x1, x2, y1, y2, batch_size, W, H, device=None):
    
    ### TODO
    mask_ext, mask_int = None, None
    
    return mask_ext, mask_int

def cutmix_data(x, y, num_classes=1000, device=None):

    '''Compute the cutmix data. Return mixed inputs, pairs of targets (one hot style)'''
    size = x.size()
    if len(size) == 4:   ## Get the size of image. Usually W=Z
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception
    batch_size = x.size()[0]
    
    lam = torch.rand(batch_size, device=device)
    s_index = torch.randperm(batch_size)      # Shuffle index
    
    rand_x = torch.randint(W, (batch_size,), device=device)
    rand_y = torch.randint(H, (batch_size,), device=device)
    cut_rat = torch.sqrt(1. - lam) ## cut ratio according to the random lambda
    
    x1 = torch.clip(rand_x - rand_x / 2, min=0).long()
    x2 = torch.clip(rand_x + rand_x / 2, max=W-1).long()
    y1 = torch.clip(rand_y - rand_y / 2, min=0).long()
    y2 = torch.clip(rand_y + rand_y / 2, max=H-1).long()
    
    lam = 1 - (x2-x1) * (y2-y1) / (W*H) # Adjust lambda to the exact area ratio
    
    mask_ext, mask_int = cut_mask(x1, x2, y1, y2, batch_size, W, H, device)
    if mask_ext!=None and mask_int!=None: 
        mixed_x = mask_ext * x + mask_int * x[s_index, :]
    else :
        mixed_x = x
        for i in range(len(mixed_x)):
            mixed_x[i,:,x1[i]:x2[i],y1[i]:y2[i]] = x[s_index[i],:,x1[i]:x2[i],y1[i]:y2[i]]
    
    if len(y.size()) == 1:
        y = torch.nn.functional.one_hot(y, num_classes)
    mixed_y = lam.view(batch_size, 1) * y + (1 - lam).view(batch_size, 1) * y[s_index, :]

    return mixed_x, mixed_y





