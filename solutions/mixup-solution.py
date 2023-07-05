import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init

import numpy as np
import torch

def mixup_data(x, y=None, num_classes=1000, alpha=1., device=None):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    batch_size = x.size()[0]
    
    #TODO : add the device parameter
    # Get lambda values for each sample in batch
    lam = torch.tensor(np.random.beta(alpha, alpha, batch_size).astype('float32'), device=device)
    # Randomly permute the order of the batch to mix 2 samples of the batch : the ordered ones and the the permuted ones 
    s_index = torch.randperm(batch_size)

    mixed_x = lam.view(batch_size, 1, 1, 1) * x + (1 - lam).view(batch_size, 1, 1, 1) * x[s_index, :]
    if y == None: return mixed_x
    
    if len(y.size()) == 1:
        y = torch.nn.functional.one_hot(y, num_classes)
    mixed_y = lam.view(batch_size, 1) * y + (1 - lam).view(batch_size, 1) * y[s_index, :]

    return mixed_x, mixed_y
