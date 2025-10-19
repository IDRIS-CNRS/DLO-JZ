#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import idr_torch
import torch
import torch.distributed as dist
from torch.autograd import Function
from torch.distributed.distributed_c10d import ProcessGroup

tp_grp: ProcessGroup | None = None
tp_rank: int | None = None
tp_degree: int | None = None

dp_grp: ProcessGroup | None = None
dp_rank: int | None = None
dp_degree: int | None = None



class Duplication(Function):
    @staticmethod
    def forward(ctx, x):
        ###### BALISE 1a ######
        ...
        ###### FIN BALISE 1a ######

    @staticmethod
    def backward(ctx, grad_output):
        ###### BALISE 1b / BALISE 9a ######
        ...
        ###### FIN BALISE 1b / BALISE 9a ######


class AllReduce(Function):
    @staticmethod
    def forward(ctx, x):
        ###### BALISE 1c / BALISE 9b ######
        ...
        ###### FIN BALISE 1c / BALISE 9b ######

    @staticmethod
    def backward(ctx, grad_output):
        ###### BALISE 1d ######
        ...
        ###### FIN BALISE 1d ######


class AllGather(Function):
    @staticmethod
    def forward(ctx, x):
        out_list = [torch.zeros_like(x) for _ in range(tp_degree)]
        ###### BALISE 9c ######
        dist.all_gather(out_list, x)
        ###### FIN BALISE 9c ######
        out = torch.cat(out_list, dim=-1)
        return out

    @staticmethod
    def backward(ctx, grad_outputs):
        dim = grad_outputs.size()[-1] // tp_degree
        return grad_outputs[..., dim * tp_rank : dim * (tp_rank + 1)]


def init(tp: int):
    global tp_grp, tp_rank, tp_degree, dp_grp, dp_rank, dp_degree


    ###### BALISE 8 ######
    tp_rank = idr_torch.rank
    tp_degree = idr_torch.world_size
    ###### FIN BALISE 8 ######


    ###### BALISE 10 ######

    ###### FIN BALISE 10 ######

