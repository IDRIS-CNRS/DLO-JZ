#############################################################
# COMPUTE PI                                                
# Inspired by IDRIS MPI course, Exercise 3: 
# http://www.idris.fr/media/formations/mpi/idrismpien.pdf 
#############################################################
#
#         _1
# pi =  _/   4 / (1 + x*x) dx 
#      0
#
#############################################################

from math import pi
import torch

### TODO_0: import torch.distributed and initialize parallel environment
import idr_torch
import torch.distributed as dist
dist.init_process_group(backend='nccl',
                        init_method='env://',
                        world_size=idr_torch.size,
                        rank=idr_torch.rank)
torch.cuda.set_device(idr_torch.local_rank)

# define number of intervals and interval width
nblocks = int(1e6)
block_width = 1./nblocks

# initialize sum_pi to zero on GPU
sum_pi = torch.tensor([0],dtype=float).cuda()

### TODO_1: get rank and size, then uncomment assertion
rank = idr_torch.rank
size = idr_torch.size
assert nblocks%size==0

### TODO_2: define istart and iend for each rank, then uncomment print
istart = int(rank * (nblocks / size))
iend = int((rank + 1) * (nblocks / size))
print(f'rank {rank} : istart = {istart}, iend = {iend}')

### TODO_3: parallelize loop
for i in range(istart,iend):
    # define coordinate at the middle of the interval
    x = block_width*(i+0.5)
    # compute block area
    sum_pi = sum_pi + block_width * (4. / (1. + x*x))

### TODO_4: make ranks communicate so that each rank stores the value pi
dist.all_reduce(sum_pi, op=dist.ReduceOp.SUM)

print(f'Pi = {sum_pi}')
print(f'Error = {sum_pi - pi}')
