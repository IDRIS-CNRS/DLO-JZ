# Standard library imports
import argparse
import contextlib
import os
import random

# Third-party library imports
import numpy as np
import torch
import torch.distributed as dist
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

# Specific module imports
from apex.parallel.LARC import LARC
from dlojz_chrono import Chronometer
from lion import Lion
import apex
import idr_torch

# Specific torch imports
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.profiler import profile, tensorboard_trace_handler, ProfilerActivity, schedule
from torchmetrics.classification import Accuracy

VAL_BATCH_SIZE=512

# Fix random seeds
random.seed(123)                           
np.random.seed(123)
torch.manual_seed(123) 


def main():
    parser = argparse.ArgumentParser()                                 
    parser.add_argument('-b', '--batch-size', default=128, type =int,  
                        help='Batch size per GPU')                     
    parser.add_argument('-e','--epochs', default=1, type=int,          
                        help='Number of total epochs to run')          
    parser.add_argument('--image-size', default=224, type=int,         
                        help='Image size')                             
    parser.add_argument('--lr', default=0.003, type=float,             
                        help='Learning rate')                          
    parser.add_argument('--mom', default=0.9, type=float,              
                        help='Momentum')                               
    parser.add_argument('--wd', default=0., type=float,                
                        help='Weight decay')                           
    parser.add_argument('--test', default=False, action='store_true',  
                        help='Test 50 iterations')                     
    parser.add_argument('--prof', default=False, action='store_true',
                        help='Enable pytorch profiling')
    parser.add_argument('--chkpt', default=False, action='store_true',
                        help='Save last checkpoint')
    args = parser.parse_args()
    train(args)
    

def train(args):
    
    ## Define Chronometer
    chrono = Chronometer(args.test, idr_torch.rank) 
   
    # Configure distribution method: define address and port of the master node 
    # and initialise communication backend (NCCL)
    dist.init_process_group(backend='nccl', 
                            init_method='env://', 
                            world_size=idr_torch.size, 
                            rank=idr_torch.rank)
    
    #########  MODEL ########
   
    # Define model
    model = models.resnet18(num_classes=10)
    model.name = 'Resnet-18'
    
    if idr_torch.rank == 0: 
        print(f'model: {model.name}')
    
    # Send to GPU
    torch.cuda.set_device(idr_torch.local_rank)
    gpu = torch.device("cuda")
    model = model.to(gpu)

    # Distribute batch size (mini-batch)                             
    num_replica = idr_torch.size                                     
    mini_batch_size = args.batch_size                                
    global_batch_size = mini_batch_size * num_replica                
                                                                     
    if idr_torch.rank == 0:                                          
        print(f'global batch size: {global_batch_size} - mini batch size: {mini_batch_size}')
                                                                     
    # Define loss function (criterion) 
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)     
    
    # Define optimizer
    #optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.mom, weight_decay=args.wd)
    #optimizer = torch.optim.AdamW(model.parameters(), args.lr, betas=(args.mom, 0.999), weight_decay=args.wd)
    #optimizer = apex.optimizers.FusedLAMB(model.parameters(), args.lr, betas=(args.mom, 0.999), weight_decay=args.wd)
    #optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.mom, weight_decay=args.wd)
    optimizer = Lion(model.parameters(), lr=args.lr, weight_decay=args.wd)
    wrapped_optimizer = optimizer
    
    # Distribute model
    model = DistributedDataParallel(model, device_ids=[idr_torch.local_rank])

    if idr_torch.rank == 0: 
        print(f'Optimizer: {wrapped_optimizer}')    
        
    # Define GradScaler for AMP
    scaler = GradScaler()
        
     
    ######  DATALOADER ######
    
    # Define a transform to pre-process the training image
    transform = transforms.Compose([ 
        transforms.RandomHorizontalFlip(),   # Data Augmentation - Horizontal Flip
        transforms.RandomCrop(32, padding=4),# Data Augmentation - Random Crop
        transforms.ToTensor()                # Convert PIL Image to a Tensor
        ])
    
    # Define train_loader
    train_dataset = torchvision.datasets.CIFAR10(root=os.environ['ALL_CCFRSCRATCH']+'/CIFAR_10', 
                                                 train=True, 
                                                 download=False, 
                                                 transform=transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=idr_torch.size,
                                                                    rank=idr_torch.rank,
                                                                    shuffle=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=mini_batch_size,
                                               shuffle=False,
                                               sampler=train_sampler,
                                               num_workers=4,
                                               persistent_workers=True,
                                               pin_memory=True,
                                               prefetch_factor=2)
    
    # Define a transform to pre-process the evaluation image 
    val_transform = transforms.Compose([
                    transforms.ToTensor() # Convert PIL Image to a Tensor
                    ])
    
    # Define val_loader
    val_dataset = torchvision.datasets.CIFAR10(root=os.environ['ALL_CCFRSCRATCH']+'/CIFAR_10',
                                               train=False, 
                                               download=False, 
                                               transform=val_transform)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset,
                                                                 num_replicas=idr_torch.size,
                                                                 rank=idr_torch.rank,
                                                                 shuffle=False)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,    
                                             batch_size=VAL_BATCH_SIZE,
                                             shuffle=False,
                                             sampler=val_sampler,
                                             num_workers=4,
                                             persistent_workers=True,
                                             pin_memory=True,
                                             prefetch_factor=2)
    
    # Define Learning Rate Scheduler
    N_batch = len(train_loader)
    N_val_batch = len(val_loader)
    N_val = len(val_dataset)
    
    #scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=N_batch, epochs=args.epochs)
    
    chrono.start()                                  
    
    ## Initialisation accuracy log 
    if idr_torch.rank == 0: 
        accuracies = []
    val_loss = torch.Tensor([0.]).to(gpu)                  # send to GPU
    val_accuracy = torch.Tensor([0.]).to(gpu)              # send to GPU

    # Torchmetrics
    train_metric_acc = Accuracy(task="multiclass", num_classes=10).to(gpu)
    valid_metric_acc = Accuracy(task="multiclass", num_classes=10).to(gpu)

    # Pytorch profiler setup
    prof =  profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    schedule=schedule(wait=1, warmup=1, active=12),
                    on_trace_ready=tensorboard_trace_handler('./profiler/' + os.environ['SLURM_JOBID']),
                    profile_memory=True,
                    record_shapes=True, 
                    with_stack=False,
                    with_flops=True
                    ) if args.prof else contextlib.nullcontext()
    
    
    ######## TRAINING #######
    with prof:
        for epoch in range(args.epochs):    
            
            train_sampler.set_epoch(epoch)
            chrono.dataload()                                   
                                                                 
            for i, (images, labels) in enumerate(train_loader):
                model.train()                                                
                csteps = i + 1 + epoch * N_batch                
                if args.test and csteps > 50: break              

                # distribution of images and labels to all GPUs
                images = images.to(gpu, non_blocking=True)
                labels = labels.to(gpu, non_blocking=True)

                chrono.dataload()
                chrono.training()
                chrono.forward()

                wrapped_optimizer.zero_grad()
                # Runs the forward pass with autocasting.
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                chrono.forward()
                chrono.backward()     

                scaler.scale(loss).backward()
                scaler.step(wrapped_optimizer)
                scaler.update()

                # Torchmetrics
                _, predicted = torch.max(outputs.data, 1)
                train_metric_acc(labels,predicted)     #update metric
                train_acc = train_metric_acc.compute() #sync all rank
                train_metric_acc.reset()               #reset
                
                chrono.backward()
                chrono.training()

                if idr_torch.rank == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc:{:.4f}'.format(
                          epoch + 1, args.epochs, i+1, N_batch, loss.item(),train_acc))                             
                    accuracies = []                             

                # scheduler update
                scheduler.step()
                
                # profiler update
                if args.prof: prof.step()

                chrono.dataload()                               
                                                                
                #### VALIDATION ############                    
                if (((i+1)%(N_batch//2)==0) or (args.test and i==49)):                                    
                    model.eval()                                
                    chrono.validation()
                    
                    for iv, (val_images, val_labels) in enumerate(val_loader): 
                        
                        if args.test and iv >= 20: break 
                                                               
                        # distribution of images and labels to all GPUs
                        val_images = val_images.to(gpu, non_blocking=True)
                        val_labels = val_labels.to(gpu, non_blocking=True)

                        # Runs the forward pass with no grade mode.
                        with torch.no_grad():
                            with autocast():
                                val_outputs = model(val_images)
                                loss = criterion(val_outputs, val_labels)

                        val_loss += loss       

                        # Torchmetrics
                        _, val_predicted = torch.max(val_outputs.data, 1)
                        valid_metric_acc(val_labels,val_predicted)#update metric
                        
                    val_acc = valid_metric_acc.compute() #sync all rank at evaluation end !
                    valid_metric_acc.reset()             #reset
                    val_loss /= iv                       #loss mean on iv

                    chrono.validation()                                
                                                                       
                    if not args.test and idr_torch.rank == 0:          
                        print('##EVALUATION STEP##')                   
                        print('Epoch [{}/{}], Validation Loss: {:.4f}, Validation Accuracy: {:.4f}'.format(epoch + 1, args.epochs,
                        val_loss.item(), val_acc))
                        print('Learning Rate: {:.4f}'.format(scheduler.get_last_lr()[0]))
                        print(">>> Validation complete in: " + str(chrono.val_time))    

                    ## Clear val_loss
                    val_loss -= val_loss
                     
        # Torchmetrics epoch reset
        train_metric_acc.reset()
        valid_metric_acc.reset()
            
    ## Be sure all process finish at the same time to avoid incoherent logs at the end of process !
    dist.barrier()
                                  
    chrono.display(N_val_batch)
    if idr_torch.rank == 0:
        print(">>> Number of batch per epoch: {}".format(N_batch)) 
        print(f'Max Memory Allocated {torch.cuda.max_memory_allocated()} Bytes') 
        
    # Save last checkpoint
    if args.chkpt and idr_torch.rank == 0:
        chkt_dict['final_model'] = model.state_dict()
        checkpoint_path = f"checkpoints/{os.environ['SLURM_JOBID']}.pt"
        torch.save(chkt_dict, checkpoint_path)
        print("Last epoch checkpointed to " + checkpoint_path)

if __name__ == '__main__':
    if idr_torch.rank == 0:
        print(">>> Training on ", len(idr_torch.hostnames), " nodes and ", idr_torch.size, " processes")
    main()
