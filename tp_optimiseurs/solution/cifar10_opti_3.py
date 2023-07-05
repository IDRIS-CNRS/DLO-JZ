import os                                        #*************************************************************
import contextlib                                #
import argparse                                  #
import torchvision                               #
import torchvision.transforms as transforms      #
import torchvision.models as models              #                   
import torch                                     #
import numpy as np                               #   ###       DON'T MODIFY    ####################
import apex                                      #
from apex.parallel.LARC import LARC              #
                                                 #
import idr_torch                                 #
from dlojz_chrono import Chronometer             #
                                                 #
import random                                    #
random.seed(123)                                 #
np.random.seed(123)                              #
torch.manual_seed(123)                           #*************************************************************

## import ... ## Add here the libraries to import
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.profiler import profile, tensorboard_trace_handler, ProfilerActivity, schedule

def main():                                                            #******************************************
    parser = argparse.ArgumentParser()                                 #
    parser.add_argument('-b', '--batch-size', default=128, type =int,  #
                        help='batch size per GPU')                     #
    parser.add_argument('-e','--epochs', default=1, type=int,          #
                        help='number of total epochs to run')          #
    parser.add_argument('--image-size', default=224, type=int,         #
                        help='Image size')                             #
    parser.add_argument('--lr', default=0.003, type=float,             #      ##    DON'T MODIFY    ########
                        help='learning rate')                          #
    parser.add_argument('--mom', default=0.9, type=float,              #
                        help='momentum')                               #
    parser.add_argument('--wd', default=0., type=float,                #
                        help='weight decay')                           #
    parser.add_argument('--test', default=False, action='store_true',  #
                        help='Test 50 iterations')                     #
    parser.add_argument('--findlr', default=False, action='store_true',#
                        help='LR finder')                              #******************************************
    parser.add_argument('--prof', default=False, action='store_true',
                        help='PROF implementation')

    args = parser.parse_args()

    train(args)
    

VAL_BATCH_SIZE=512


def train(args):
    
    ## chronometer initialisation (test and rank)
    chrono = Chronometer(args.test, idr_torch.rank)       ### DON'T MODIFY ### 
    
    # configure distribution method: define address and port of the master node and initialise communication backend (NCCL)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=idr_torch.size, rank=idr_torch.rank)
    
    # define model
    torch.cuda.set_device(idr_torch.local_rank)
    gpu = torch.device("cuda")
    model = models.resnet18()
    model = model.to(gpu)

    
    model.name = 'Resnet-18'
    if idr_torch.rank == 0: print(f'model: {model.name}')  ### DON'T MODIFY ###
    
    
    # distribute batch size (mini-batch)                             #*************************************************
    num_replica = idr_torch.size                                     #
    mini_batch_size = args.batch_size                                #
    global_batch_size = mini_batch_size * num_replica                #
                                                                     #          ### DON'T MODIFY ################## 
    if idr_torch.rank == 0:                                          #
        print(f'global batch size: {global_batch_size} - mini batch size: {mini_batch_size}')
                                                                     #*************************************************
    
    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    #### TO MODIFY #####
    optimizer = apex.optimizers.FusedLAMB(model.parameters(), args.lr, betas=(args.mom, 0.999), weight_decay=args.wd)
    
    wrapped_optimizer = optimizer

    
    if idr_torch.rank == 0: print(f'Optimizer: {wrapped_optimizer}')    ### DON'T MODIFY ###
        
    model = DistributedDataParallel(model, device_ids=[idr_torch.local_rank])
        
    # Creates a GradScaler once at the beginning of training.
    scaler = GradScaler()
        
    #########  DATALOADER ############ 
    # Define a transform to pre-process the training images.

    transform = transforms.Compose([ 
        transforms.RandomHorizontalFlip(),              # Horizontal Flip - Data Augmentation
        transforms.ToTensor()                          # convert the PIL Image to a tensor
        ])
    
    
    train_dataset = torchvision.datasets.CIFAR10(root=os.environ['ALL_CCFRSCRATCH']+'/CIFAR_10', train=True, download=False, transform=transform)
    
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
    
        
    val_transform = transforms.Compose([
                    transforms.ToTensor()                           # convert the PIL Image to a tensor
                    ])
    
    val_dataset = torchvision.datasets.CIFAR10(root=os.environ['ALL_CCFRSCRATCH']+'/CIFAR_10', train=False, download=False, transform=val_transform)
    
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
    
    N_batch = len(train_loader)
    N_val_batch = len(val_loader)
    N_val = len(val_dataset)
    
        
    ## LR Finder #####                                             #***************************************************
    if args.findlr:                                                #
        if args.lr == 0.1: args.lr = 10                            #
        lrs, losses=[],[]                                          #
        mult = (args.lr / 1e-8) ** (1/((N_batch*args.epochs)-1))   #       ### DON'T MODIFY #########################
        optimizer_arg = optimizer
        optimizer_arg.param_groups[0]['lr'] = 1e-8                 #
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer_arg, step_size=1, gamma=mult) 
        
                                                                   #
    else:                                                          #***************************************************
        #### TO MODIFY ##### LR scheduler to accelerate the training time
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=N_batch, epochs=args.epochs)
    
    chrono.start()     ### DON'T MODIFY ####                                    
    
    ## Initialisation  
    if idr_torch.rank == 0: accuracies = []
    val_loss = torch.Tensor([0.]).to(gpu)                  # send to GPU
    val_accuracy = torch.Tensor([0.]).to(gpu)              # send to GPU
    
    # Pytorch profiler setup
    prof =  profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    schedule=schedule(wait=1, warmup=1, active=12),
                    on_trace_ready=tensorboard_trace_handler('./profiler/' + os.environ['SLURM_JOBID']),
                    profile_memory=True,
                    record_shapes=True, 
                    with_stack=False,
                    with_flops=True
                    ) if args.prof else contextlib.nullcontext()
    
    #### TRAINING ############
    with prof:
        for epoch in range(args.epochs):    

            train_sampler.set_epoch(epoch)
            chrono.dataload()                                    #**********************************************************
                                                                 #
            for i, (images, labels) in enumerate(train_loader):  #        ### DON'T MODIFY ##########################
                                                                 #
                csteps = i + 1 + epoch * N_batch                 #
                if args.test and csteps > 50: break              #**********************************************************

                # distribution of images and labels to all GPUs
                images = images.to(gpu, non_blocking=True)
                labels = labels.to(gpu, non_blocking=True)

                chrono.dataload()   #*******************************************************************************
                chrono.training()   #            ### DON'T MODIFY ##################
                chrono.forward()    #*******************************************************************************

                wrapped_optimizer.zero_grad()
                # Runs the forward pass with autocasting.
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                chrono.forward()    #*******************************************************************************
                chrono.backward()   #             ### DON'T MODIFY ################
                                    #*******************************************************************************            

                scaler.scale(loss).backward()
                scaler.step(wrapped_optimizer)
                scaler.update()

                # Metric mesurement
                _, predicted = torch.max(outputs.data, 1)      
                accuracy = (predicted == labels).sum() / labels.size(0)
                dist.all_reduce(accuracy, op=dist.ReduceOp.SUM)
                accuracy /= idr_torch.size
                if idr_torch.rank == 0: accuracies.append(accuracy.item())

                chrono.backward()                               #*****************************************************
                chrono.training()                               #
                                                                #
                if args.findlr:                                 #
                    lrs.append(scheduler.get_last_lr()[0])           #
                    losses.append(loss.item())                  #     ### DON'T MODIFY ############
                                                                #
                elif idr_torch.rank == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc:{:.4f}'.format(
                          epoch + 1, args.epochs, i+1, N_batch, loss.item(), np.mean(accuracies)))
                                                                #
                    accuracies = []                             #*****************************************************

                # scheduler update
                scheduler.step()
                
                # profiler update
                if args.prof: prof.step()

                chrono.dataload()                               #*****************************************************
                                                                #
                #### VALIDATION ############                    #
                if (((i+1)%(N_batch//2)==0) or (args.test and i==49)) and not args.findlr: 
                                                                #
                    model.eval()                                #       ### DON'T MODIFY #############
                    chrono.validation()                         #
                                                                #  
                    for iv, (val_images, val_labels) in enumerate(val_loader):   
                                                                #*****************************************************

                        # distribution of images and labels to all GPUs
                        val_images = val_images.to(gpu, non_blocking=True)
                        val_labels = val_labels.to(gpu, non_blocking=True)

                        # Runs the forward pass with no grade mode.
                        with torch.no_grad():
                            with autocast():
                                val_outputs = model(val_images)
                                loss = criterion(val_outputs, val_labels)

                        val_loss += (loss * val_images.size(0) / N_val)      #*****************************************
                        _, predicted = torch.max(val_outputs.data, 1)        #
                        val_accuracy += ((predicted == val_labels).sum() / N_val)  ### DON'T MODIFY #######
                                                                             #
                        if args.test and iv >= 20: break                     #*****************************************

                    dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
                    dist.all_reduce(val_accuracy, op=dist.ReduceOp.SUM)

                    chrono.validation()                                #***********************************************
                    model.train()                                      #
                                                                       #
                    if not args.test and idr_torch.rank == 0:          #    ### DON'T MODIFY ##############
                        print('##EVALUATION STEP##')                   #
                        print('Epoch [{}/{}], Validation Loss: {:.4f}, Validation Accuracy: {:.4f}'.format(epoch + 1, args.epochs,
                        val_loss.item(), val_accuracy.item()))
                        print('Learning Rate: {:.4f}'.format(scheduler.get_last_lr()[0]))
                        print(">>> Validation complete in: " + str(chrono.val_time))    
                                                                       #***********************************************  


                    ## Clear validations metrics
                    val_loss -= val_loss
                    val_accuracy -= val_accuracy 

    ## Be sure all process finish at the same time to avoid incoherent logs at the end of process 
    dist.barrier()
    
    if args.findlr:                                     #***********************************************************
        if idr_torch.rank == 0:                         #
            print(f'accuracies: {accuracies}')          #
            print(f'loss list: {losses}')               #
            print(f'learning rates: {lrs}')             #        ### DON'T MODIFY #############################
    else:                                               #
        chrono.display(N_val_batch)                     #
        if idr_torch.rank == 0:                         #
            print(">>> Number of batch per epoch: {}".format(N_batch)) 
            print(f'Max Memory Allocated {torch.cuda.max_memory_allocated()} Bytes') 
                                                        #***********************************************************
         

if __name__ == '__main__':
    
    # display info
    if idr_torch.rank == 0:
        print(">>> Training on ", len(idr_torch.hostnames), " nodes and ", idr_torch.size, " processes")
    main()
