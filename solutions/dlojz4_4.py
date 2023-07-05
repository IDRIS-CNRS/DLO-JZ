import os                                        #*************************************************************
import contextlib                                #
import argparse                                  #
import torchvision                               #
import torchvision.transforms as transforms      #
import torchvision.models as models              #             
from torch.utils.checkpoint import checkpoint_sequential         
import torch                                     #
import numpy as np                               #   ###       DON'T MODIFY    ####################
import apex                                      #
import wandb                                     #
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
from CoAtNet.coatnet import coatnet_6
import deepspeed
from deepspeed.pipe import PipelineModule
import json


def main():                                                            
    parser = argparse.ArgumentParser()                                 
    parser.add_argument('-e','--epochs', default=1, type=int,         
                        help='number of total epochs to run')         
    parser.add_argument('--image-size', default=224, type=int,        
                        help='Image size')                             
    parser.add_argument('--test', default=False, action='store_true',  
                        help='Test 50 iterations')                                               
    parser.add_argument('--test-nsteps', default='50', type=int,                  #
                        help='the number of steps in test mode')                  #
    parser.add_argument('--prof', default=False, action='store_true', help='PROF implementation')
    parser.add_argument('-p' , '--nb-pipeline-stages', default=1, type=int, help='Number of pipeline for stages')
    parser.add_argument('--partition-param', default=False, action='store_true', help='Partition Method = parameters else uniform')
    parser.add_argument('--num-workers', default=10, type=int, help='num workers in dataloader')
    parser.add_argument('--persistent-workers', default=True, action=argparse.BooleanOptionalAction, help='activate persistent workers in dataloader')
    parser.add_argument('--pin-memory', default=True, action=argparse.BooleanOptionalAction, help='activate pin memory option in dataloader')
    parser.add_argument('--non-blocking', default=True, action=argparse.BooleanOptionalAction, help='activate asynchronuous GPU transfer')
    parser.add_argument('--prefetch-factor', default=3, type=int, help='prefectch factor in dataloader')
    parser.add_argument('--drop-last', default=False, action=argparse.BooleanOptionalAction, help='activate drop_last option in dataloader')
      
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
      
    args = parser.parse_args()

    train(args)
    

def train(args):
    
    ## chronometer initialisation (test and rank)
    chrono = Chronometer(args.test, idr_torch.rank)       ### DON'T MODIFY ### 
    
    # define model
    torch.cuda.set_device(idr_torch.local_rank)
    gpu = torch.device("cuda")
    model = coatnet_6((args.image_size,args.image_size))
    
    archi_model = 'CoAtNet-6'
    if idr_torch.rank == 0: print(f'model: {archi_model}')  ### DON'T MODIFY ###
    if idr_torch.rank == 0: print('number of parameters: {}'.format(sum([p.numel() ### DON'T MODIFY ###
                                                  for p in model.parameters()]))) ### DON'T MODIFY ###
    
    # define loss function (criterion)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1) 
    
    # Define Pipeline Module
    deepspeed.init_distributed(distributed_port=os.environ['MASTER_PORT'])
    model = PipelineModule(layers = [
                        *model.s0, *model.s1, *model.s2, *model.pres3, *model.s3, *model.s4,
                         model.pool, lambda x: x.view(-1, 2048), model.fc],
                         num_stages = args.nb_pipeline_stages,
                         loss_fn=criterion,
                         partition_method = 'parameters' if args.partition_param else 'uniform')
    
    
    #########  DATALOADER ############ 
    # Define a transform to pre-process the training images.

    if idr_torch.rank == 0: print(f"DATALOADER {args.num_workers} {args.persistent_workers} {args.pin_memory} {args.non_blocking} {args.prefetch_factor} {args.drop_last} ") ### DON'T MODIFY ###
    
    transform = transforms.Compose([ 
        transforms.RandomResizedCrop(args.image_size),  # Random resize - Data Augmentation
        transforms.RandomHorizontalFlip(),              # Horizontal Flip - Data Augmentation 
        transforms.ToTensor(),                           # convert the PIL Image to a tensor
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
        transforms.Lambda(lambda x : x.half())
        ])
    
    
    train_dataset = torchvision.datasets.ImageNet(root=os.environ['ALL_CCFRSCRATCH']+'/imagenet',
                                                  transform=transform)
    
    # Deepspeed initialization - force port number if several job run on the same node 
    model_engine, optimizer, _, scheduler = deepspeed.initialize(args=args,
                                                         model=model, 
                                                         model_parameters=model.parameters(),
                                                         training_data=train_dataset)
                                                         
    dsconfig = model_engine._config
    # distribute batch size (mini-batch)                             #*************************************************
    num_replica = dsconfig.world_size                                #
    micro_batch_size = dsconfig.train_micro_batch_size_per_gpu
    n_chunks = dsconfig.gradient_accumulation_steps
    mini_batch_size = micro_batch_size *  n_chunks  
    global_batch_size = dsconfig.train_batch_size                
                                                                     #          ### DON'T MODIFY ################## 
    if idr_torch.rank == 0:                                          #
        print(f'global batch size: {global_batch_size} - mini batch size: {mini_batch_size}')
                                                                     #*************************************************
    dp_global_rank = model_engine.mpu.get_data_parallel_rank()        

    

    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                num_replicas=num_replica,
                                                                rank=dp_global_rank,
                                                                shuffle=True)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=micro_batch_size,
                                               shuffle=False,
                                               sampler=train_sampler,
                                               num_workers=args.num_workers,
                                               persistent_workers=args.persistent_workers,
                                               pin_memory=args.pin_memory,
                                               prefetch_factor=args.prefetch_factor,
                                               drop_last=args.drop_last)
    
        
    val_transform = transforms.Compose([
                                        transforms.Resize((int(8/7*args.image_size), args.image_size)),
                                        transforms.CenterCrop(args.image_size),
                                        transforms.ToTensor(),   # convert the PIL Image to a tensor
                                        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                             std=(0.229, 0.224, 0.225)),
                                        transforms.Lambda(lambda x : x.half())
                                        ])
    
    val_dataset = torchvision.datasets.ImageNet(root=os.environ['ALL_CCFRSCRATCH']+'/imagenet', split='val',
                        transform=val_transform)
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset,
                                                              num_replicas=num_replica,
                                                              rank=dp_global_rank,
                                                              shuffle=False)
    
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,    
                                             batch_size=micro_batch_size,
                                             shuffle=False,
                                             sampler=val_sampler,
                                             num_workers=args.num_workers,
                                             persistent_workers=args.persistent_workers,
                                             pin_memory=args.pin_memory,
                                             prefetch_factor=args.prefetch_factor,
                                             drop_last=args.drop_last)
    
    N_batch = len(train_loader) // n_chunks
    if len(train_loader) % n_chunks != 0: N_batch += 1
    N_val_batch = len(val_loader) // n_chunks
    if len(val_loader) % n_chunks != 0: N_val_batch += 1
    N_val = len(val_dataset)

    
    chrono.start()     ### DON'T MODIFY ####                                    
    
    
    # Pytorch profiler setup
    prof =  profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    schedule=schedule(wait=1, warmup=1, active=12, repeat=1),
                    on_trace_ready=tensorboard_trace_handler('./profiler/' + os.environ['SLURM_JOB_NAME'] 
                                                             + '_' + os.environ['SLURM_JOBID'] + '_bs' +
                                                             str(mini_batch_size)  + '_is' + str(args.image_size)),
                    profile_memory=True,
                    record_shapes=False, 
                    with_stack=False,
                    with_flops=False
                    ) if args.prof else contextlib.nullcontext()
    
    
    #### TRAINING ############
    with prof:
        for epoch in range(args.epochs):    
            train_sampler.set_epoch(epoch)

            chrono.dataload()                                    
                                                                 
            for i in range(30 if args.test else N_batch):                   
                
                chrono.dataload()   #
                chrono.training()   #            ### DON'T MODIFY ##################
                
                
                loss = model_engine.train_batch()
              
                
                chrono.training()                               
                                                                
               
                # profiler update
                if args.prof: prof.step()


                chrono.dataload()                          
                                                                
                #### VALIDATION ############                    
                if ((i == N_batch - 1) or (args.test and i==29)): 
                                                                
                    val_loss = []                              
                    chrono.validation()                         
                                                                 
                    val_iter = iter(val_loader)
                    for iv in range(20 if args.test else N_val_batch):
             
                        vloss = model_engine.eval_batch(val_iter)
                        if idr_torch.rank == 0: val_loss.append(vloss.item())
                        


                    chrono.validation()                                
                    model_engine.train()                               
                                                                       
                    if idr_torch.rank == 0:             
                        print('##EVALUATION STEP##')                   
                        print('Epoch [{}/{}], Validation Loss: {:.4f}'.format(epoch + 1,
                                                                 args.epochs, np.mean(val_loss)))         #
                        print(">>> Validation complete in: " + str(chrono.val_time))                                                
    
    ## Be sure all process finish at the same time to avoid incoherent logs at the end of process
    dist.barrier()
    
                                              
    chrono.display(N_val_batch)                     
    if idr_torch.rank == 0:                         
        print(">>> Number of batch per epoch: {}".format(N_batch)) 
        print(f'Max Memory Allocated GPU:{idr_torch.rank} {torch.cuda.max_memory_allocated()} Bytes')
    else:
        print(f'MaxMemory for GPU:{idr_torch.rank} {torch.cuda.max_memory_allocated()} Bytes')  
                                                    
if __name__ == '__main__':   
    # display info
    if idr_torch.rank == 0:
        print(">>> Training on ", len(idr_torch.hostnames), " nodes and ", idr_torch.size, " processes")
    main()
