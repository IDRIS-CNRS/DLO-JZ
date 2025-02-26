## Author : Bertrand Cabot / IDRIS
#**************************************************************************************************************
import os                                                                                                     #
import contextlib                                                                                             #          
import argparse                                                                                               #
import torchvision                                                                                            #
import torchvision.transforms as transforms                                                                   #
import torchvision.models as models                                                                           #             
from torch.utils.checkpoint import checkpoint_sequential                                                      #         
import torch                                                                                                  #
import numpy as np                                     ####       DON'T MODIFY    ####################        #
import apex                                                                                                   #
import wandb                                                                                                  #
                                                                                                              #
import idr_torch                                                                                              #
from dlojz_chrono import Chronometer                                                                          #
                                                                                                              #
import random                                                                                                 #
random.seed(123)                                                                                              #
np.random.seed(123)                                                                                           #
torch.manual_seed(123)                                                                                        #
#**************************************************************************************************************

## import ... ## Add here the libraries to import
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.profiler import profile, tensorboard_trace_handler, ProfilerActivity, schedule
from CoAtNet.coatnet import coatnet_7
from torch.distributed.pipeline.sync import Pipe
import tempfile
from torch.distributed import rpc

VAL_BATCH_SIZE=256

#**************************************************************************************************************
def train():                                                                                                  #
    parser = argparse.ArgumentParser()                                                                        #
    parser.add_argument('-b', '--batch-size', default=128, type =int,                                         #
                        help='batch size per GPU')                                                            #
    parser.add_argument('-e','--epochs', default=1, type=int,                                                 #
                        help='number of total epochs to run')                                                 #
    parser.add_argument('--image-size', default=224, type=int,                                                #
                        help='Image size')                                                                    #
    parser.add_argument('--test', default=False, action='store_true',     ##    DON'T MODIFY    ########      #
                        help='Test 50 iterations')                                                            #
    parser.add_argument('--test-nsteps', default='50', type=int,                                              #
                        help='the number of steps in test mode')                                              #
    parser.add_argument('--num-workers', default=10, type=int,                                                #
                        help='num workers in dataloader')                                                     #
    parser.add_argument('--persistent-workers', default=True, action=argparse.BooleanOptionalAction,          # 
                        help='activate persistent workers in dataloader')                                     #
    parser.add_argument('--pin-memory', default=True, action=argparse.BooleanOptionalAction,                  #
                        help='activate pin memory option in dataloader')                                      #
    parser.add_argument('--non-blocking', default=True, action=argparse.BooleanOptionalAction,                #
                        help='activate asynchronuous GPU transfer')                                           #
    parser.add_argument('--prefetch-factor', default=3, type=int,                                             #
                        help='prefectch factor in dataloader')                                                #
    parser.add_argument('--drop-last', default=False, action=argparse.BooleanOptionalAction,                  #
                        help='activate drop_last option in dataloader')                                       #
#**************************************************************************************************************

    ## Add parser arguments
    parser.add_argument('--prof', default=False, action='store_true', help='PROF implementation')
    parser.add_argument('--chunks', default=1, type=int, help='number of chunks for Pipelined Parallelism')
    
    args = parser.parse_args()

    
    ## chronometer initialisation (test and rank)
    chrono = Chronometer(args.test, idr_torch.rank)       ### DON'T MODIFY ### 
    
    # configure distribution method: define rank and initialise communication backend (NCCL)
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=idr_torch.size, rank=idr_torch.rank)
    
    # Initialize RPC Framework, Pipe depends on it
    tmpfile = tempfile.NamedTemporaryFile()
    rpc.init_rpc(
        name=f'worker{idr_torch.rank}',
        rank=0,
        world_size=1,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
            init_method="file://{}".format(tmpfile.name),
            # Specifying _transports and _channels is a workaround and we no longer
            # will have to specify _transports and _channels for PyTorch 
            # versions >= 1.8.1 (Not True for Jean Zay)
            # With Jean Zay, _transports must be equal to ["shm", "uv"] and not ["ibv", "uv"]
            _transports=["shm", "uv"],
            _channels=["cuda_ipc", "cuda_basic"],
        )
    )

    
    # define model
    model = coatnet_7((args.image_size,args.image_size))

    # How many sections
    nb_part = torch.cuda.device_count()//int(os.environ['SLURM_NTASKS_PER_NODE']) 
    # device number where the first part of the model will run
    first_part = idr_torch.local_rank*nb_part
    # list of devices involved for pipelined Parallelism
    gpus = [g for g in range(first_part, first_part+nb_part)]

    class LambdaModule(torch.nn.Module):
        def __init__(self, lambd):
            super().__init__()
            assert isinstance(lambd, type(lambda x: x))
            self.lambd = lambd

        def forward(self, x):
            return self.lambd(x)

    lambda_fc = LambdaModule(lambda x: x.view(-1, 3072))

    section0 = torch.nn.Sequential(*model.s0, *model.s1, *model.s2, *model.pres3).to(gpus[0])
    section1 = torch.nn.Sequential(*model.s3[:15]).to(gpus[1])
    section2 = torch.nn.Sequential(*model.s3[15:30]).to(gpus[2])
    section3 = torch.nn.Sequential(*model.s3[30:], *model.s4, model.pool, lambda_fc, model.fc).to(gpus[3])
    pipe_model = torch.nn.Sequential(*section0, *section1, *section2, *section3)

    # Pipe the model, chunks=n means that the batch (size according to batch size) will be shared to n micro batches (size = batch_size/chunks)
    model = Pipe(pipe_model, chunks=args.chunks, checkpoint="never")

    archi_model = 'CoAtNet-7'
    
#**************************************************************************************************************
    if idr_torch.rank == 0: print(f'model: {archi_model}')                                                    #
    if idr_torch.rank == 0: print('number of parameters: {}'.format(sum([p.numel() ### DON'T MODIFY ####      #
                                              for p in model.parameters()])))                                 #
#*************************************************************************************************************#

    model = DistributedDataParallel(model)

#*************************************************************************************************************#
    # distribute batch size (mini-batch)                                                                      #
    num_replica = idr_torch.size                                    ### DON'T MODIFY ##################       #
    mini_batch_size = args.batch_size                                                                         #
    global_batch_size = mini_batch_size * num_replica                                                         #
                                                                                                              #          
    if idr_torch.rank == 0:                                                                                   #
        print(f'global batch size: {global_batch_size} - mini batch size: {mini_batch_size}')                 #
                                                                                                              #
#**************************************************************************************************************
    
    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1) 
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    if idr_torch.rank == 0: print(f'Optimizer: {optimizer}')    ### DON'T MODIFY ###
    
    #########  DATALOADER ############ 
    # Define a transform to pre-process the training images.

    if idr_torch.rank == 0: print(f"DATALOADER {args.num_workers} {args.persistent_workers} {args.pin_memory} {args.non_blocking} {args.prefetch_factor} {args.drop_last} ") ### DON'T MODIFY ###
    
    transform = transforms.Compose([ 
            transforms.RandomResizedCrop(args.image_size),  # Random resize - Data Augmentation
            transforms.RandomHorizontalFlip(),              # Horizontal Flip - Data Augmentation
            transforms.ToTensor(),                          # convert the PIL Image to a tensor
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
            ])
        
    
    
    train_dataset = torchvision.datasets.ImageNet(root=os.environ['ALL_CCFRSCRATCH']+'/imagenet',
                                                  transform=transform)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                num_replicas=idr_torch.size,
                                                                rank=idr_torch.rank,
                                                                shuffle=True)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=mini_batch_size,
                                               shuffle=False,
                                               sampler=train_sampler,
                                               num_workers=args.num_workers,
                                               persistent_workers=args.persistent_workers,
                                               pin_memory=args.pin_memory,
                                               prefetch_factor=args.prefetch_factor,
                                               drop_last=args.drop_last)
    
        
    val_transform = transforms.Compose([
              transforms.Resize((256, 256)),
              transforms.CenterCrop(224),
              transforms.ToTensor(),   # convert the PIL Image to a tensor
              transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                   std=(0.229, 0.224, 0.225))])
    
    val_dataset = torchvision.datasets.ImageNet(root=os.environ['ALL_CCFRSCRATCH']+'/imagenet', split='val',
                        transform=val_transform)
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset,
                                                              num_replicas=idr_torch.size,
                                                              rank=idr_torch.rank,
                                                              shuffle=False)
    
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,    
                                             batch_size=VAL_BATCH_SIZE,
                                             shuffle=False,
                                             sampler=val_sampler,
                                             num_workers=args.num_workers,
                                             persistent_workers=args.persistent_workers,
                                             pin_memory=args.pin_memory,
                                             prefetch_factor=args.prefetch_factor,
                                             drop_last=args.drop_last)
    
    N_batch = len(train_loader)
    N_val_batch = len(val_loader)
    N_val = len(val_dataset)

    #LR scheduler to accelerate the training time
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4,
                                                steps_per_epoch=N_batch, epochs=args.epochs)

    scaler = GradScaler()
    
    chrono.start()     ### DON'T MODIFY ####
    
    ## Initialisation  
    if idr_torch.rank == 0: accuracies = []
    val_loss = torch.Tensor([0.]).to(gpus[-1])                  # send to GPU
    val_accuracy = torch.Tensor([0.]).to(gpus[-1])              # send to GPU
    
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
    
#**************************************************************************************************************    
    ### Weight and biases initialization                                                                      #
    if not args.test and idr_torch.rank == 0:                                                                 #
        config = dict(                                                                                        # 
          architecture = archi_model,                                                                         #
          batch_size = args.batch_size,                                                                       #
          epochs = args.epochs,                                                                               #
          image_size = args.image_size,                                                                       #
          learning_rate = args.lr,                                                                            #
          weight_decay = args.wd,                                                                             #
          momentum = args.mom,                                                                                #
          optimizer = optimizer.__class__.__name__,                                                           #
          lr_scheduler = scheduler.__class__.__name__                      #### DON'T MODIFY ######           #
        )                                                                                                     #
                                                                                                              #
        wandb.init(                                                                                           #
          project="Imagenet Race Cup",                                                                        #
          entity="dlojz",                                                                                     #
          name=os.environ['SLURM_JOB_NAME']+'_'+os.environ['SLURM_JOBID'],                                    #
          tags=['label smoothing'],                                                                           #
          config=config,                                                                                      #
          mode='offline'                                                                                      #
          )                                                                                                   #
        wandb.watch(model, log="all", log_freq=N_batch)                                                       #
#**************************************************************************************************************
    
    #### TRAINING ############
    with prof:
        for epoch in range(args.epochs):
            train_sampler.set_epoch(epoch)

    #**************************************************************************************************************
            chrono.dataload()                                                                                     #
            if idr_torch.rank == 0: chrono.tac_time(clear=True)                                                   #
                                                                                                                  #
            for i, (images, labels) in enumerate(train_loader):                                                   #        
                                                                         ### DON'T MODIFY ##############          #
                csteps = i + 1 + epoch * N_batch
                if args.test: print(f'Train step {csteps} - rank {idr_torch.rank}')#
                if args.test and csteps > args.test_nsteps: break                                                 #
                if i == 0 and idr_torch.rank == 0:                                                                #
                    print(f'image batch shape : {images.size()}')                                                 #
                                                                                                                  #
    #**************************************************************************************************************

                # distribution of images and labels to all GPUs
                images = images.to(gpus[0], non_blocking=args.non_blocking)
                labels = labels.to(gpus[-1], non_blocking=args.non_blocking)


    #**************************************************************************************************************
                chrono.dataload()                                                                                 #
                chrono.training()               ### DON'T MODIFY #################                                # 
                chrono.forward()                                                                                  #
    #**************************************************************************************************************

                optimizer.zero_grad()
                # Runs the forward pass.
                with autocast():
                    outputs = model(images).local_value()
                loss = criterion(outputs, labels)

    #**************************************************************************************************************
                chrono.forward()                                                                                  #
                chrono.backward()                 ### DON'T MODIFY ###############                                #
    #**************************************************************************************************************         

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # Metric mesurement
                _, predicted = torch.max(outputs.data, 1)
                accuracy = (predicted == labels).sum() / labels.size(0)
                dist.all_reduce(accuracy, op=dist.ReduceOp.SUM)
                accuracy /= idr_torch.size
                if idr_torch.rank == 0: accuracies.append(accuracy.item())

    #***************************************************************************************************************
                if not args.test and idr_torch.rank == 0 and csteps%10 == 0:                                       #                                         
                    wandb.log({"train accuracy": accuracy.item(),                                                  #
                               "train loss":  loss.item(),                                                         #
                               "learning rate": scheduler.get_lr()[0]}, step=csteps)                               #
                                                                                                                   #
                chrono.backward()                                                                                  #
                chrono.training()                                        ### DON'T MODIFY ###########              #
                                                                                                                   #   
                                                                                                                   #
                if ((i + 1) % (N_batch//10) == 0 or i == N_batch - 1) and idr_torch.rank == 0:                     #
                    print('Epoch [{}/{}], Step [{}/{}], Time: {:.3f}, Loss: {:.4f}, Acc:{:.4f}'.format(            #
                          epoch + 1, args.epochs, i+1, N_batch,                                                    #
                          chrono.tac_time(), loss.item(), np.mean(accuracies)))                                    #
                                                                                                                   #
                    accuracies = []                                                                                #
    #***************************************************************************************************************

                # scheduler update
                scheduler.step()
            
                # profiler update
                if args.prof: prof.step()

    #***************************************************************************************************************            
                chrono.dataload()                                                                                  #
                                                                                                                   #
                #### VALIDATION ############                                                                       #
                if ((i == N_batch - 1) or (args.test and i==args.test_nsteps-1)) :                                 # 
                                                                                                                   #
                    chrono.validation()                                                                            #
                    model.eval()                                        ### DON'T MODIFY ############              #  
                    if args.test: print(f'Train step 100 - rank {idr_torch.rank}')                                                 #  
                    for iv, (val_images, val_labels) in enumerate(val_loader):                                     #  
    #***************************************************************************************************************

                        # distribution of images and labels to all GPUs
                        val_images = val_images.to(gpus[0], non_blocking=args.non_blocking)
                        val_labels = val_labels.to(gpus[-1], non_blocking=args.non_blocking)


                        # Runs the forward pass with no grade mode.
                        with torch.no_grad(), autocast():
                            val_outputs = model(val_images).local_value()
                            loss = criterion(val_outputs, val_labels)

    #***************************************************************************************************************
                        val_loss += (loss * val_images.size(0) / N_val)                                            #
                        _, predicted = torch.max(val_outputs.data, 1)                                              #
                        val_accuracy += ((predicted == val_labels).sum() / N_val)  ### DON'T MODIFY #######        #
                                                                                                                   #
                        if args.test and iv >= 20: break                                                           #
    #***************************************************************************************************************                                                    

                    dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
                    dist.all_reduce(val_accuracy, op=dist.ReduceOp.SUM) 

    #***************************************************************************************************************
                    model.train()                                                                                  #
                    chrono.validation()                                                                            #
                    if idr_torch.rank == 0: assert val_accuracy.item() <= 1., 'Something wrong with your allreduce'#
                    if not args.test and idr_torch.rank == 0:                    ### DON'T MODIFY #############    #    
                        print('##EVALUATION STEP##')                                                               #
                        print('Epoch [{}/{}], Validation Loss: {:.4f}, Validation Accuracy: {:.4f}'.format(        #
                                             epoch + 1, args.epochs, val_loss.item(), val_accuracy.item()))        #
                        print(">>> Validation complete in: " + str(chrono.val_time))                               #
                        if val_accuracy.item() > 1.:                                                               #
                            print('ddp implementation error : accuracy outlier !!')                                #
                            wandb.log({"test accuracy": None,                                                      #
                                   "test loss":  val_loss.item()})                                                 #
                        else:                                                                                      #
                            wandb.log({"test accuracy": val_accuracy.item(),                                       #
                                   "test loss":  val_loss.item()})                                                 #
    #***************************************************************************************************************

                    ## Clear validations metrics
                    val_loss -= val_loss
                    val_accuracy -= val_accuracy

    ## Be sure all process finish at the same time to avoid incoherent logs at the end of process
    dist.barrier()
    
#***************************************************************************************************************                                                                                                                                       #
    chrono.display(N_val_batch)                                                                                #
    if idr_torch.rank == 0:                                      ### DON'T MODIFY ###############              #
        print(">>> Number of batch per epoch: {}".format(N_batch))                                             # 
        print(f'Max Memory Allocated {torch.cuda.max_memory_allocated()} Bytes')                               #
    else:                                                                                                      #
        print(f'MaxMemory for GPU:{idr_torch.rank} {torch.cuda.max_memory_allocated()} Bytes')                 #
#***************************************************************************************************************
    for g in gpus: print(f'MaxMemory for GPU:{g} {torch.cuda.max_memory_allocated(device=g)} Bytes') 
        
    # Save last checkpoint
    if not args.test and idr_torch.rank == 0:
        checkpoint_path = f"checkpoints/{os.environ['SLURM_JOBID']}_{global_batch_size}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print("Last epoch checkpointed to " + checkpoint_path)
        

if __name__ == '__main__':
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,COLL"
    # display info
    if idr_torch.rank == 0:
        print(">>> Training on ", len(idr_torch.nodelist), " nodes and ", idr_torch.size, " processes")
    train()
