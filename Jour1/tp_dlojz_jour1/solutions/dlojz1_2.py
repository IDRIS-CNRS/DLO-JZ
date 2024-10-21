## Author : Bertrand Cabot / IDRIS

import os                                                                                                     
import contextlib                                                                                                       
import argparse                                                                                               
import torchvision                                                                                            
import torchvision.transforms as transforms                                                                   
import torchvision.models as models                                                                                        
from torch.utils.checkpoint import checkpoint_sequential                                                              
import torch                                                                                                  
import numpy as np                                     
import apex
                                                                                                              
import idr_torch                                                                                              
from dlojz_chrono import Chronometer
from dlojz_torch import distributed_accuracy
                                                                                                              
import random                                                                                                 
random.seed(123)                                                                                              
np.random.seed(123)                                                                                           
torch.manual_seed(123)                                                                                        

## import ... ## Add here the libraries to import
# TODO
from torch.cuda.amp import autocast, GradScaler

VAL_BATCH_SIZE=256


def train():                                                                                                  
    parser = argparse.ArgumentParser()                                                                        
    parser.add_argument('-b', '--batch-size', default=128, type =int,                                         
                        help='batch size per GPU')                                                            
    parser.add_argument('-e','--epochs', default=1, type=int,                                                 
                        help='number of total epochs to run')                                                 
    parser.add_argument('--image-size', default=224, type=int,                                                
                        help='Image size')                                                                    
    parser.add_argument('--lr', default=0.1, type=float,                                                      
                        help='learning rate')                                                                 
    parser.add_argument('--wd', default=0., type=float,                                                       
                        help='weight decay')                                                                  
    parser.add_argument('--mom', default=0.9, type=float,                                                     
                        help='momentum')                                                                      
    parser.add_argument('--test', default=False, action='store_true',
                        help='Test 50 iterations')                                                            
    parser.add_argument('--test-nsteps', default='50', type=int,                                              
                        help='the number of steps in test mode')                                              
    parser.add_argument('--num-workers', default=8, type=int,                                                
                        help='num workers in dataloader')                                                     
    parser.add_argument('--persistent-workers', default=True, action=argparse.BooleanOptionalAction,          
                        help='activate persistent workers in dataloader')                                     
    parser.add_argument('--pin-memory', default=True, action=argparse.BooleanOptionalAction,                  
                        help='activate pin memory option in dataloader')                                      
    parser.add_argument('--non-blocking', default=True, action=argparse.BooleanOptionalAction,                
                        help='activate asynchronuous GPU transfer')                                           
    parser.add_argument('--prefetch-factor', default=2, type=int,                                             
                        help='prefectch factor in dataloader')                                                
    parser.add_argument('--drop-last', default=False, action=argparse.BooleanOptionalAction,                  
                        help='activate drop_last option in dataloader')                                       


    ## Add parser arguments
    args = parser.parse_args()

    
    ## chronometer initialisation
    chrono = Chronometer()
    
    # define model
    gpu = torch.device("cuda")
    model = models.resnet152()
    model = model.to(gpu)
    
    archi_model = 'Resnet-152'
    
    if idr_torch.rank == 0: print(f'model: {archi_model}')                                                   
    if idr_torch.rank == 0: print('number of parameters: {}'.format(sum([p.numel()
                                              for p in model.parameters()])))                                 


    # distribute batch size (mini-batch)                                                                      
    num_replica = idr_torch.size                                    
    mini_batch_size = args.batch_size                                                                         
    global_batch_size = mini_batch_size * num_replica                                                         
                                                                                                                        
    if idr_torch.rank == 0:                                                                                   
        print(f'global batch size: {global_batch_size} - mini batch size: {mini_batch_size}')              
                                                                                                              
    
    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1) 
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.mom, weight_decay=args.wd)
    
    if idr_torch.rank == 0: print(f'Optimizer: {optimizer}')
    
    # define metrics
    train_metric = distributed_accuracy()
    val_metric = distributed_accuracy()
    
    # Creates a GradScaler once at the beginning of training.
    # TODO
    scaler = GradScaler()

    #########  DATALOADER ############ 
    # Define a transform to pre-process the training images.

    if idr_torch.rank == 0:
        print(f"DATALOADER {args.num_workers} {args.persistent_workers} {args.pin_memory} \
{args.non_blocking} {args.prefetch_factor} {args.drop_last} ") ### DON'T MODIFY ###
    
    transform = transforms.Compose([ 
            transforms.RandomResizedCrop(args.image_size),  # Random resize - Data Augmentation
            transforms.RandomHorizontalFlip(),              # Horizontal Flip - Data Augmentation
            transforms.ToTensor(),                          # convert the PIL Image to a tensor
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
            ])
        
    
    
    train_dataset = torchvision.datasets.ImageNet(root=os.environ['ALL_CCFRSCRATCH']+'/imagenet',
                                                  transform=transform)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=mini_batch_size,
                                               shuffle=True,
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
    
    val_dataset = torchvision.datasets.ImageNet(root=os.environ['ALL_CCFRSCRATCH']+'/imagenet',
                                                split='val',
                                                transform=val_transform)
    
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,    
                                             batch_size=VAL_BATCH_SIZE,
                                             shuffle=False,
                                             num_workers=args.num_workers,
                                             persistent_workers=args.persistent_workers,
                                             pin_memory=args.pin_memory,
                                             prefetch_factor=args.prefetch_factor,
                                             drop_last=args.drop_last)
    
    N_batch = len(train_loader)
    N_val_batch = len(val_loader)
    N_val = len(val_dataset)
    
    #LR scheduler to accelerate the training time
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
                                                steps_per_epoch=N_batch, epochs=args.epochs)

    
    
    chrono.start()                               

    
    #### TRAINING ############
    for epoch in range(args.epochs):    
        
        if args.test: chrono.next_iter()
        if idr_torch.rank == 0: chrono.tac_time(clear=True)
        
        for i, (images, labels) in enumerate(train_loader):    

            csteps = i + 1 + epoch * N_batch
            if args.test and csteps > args.test_nsteps: break
            if i == 0 and idr_torch.rank == 0:
                print(f'image batch shape : {images.size()}')
                
            # distribution of images and labels to all GPUs
            images = images.to(gpu)
            labels = labels.to(gpu)
            
            if args.test: chrono.forward()
            
            optimizer.zero_grad()
            # Implement autocasting
            # TODO
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            if args.test: chrono.backward()       

            # Implement gradient scaling
            # TODO
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Metric mesurement
            train_metric.update(loss, outputs, labels)

            if args.test: chrono.update()
 
            if ((i + 1) % (N_batch//10) == 0 or i == N_batch - 1):
                train_loss, accuracy = train_metric.compute()
                if idr_torch.rank == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Time: {:.3f}, Loss: {:.4f}, Acc:{:.4f}'.format(
                          epoch + 1, args.epochs, i+1, N_batch,
                          chrono.tac_time(), train_loss, accuracy))

            # scheduler update
            scheduler.step()

            #### VALIDATION ############   
            if ((i == N_batch - 1) or (args.test and i==args.test_nsteps-1)) :
 
                chrono.validation()
                model.eval()

                for iv, (val_images, val_labels) in enumerate(val_loader):
                
                    # distribution of images and labels to all GPUs
                    val_images = val_images.to(gpu)
                    val_labels = val_labels.to(gpu)

                    # Runs the forward pass with no grad mode.
                    with torch.no_grad():
                        # Implement autocasting
                        # TODO
                        with autocast():
                            val_outputs = model(val_images)
                            val_loss = criterion(val_outputs, val_labels)

                    val_metric.update(val_loss, val_outputs, val_labels)
                                 
                    if args.test and iv >= 20: break
                                                   
                val_loss, val_accuracy = val_metric.compute() 


                model.train()  
                chrono.validation()   
                if not args.test and idr_torch.rank == 0:
                    print('##EVALUATION STEP##')
                    print('Epoch [{}/{}], Validation Loss: {:.4f}, Validation Accuracy: {:.4f}'.format(
                                         epoch + 1, args.epochs, val_loss, val_accuracy))
                    print(">>> Validation complete in: " + str(chrono.val_time))
     
            #### END OF VALIDATION ############
            
            if args.test: chrono.next_iter()
    
                                                             
    chrono.stop()
    if idr_torch.rank == 0:
        chrono.display()
        print(">>> Number of batch per epoch: {}".format(N_batch))
        print(f'Max Memory Allocated {torch.cuda.max_memory_allocated()} Bytes')

        
    # Save last checkpoint
    if not args.test and idr_torch.rank == 0:
        checkpoint_path = f"checkpoints/{os.environ['SLURM_JOBID']}_{global_batch_size}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print("Last epoch checkpointed to " + checkpoint_path)
        

if __name__ == '__main__':
    
    # display info
    if idr_torch.rank == 0:
        print(">>> Training on ", len(idr_torch.hostnames), " nodes and ", idr_torch.size, " processes")
    train()
