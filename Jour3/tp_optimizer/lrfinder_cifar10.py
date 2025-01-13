import os
import copy                                      
import contextlib                                
import argparse                                  
import torchvision                               
import torchvision.transforms as transforms      
import torchvision.models as models                             
import torch                                     
import numpy as np                              
import apex                                      
from apex.parallel.LARC import LARC              
                                                
import idr_torch                                 
                                                 
import random                                    
random.seed(123)                                 
np.random.seed(123)                              
torch.manual_seed(123)                           

from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from lion import Lion

VAL_BATCH_SIZE=512

import matplotlib.pyplot as plt

def save_loss_vs_learning_rate_plot(learning_rates, losses, filename):
    """
    Save a visualization depicting the relationship between learning rates and losses.
    
    Parameters:
    - learning_rates (list of floats): The learning rates for which losses were computed.
    - losses (list of floats): The computed losses corresponding to each learning rate.
    - filename (str): The name of the file (without extension) to which the plot will be saved.
    
    The resulting plot will be saved as a PNG in the "images" directory.
    """
    
    plt.figure(figsize=(10,5))
    
    plt.plot(learning_rates, losses, color='blue', label="Average Loss")
    plt.xlabel('Learning Rate', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.legend(loc="upper left")
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim([min(losses),  max(losses)])
    
    # Check if "images" directory exists, if not, create it
    if not os.path.exists("images_lrfinder"):
        os.makedirs("images_lrfinder")

    plt.savefig(os.path.join("images_lrfinder", filename+'.png'))

def explore_lrs_decorelated(dataloader, 
                            model, 
                            optimizer,
                            criterion,
                            device,
                            min_learning_rate_power=-8, 
                            max_learning_rate_power = 1,
                            num_lrs=10,
                            steps_per_lr=50):
  
    lrs = np.logspace(min_learning_rate_power, max_learning_rate_power, num=num_lrs)
    print("Learning rate space : ", lrs)
    model_init_state = model.state_dict()

  
    lrs_losses, lrs_metric_avg, lrs_metric_var =[], [],[]
  
    # Iterate through learning rates to test
    for lr in lrs:
        print("Testing lr:", lr)
        # Reset model
        model.load_state_dict(model_init_state)

        # Change learning rate in optimizer
        for group in optimizer.param_groups:
            group['lr'] = lr

        # Reset metric tracking
        lr_losses =[]

        # Training steps
        for step in range(steps_per_lr):
            images, labels = next(iter(dataloader))
            # distribution of images and labels to all GPUs
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            lr_losses.append(loss.item())

        # Compute loss average for lr
        lr_loss_avg = np.mean(lr_losses)
        lr_loss_avg = lr_losses[-1]

        lrs_losses.append(lr_loss_avg)

        # Compute metric (discounted average gradient of the loss)
        lr_gradients = np.gradient(lr_losses)
        #lr_metric_avg = np.average(lr_gradients, weights=np.linspace(0.1,1,len(lr_gradients)))
        lr_metric_avg = np.mean(lr_gradients)
        lr_metric_var = np.var(lr_gradients)
        lrs_metric_avg.append(lr_metric_avg)    
        lrs_metric_var.append(lr_metric_var)

    return lrs, lrs_losses, lrs_metric_avg, lrs_metric_var

def main():                                                            #******************************************
    parser = argparse.ArgumentParser()                                 #
    parser.add_argument('-b', '--batch-size', default=128, type =int,  #
                        help='batch size per GPU')                     #
    parser.add_argument('-e','--epochs', default=1, type=int,          #
                        help='number of total epochs to run')          #
    parser.add_argument('--image-size', default=224, type=int,         #
                        help='Image size')                             #
    parser.add_argument('--lr_type', default="corelated", type=str,             #      ##    DON'T MODIFY    ########
                        help='lrfinder type')                          #
    parser.add_argument('--mom', default=0.9, type=float,              #
                        help='momentum')                               #
    parser.add_argument('--wd', default=0., type=float,                #
                        help='weight decay')                           #
    args = parser.parse_args()

    train(args)

def train(args):
    # dist config
    dist.init_process_group(backend='nccl', 
                            init_method='env://', 
                            world_size=idr_torch.size,
                            rank=idr_torch.rank)
    
    # define model
    torch.cuda.set_device(idr_torch.local_rank)
    gpu = torch.device("cuda")
    
    model = models.resnet18(num_classes=10)
    model = model.to(gpu)
    model = DistributedDataParallel(model, device_ids=[idr_torch.local_rank])

    
    model.name = 'Resnet-18'
    if idr_torch.rank == 0: print(f'model: {model.name}')  ### DON'T MODIFY ###
    
    

    num_replica = idr_torch.size                                     
    mini_batch_size = args.batch_size                                
    global_batch_size = mini_batch_size * num_replica                
                                                                      
    if idr_torch.rank == 0:                                          
        print(f'global batch size: {global_batch_size} - mini batch size: {mini_batch_size}')
    
    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1) 
    #optimizer = torch.optim.SGD(model.parameters(), 0.01 , momentum=args.mom, weight_decay=args.wd)
    #optimizer = torch.optim.AdamW(model.parameters(), 0.01, betas=(args.mom, 0.999), weight_decay=args.wd)
    #optimizer = apex.optimizers.FusedLAMB(model.parameters(), 0.01, betas=(args.mom, 0.999), weight_decay=args.wd)
    base_optimizer = torch.optim.SGD(model.parameters(), 0.01, momentum=args.mom, weight_decay=args.wd)
    optimizer = LARC(base_optimizer)
    #optimizer = Lion(model.parameters(), lr=0.01, weight_decay=args.wd)
    
    
    
    if idr_torch.rank == 0: 
        print(f'Criterion: {criterion}')
        print(f'Optimizer: {optimizer}')
        
    # Creates a GradScaler once at the beginning of training.
    scaler = GradScaler()
        
    # Dataloader
    transform = transforms.Compose([ 
        transforms.RandomHorizontalFlip(),     # Horizontal Flip - Data Augmentation
        transforms.RandomCrop(32, padding=4),  # Data Augmentation    
        transforms.ToTensor()])                # convert the PIL Image to a tensor

        
    train_dataset = torchvision.datasets.CIFAR10(root=os.environ['ALL_CCFRSCRATCH']+'/CIFAR_10', train=True, download=False, transform=transform)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=idr_torch.size,
                                                                    rank=idr_torch.rank,
                                                                    shuffle=True)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=mini_batch_size,
                                               shuffle=False,
                                               sampler=train_sampler,
                                               num_workers=8,
                                               persistent_workers=True,
                                               pin_memory=True,
                                               prefetch_factor=3)

    
    N_batch = len(train_loader)
             
        
        
        
    # Find LR
    if (args.lr_type == "decorelated"):
        #test sur 256 et 8192
        steps_per_lr = int((4096/np.sqrt(global_batch_size)))
        lrs, lrs_losses, lrs_metric_avg, lrs_metric_var = explore_lrs_decorelated(train_loader, 
                                                                            model, 
                                                                            optimizer,
                                                                            criterion,
                                                                            gpu,
                                                                            min_learning_rate_power=-8, 
                                                                            max_learning_rate_power = 1,
                                                                            num_lrs=10,
                                                                            steps_per_lr=steps_per_lr)
        if idr_torch.rank == 0:
            image_name = 'lr_decorelated_'+ type(optimizer).__name__ + '_' + str(global_batch_size)
            save_loss_vs_learning_rate_plot(lrs, lrs_losses, image_name)
                
    else: #classic LR or not provided
        #explore_lr()
        print("classic_lr not implemented")
        
if __name__ == '__main__':
    if idr_torch.rank == 0:
        print(">>> Training on ", len(idr_torch.nodelist), " nodes and ", idr_torch.size, " processes")
    main()
    
'''
# To run the code:
command = [f'lrfinder_cifar10.py -b 128 -e 10 --lr_type decorelated', 
           f'lrfinder_cifar10.py -b 4096 -e 10 --lr_type decorelated'
          ]
jobid_sgd_lrf = gpu_jobs_submitter(command, n_gpu, MODULE, name=name,
                   account=account, time_max='00:30:00')
print(f'jobid_sgd_lrf = {jobid_sgd_lrf}')
'''