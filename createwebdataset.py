import os                                        #*************************************************************
import torchvision                               #       
import torch                                     #
import numpy as np                               
import webdataset as wds


if __name__ == '__main__':
    
    train_dataset = torchvision.datasets.ImageNet(root=os.environ['ALL_CCFRSCRATCH']+'/imagenet')
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=None,
                                               shuffle=True,
                                               num_workers=10)
    
    sink = wds.ShardWriter(os.environ['ALL_CCFRSCRATCH']+'/imagenet/webdataset/imagenet_train-%06d.tar',
                           maxcount=10010, maxsize=3e12)

    for index, (input, output) in enumerate(train_loader):
        sink.write({
            "__key__": "sample%06d" % index,
            "input.pyd": input,
            "output.pyd": output,
        })
    sink.close()