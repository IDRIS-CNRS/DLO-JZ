from datetime import datetime
from time import time
import numpy as np


class Chronometer:
    def __init__(self, test, rank):
        self.test = test
        self.rank = rank
        self.time_perf_train = []
        self.time_perf_load = []
        self.time_perf_forward = []
        self.time_perf_backward = []
        self.start_proc = None
        self.start_training = None
        self.start_dataload = None
        self.start_backward = None
        self.start_forward = None
        self.start_valid = None
        self.val_time = None
        
    def clear(self):
        self.time_perf_train = []
        self.time_perf_load = []
        self.time_perf_forward = []
        self.time_perf_backward = []
        
    def start(self):
        if self.rank == 0: self.start_proc = datetime.now()
            
    def dataload(self):
        if self.rank == 0 and self.test:
            if self.start_dataload==None: self.start_dataload = time()
            else:
                self.time_perf_load.append(time() - self.start_dataload)
                self.start_dataload = None
                
    def training(self):
        if self.rank == 0 and self.test:
            if self.start_training==None: self.start_training = time()
            else:
                self.time_perf_train.append(time() - self.start_training)
                self.start_training = None
                
    def forward(self):
        if self.rank == 0 and self.test:
            if self.start_forward==None: self.start_forward = time()
            else:
                self.time_perf_forward.append(time() - self.start_forward)
                self.start_forward = None
                
    def backward(self):
        if self.rank == 0 and self.test:
            if self.start_backward==None: self.start_backward = time()
            else:
                self.time_perf_backward.append(time() - self.start_backward)
                self.start_backward = None
                
    def validation(self):
        if self.rank == 0:
            if self.start_valid==None: self.start_valid = datetime.now()
            else: 
                self.val_time = datetime.now() - self.start_valid
                self.start_valid = None
                
    def display(self, val_steps):
        if self.rank == 0:
            print(">>> Training complete in: " + str(datetime.now() - self.start_proc))
            if self.test:
                print(">>> Training performance time: {} seconds".format(np.mean(self.time_perf_train[10:])))
                print(">>> Loading performance time: {} seconds".format(np.mean(self.time_perf_load[10:])))
                print(">>> Forward performance time: {} seconds".format(np.mean(self.time_perf_forward[10:])))
                print(">>> Backward performance time: {} seconds".format(np.mean(self.time_perf_backward[10:])))
                print(">>> Validation time estimation: {}".format(self.val_time/20 * val_steps))
                
                