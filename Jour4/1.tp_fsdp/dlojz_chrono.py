from datetime import datetime
from time import time
import numpy as np
import json

###############################
#Author : Bertrand CABOT from IDRIS(CNRS)
#
########################


class Chronometer:
    def __init__(self, test, rank):
        self.test = test
        self.rank = rank
        self.time_perf_train = []
        self.time_perf_load = []
        self.time_perf_forward = []
        self.time_perf_backward = []
        self.power = []
        self.start_proc = None
        self.start_training = None
        self.start_dataload = None
        self.start_backward = None
        self.start_forward = None
        self.time_point = None
        
    
    def tac_time(self, clear=False):
        if self.time_point == None or clear:
            self.time_point = time()
            return
        else:
            new_time = time() - self.time_point
            self.time_point = time()
            return new_time
    
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
                
                
    def display(self):
        if self.rank == 0:
            print(">>> Training complete in: " + str(datetime.now() - self.start_proc))
            if self.test:
                print(">>> Training performance time: min {} avg {} seconds (+/- {})".format(np.min(self.time_perf_train[1:]), np.median(self.time_perf_train[1:]), np.std(self.time_perf_train[1:])))
                print(">>> Loading performance time: min {} avg {} seconds (+/- {})".format(np.min(self.time_perf_load[1:]), np.mean(self.time_perf_load[1:]), np.std(self.time_perf_load[1:])))
                print(">>> Forward performance time: {} seconds (+/- {})".format(np.mean(self.time_perf_forward[1:]), np.std(self.time_perf_forward[1:])))
                print(">>> Backward performance time: {} seconds (+/- {})".format(np.mean(self.time_perf_backward[1:]), np.std(self.time_perf_backward[1:])))
                print(">>> Sortie trace #####################################" )
                print(">>>JSON", json.dumps({'GPU process - Forward/Backward':self.time_perf_train, 'CPU process - Dataloader':self.time_perf_load}))
                
