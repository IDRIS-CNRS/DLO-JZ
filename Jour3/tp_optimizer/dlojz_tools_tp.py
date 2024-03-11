import plotly.graph_objects as go
from PIL import Image
import difflib
import pandas as pd
from idr_pytools import search_log
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import torch

class JobCrashError(Exception):
    def __init__(self, jobid, message=None):
        self.jobid = jobid
        self.path = search_log(contains=jobid)
        self.message = f"Job {self.jobid} might have crashed. \nCheck {self.path} \n and the error file.\n {message}"
        super().__init__(self.message)
    pass

def controle_technique(jobid):
    n_epoch=36
    it_time=None
    load_time=None
    n_batch=None
    write_optim=False 
    with open(search_log(contains=jobid[0])[0], "r") as f:
        for line in f:
            if "Training performance" in line: 
                it_time = float(line.split(' ')[-2])
            if "Loading performance" in line: 
                load_time = float(line.split(' ')[-2])
            if "batch per epoch" in line: 
                n_batch = float(line.split(' ')[-1])
            if "time estimation" in line: 
                val_time = float(line.split(':')[-1]) + float(line.split(':')[-2])*60
            if ">>> Training on " in line:
                gpu = line.split()[-2]
            if "global batch size" in line:
                bs = line.split()[3]
            if "Optimizer:" in line:
                write_optim=True
                optim = line.split(':')[-1] + '<br>'
            elif write_optim:
                optim = optim + line + '<br>'
                if ')' in line and '(' not in line:
                    write_optim=False
                
    layout = go.Layout(
    autosize=False, 
    width=980,
    height=500)
    
    engine=Image.open('images/noun-engine-2952049.png')
    steering=Image.open('images/noun-steering-2879261.png')
    tire=Image.open('images/noun-tire-1467520.png')
    
    throughput = int(bs)/it_time
    
    fig = go.Figure(go.Indicator(
    domain = {'x': [0, 0.5], 'y': [0, 1]},
    value = throughput,
    mode = "gauge+number",
    title = {'text': "Images/second"},
    gauge = {'axis': {'range': [None, 6000]},
             'steps' : [
                 {'range': [0, 1200], 'color': "lightgray"},
                 {'range': [1200, 5000], 'color': "gray"}],
             'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 4300}}),
    layout=layout)
    
    fig.add_layout_image(
        dict(
            source=engine,
            x=0.6,
            y=1.,
            sizex=.2,
            sizey=.2,
            xanchor='center'))

    fig.add_annotation(x=0.6, y=0.8,
            text=f"{gpu} GPU",
            showarrow=False,
            align='center',
            font=dict(size=24),
            xanchor='center')

    fig.add_layout_image(
        dict(
            source=tire,
            x=0.9,
            y=1.,
            sizex=.2,
            sizey=.2,
            xanchor='center'))

    fig.add_annotation(x=0.9, y=0.8,
            text=f"batch size: {bs}",
            showarrow=False,
            align='center',
            font=dict(size=24),
            xanchor='center')

    fig.add_layout_image(
        dict(
            source=steering,
            x=0.65,
            y=0.6,
            sizex=.2,
            sizey=.2,
            xanchor='center'))
    fig.add_annotation(x=0.85, y=0.1,
            text=optim,
            showarrow=False,
            font=dict(size=16),
            xanchor='center',
            align='left')

    fig.show()
    print(f'throughput: {throughput:.2f} images/second')
    print(f'epoch time: {(it_time+load_time)*n_batch:.2f} seconds')
    print(f'training time estimation for 36 epochs (with validations): {((it_time+load_time)*n_batch+val_time)*n_epoch/3600:.2f} hours')
    print('-----------')
    print(f'training step time average (fwd/bkwd on GPU): {it_time} sec')
    print(f'loading step time average (CPU to GPU): {load_time} sec')
    print('-----------')
    if throughput > 6000:
        print('SELECTED in 50 epochs competition')
    elif throughput > 4300:
        print('SELECTED in 36 epochs competition')
    else:
        print('NOT SELECTED in competition')
    

def memory_check(jobids):
    mem = []
    bsize = []
    it_time = []
    for i,out in enumerate(np.array([search_log(contains=j) for j in jobids]).reshape(-1)):
        oom=True
        with open(out, "r") as f:
            for line in f:
                if "Max Memory Allocated" in line: 
                    mem.append(float(line.split(' ')[-2])/2**30)
                    oom=False
                if "mini batch size" in line: 
                    bsize.append(line.split(' ')[-1])
                if "Training performance" in line: 
                    it_time.append(float(line.split(' ')[-2]))
                
        if oom:
            with open(np.array([search_log(contains=j, with_err=True)['stderr'] 
                                for j in jobids]).reshape(-1)[i],"r") as f:
                check=False
                for line in f:
                    if "CUDA out of memory" in line:
                        mem.append('OOM')
                        check=True
                assert check, 'erreur dans le code pas de OOM trouvé!!'
                
    for i, bs in enumerate(bsize):
        if mem[i] == 'OOM':
            print(f'Batch size per GPU: {bs} CUDA out of memory')
        else:
            print(f'Batch size per GPU: {bs} Max GPU Memory Allocated: {mem[i]:.2f}, Troughput: {int(bs)/it_time[i]:.3f} images/second')
        print('')
    
    

def compare(script1, script2):

    a = open(script1, "r").readlines()
    b = open(script2, "r").readlines()

    difference = difflib.HtmlDiff(tabsize=2)

    with open("compare.html", "w") as fp:
        html = difference.make_file(fromlines=a, tolines=b, fromdesc=script1, todesc=script2)
        fp.write(html)
        
        
def val_acc(jobid):
    accs = []
    train_accs = []
    opt = None
    time_train = ' '
    wd = None
    model = 'no model name'
    with open(search_log(contains=jobid[0])[0], "r") as f:
        for line in f:
            if "Validation Accuracy" in line: 
                accs.append(float(line.split(' ')[-1]))
            if "Step" in line and "Loss:" in line:
                train_accs.append(float(line.split()[-1].split(':')[-1]))
            if "global batch size" in line:
                bsize = line.split(' ')[3]
            if "(" in line and opt == None:
                opt = line.split(' ')[1]
            if "lr:" in line:
                lr = line.split(' ')[-1][:-1]
            if "weight_decay:" in line and wd == None:
                wd = line.split(' ')[-1][:-1]
            if "Training complete in:" in line:
                time_train = line[:-1]
            if "model:" in line:
                model = line.split()[-1]
            if " Number of batch per epoch:" in line:
                Nbatch = int(line.split()[-1])
    try:
        return accs, train_accs, f'{opt} bs: {bsize} lr: {lr} wd: {wd} top-1: {max(accs)}', time_train, model, Nbatch
    except Exception as e:
        raise JobCrashError(jobid[0], message=str(e))

def plot_accuracy(jobids):
    jobids = np.array(jobids).reshape(-1,1)
    fig, ax = plt.subplots(figsize=(12,7))
    times = []
    colors = ['blue', 'orange', 'green', 'cyan', 'purple', 'pink', 'brown', 'gold']
    for i, j in enumerate(jobids):
        acc, train_acc, label, time_train, model, Nbatch = val_acc(j)
        ax.plot(np.arange(len(acc))/2+0.5, acc, label=label, color=colors[i%8])
        ax.plot(np.arange(len(train_acc))/Nbatch, train_acc, '--', alpha=0.35, color=colors[i%8])
        times.append(model+': '+label+' '+time_train)
    plt.legend(fontsize=15)
    plt.title('Validation Accuracies', fontsize=18)
    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('Validation Accuracy', fontsize=15)
    plt.show()
    for t in times:
        print(t)
    
def smooth(liste, beta=0.98):
    avg = 0.
    threshold = 0.
    smoothed_list = []
    for i,l in enumerate(liste):
        #Compute the smoothed loss
        avg = beta * avg + (1-beta) *l
        smoothed = avg / (1 - beta**(i+1))
        #Stop if the loss is exploding
        if i > len(liste)//2 and smoothed >= threshold:
            break
        #Record the best loss
        if i==len(liste)//3:
            threshold = smoothed
        smoothed_list.append(smoothed)
    return smoothed_list

def lrfind_plot(jobid):
    with open(search_log(contains=jobid[0])[0], "r") as f:
        for line in f:
            if "learning rates:" in line: 
                lr=[float(x) for x in line.split(': ')[-1][1:-2].split(', ')]
            if "loss list:" in line: 
                loss=[float(x) for x in line.split(': ')[-1][1:-2].split(', ')]
            if "accuracies:" in line: 
                acc=[float(x) for x in line.split(': ')[-1][1:-2].split(', ')]
    fig, ax = plt.subplots(figsize=(10,5))
    trace=smooth(loss)
    plt.plot(lr[:len(loss)], loss, color='lightsteelblue', alpha=0.4)
    plt.plot(lr[:len(trace)], trace, color='navy')
    
    plt.title('LR Finder', fontsize=18)
    plt.xlabel('learning rate', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.xscale('log')
    plt.xticks(np.array([np.arange(1,10)*10**(-8+i) for i in range(1,10)]).flatten())
    plt.ylim(0, 1.05*max(trace))
    plt.show()

def time_train(jobid):
    time = None
    with open(search_log(contains=jobid[0])[0], "r") as f:
        for line in f:
            if "Training complete" in line: 
                time = line.split(' ')[-1].split('\n')[0]
    return pd.Timedelta(time) / pd.Timedelta('1 hour')


def val_acc_lr(jobid):
    accs = []
    lrs = []
    opt = None
    time_train = ' '
    wd = None
    model = 'no model name'
    logfile = search_log(contains=jobid[0])[0]
    with open(logfile, 'r') as f:
        for line in f:
            if "Validation Accuracy" in line: 
                accs.append(float(line.split(' ')[-1]))
            if "Learning Rate" in line:
                lrs.append(float(line.split(' ')[-1]))
            if "global batch size" in line:
                bsize = line.split(' ')[3]
            if "(" in line and opt == None:
                opt = line.split(' ')[1]
            if "lr:" in line:
                lr = line.split(' ')[-1][:-1]
            if "weight_decay:" in line and wd == None:
                wd = line.split(' ')[-1][:-1]
            if "Training complete in:" in line:
                time_train = line[:-1]
            if "model:" in line:
                model = line.split()[-1]
    try:
        return accs, lrs, f'{opt} bs: {bsize} lr: {lr} wd: {wd} top-1: {max(accs)}', time_train, model
    except Exception as e:
        raise JobCrashError(jobid[0], message=str(e))

    
def plot_accuracy_lr(jobids):
    jobids = np.array(jobids).reshape(-1,1)
    fig, ax = plt.subplots(1,2,figsize=(24,7))
    times = []
    for j in jobids:
        acc, lrs, label, time_train, model = val_acc_lr(j)
        ax[0].plot(np.arange(len(acc))/2, acc, label=label)
        ax[1].plot(np.arange(len(lrs))/2, lrs, label=label)
        times.append(model+': '+label+' '+time_train)
    ax[0].legend(fontsize=15)
    ax[0].set_title('Validation Accuracies', fontsize=18)
    ax[0].set_xlabel('Epochs', fontsize=15)
    ax[0].set_ylabel('Validation Accuracy', fontsize=15)
    ax[1].legend(fontsize=15)
    ax[1].set_title('Learning rates', fontsize=18)
    ax[1].set_xlabel('Epochs', fontsize=15)
    ax[1].set_ylabel('Learning rates', fontsize=15)
    plt.show()
    for t in times:
        print(t)

                

            
## Utils model vizu
def model_vizu(model, jobid=None):
    if jobid:
        title = 'Final Model Parameters'
        checkpoint = torch.load('checkpoints/' + jobid + '.pt', map_location=torch.device('cpu'))

        from collections import OrderedDict
        new_checkpoint = OrderedDict()
        for k, v in checkpoint['final_model'].items():
            name = k[7:] # remove 'module.' of DataParallel/DistributedDataParallel
            new_checkpoint[name] = v

        model.load_state_dict(new_checkpoint)
        
    else:
        title = 'Initial Model Parameters'
    
    layers = [n for i,n in enumerate([k for k in model.state_dict().keys() if 'running' not in k and 'tracked' not in k]) if 'conv' in n or 'fc' in n]
    params = [model.state_dict()[n].cpu().numpy().flatten() for n in layers]

    fig, ax = plt.subplots(figsize=(10,4))
    ax.boxplot(params)
    ax.set_xticklabels(layers, rotation='vertical')
    ax.set_title(title, fontsize=16)
    plt.show()
    
def update_vizu(model, jobid):
    
    layers = [n for i,n in enumerate([k for k in model.state_dict().keys() if 'running' not in k and 'tracked' not in k]) if 'conv' in n or 'fc' in n]
    checkpoint = torch.load('checkpoints/' + jobid + '.pt', map_location=torch.device('cpu'))

    fig, ax = plt.subplots(1,3,figsize=(24,7))
    titles = ['Updates - epoch 1', 'Updates - 1/3 of training', 'Updates - Final epoch']

    for i, upd in enumerate(['update_1', 'update_2', 'update_3']):
        ax[i].boxplot(checkpoint[upd])
        ax[i].set_xticklabels(layers, rotation='vertical')
        ax[i].set_title(titles[i], fontsize=16)      

    plt.show()

    
def model_distribution(model, jobid=None):
    if jobid:
        checkpoint = torch.load(glob(f'checkpoints/{jobid}_*')[0], map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint, strict=False)
    layers = [k for k in model.state_dict().keys() if 'conv' in k or 'fc' in k]
    data = [model.state_dict()[l].flatten() for l in layers]
    fig, axes = plt.subplots(figsize=(8, 3))
    axes.boxplot(data)
    axes.set_title('parameters distribution')
    axes.set_xticklabels(layers, rotation='vertical')
    plt.show()
    
def weight_norm(jobid):
    w2 = []
    opt = None
    time_train = ' '
    wd = None
    model = 'no model name'
    logfile = search_log(contains=jobid[0])[0]
    with open(logfile, 'r') as f:
        for line in f:
            if "||w||2:" in line:
                w2.append(float(line.split(' ')[-1]))
            if "global batch size" in line:
                bsize = line.split(' ')[3]
            if "(" in line and opt == None:
                opt = line.split(' ')[1]
            if "lr:" in line:
                lr = line.split(' ')[-1][:-1]
            if "weight_decay:" in line and wd == None:
                wd = line.split(' ')[-1][:-1]
            if "Training complete in:" in line:
                time_train = line[:-1]
            if "model:" in line:
                model = line.split()[-1]
    try:
        return w2, f'{opt} bs: {bsize} lr: {lr} wd: {wd}', time_train, model
    except Exception as e:
        raise JobCrashError(jobid[0], message=str(e))

    
def plot_weight(jobids):
    jobids = np.array(jobids).reshape(-1,1)
    fig, ax = plt.subplots(1,1,figsize=(8,5))
    times = []
    for j in jobids:
        w2, label, time_train, model = weight_norm(j)
        ax.plot(np.arange(len(w2))/2, w2, label=label)
        times.append(model+': '+label+' '+time_train)
    ax.legend(fontsize=15)
    ax.set_title('model parameters', fontsize=14)
    ax.set_xlabel('Epochs', fontsize=12)
    ax.set_ylabel('||w||2', fontsize=12)
    plt.show()