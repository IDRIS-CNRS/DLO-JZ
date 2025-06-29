from IPython.display import display, Markdown
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = 'notebook_connected'
from PIL import Image
import difflib
import pandas as pd
from idr_pytools import search_log
import numpy as np
import matplotlib.pyplot as plt
import json
from glob import glob
import torch

###############################
# Author : Bertrand CABOT, Myriam Peyrounette from IDRIS(CNRS)
#
# #######################


def controle_technique(jobid):
    # jobid can either be a list, a tuple, an int, or a string
    if isinstance(jobid, (list, tuple)):
        jobid = jobid[0]
    if isinstance(jobid, int):
        jobid = str(jobid)

    n_epoch=90
    it_time=None
    load_time=None
    n_batch=None
    write_optim=False
    bs=None
    power=None
    oom=False
    log_out = search_log(contains=jobid)[0]
    with open(log_out, "r") as f:
        for line in f:
            if "Training performance" in line: 
                it_time = float(line.split(' ')[-4])
                it_time_std = float(line.split(' ')[-1][:-2])
                it_time_min = float(line.split(' ')[-6])
            if "Loading performance" in line: 
                load_time = float(line.split(' ')[-4])
                load_time_std = float(line.split(' ')[-1][:-2])
                load_time_min = float(line.split(' ')[-6])
            if "Forward performance" in line: 
                for_time = float(line.split(' ')[-4])
            if "Backward performance" in line: 
                back_time = float(line.split(' ')[-4])
            if "batch per epoch" in line: 
                n_batch = float(line.split(' ')[-1])
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
            if 'optimizer_name' in line:
                optim = line.split('  ')[-1]
            if 'optimizer_params' in line:
                optim = optim + '<br>' + ('<br>').join(line.split('  ')[-1].split('. ')[-1].split(', '))
            if 'scheduler_name' in line:
                optim = optim + '<br>' + line.split('  ')[-1]
            if 'scheduler_params' in line:
                optim = optim + '<br>' + ('<br>').join(line.split('  ')[-1].split('. ')[-1].split(', '))
            if 'Peak Power during training' in line:
                power = line.split()[-2].split('.')[0]
            
                
    layout = go.Layout(
    autosize=False, 
    width=980,
    height=500)
    
    engine=Image.open('images/noun-engine-2952049.png')
    steering=Image.open('images/noun-steering-2879261.png')
    tire=Image.open('images/noun-tire-1467520.png')
    
    throughput = int(bs)/it_time if it_time else None
    throughput_tot = int(bs)/(it_time + load_time) if it_time else None
    
    if throughput==None:
        log_err = search_log(contains=jobid, with_err=True)['stderr'][0]
        with open(log_err, "r") as f:
            for line in f:
                 if "CUDA out of memory" in line or "Out Of Memory" in line:
                    oom=True
                    break

        
        
            
    
    fig = go.Figure(go.Indicator(
    domain = {'x': [0, 0.5], 'y': [0, 1]},
    value = throughput_tot,
    mode = "gauge+number",
    title = {'text': "Sequences/second"},
    gauge = {'axis': {'range': [None, 10000]},
             'steps' : [
                 {'range': [0, 1200], 'color': "lightgray"},
                 {'range': [1200, 2800], 'color': "gray"}],
            # 'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 3000}
            }),
    layout=layout)
    
    if throughput: fig.add_annotation(x=0.25, y=0.1,
            text=f"GPU : {throughput/1000:.2f} k",
            showarrow=False,
            align='center',
            font=dict(size=28),
            xanchor='center')
    
    fig.add_layout_image(
        dict(
            source=engine,
            x=0.6,
            y=1.,
            sizex=.2,
            sizey=.2,
            xanchor='center'))

    if bs: fig.add_annotation(x=0.6, y=0.8,
            text=f"{gpu} GPU",
            showarrow=False,
            align='center',
            font=dict(size=24),
            xanchor='center')
    
    fig.add_annotation(x=0.5, y=1.2,
            text=log_out.split('/')[-1].split('.')[0],
            showarrow=False,
            align='center',
            font=dict(size=20),
            xanchor='center')

    fig.add_layout_image(
        dict(
            source=tire,
            x=0.9,
            y=1.,
            sizex=.2,
            sizey=.2,
            xanchor='center'))

    if bs: fig.add_annotation(x=0.9, y=0.8,
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
    if bs: fig.add_annotation(x=0.85, y=0.0,
            text=optim,
            showarrow=False,
            font=dict(size=16),
            xanchor='center',
            align='left')
    if power: fig.add_annotation(x=0.25, y=0.5,
            text=f"{int(power) * int(gpu)} W",
            showarrow=False,
            align='center',
            font=dict(size=28),
            xanchor='center')
    
    if throughput:
        fig.show()
        print(f'Train throughput: {throughput_tot:.2f} Sequences/second')
        print(f'GPU throughput: {throughput:.2f} Sequences/second')
        print(f'epoch time: {(it_time+load_time)*n_batch:.2f} seconds')
        print('-----------')
        print(f'training step time average (fwd/bkwd on GPU): {it_time:.6f} sec ({for_time/it_time*100:.1f}%/{back_time/it_time*100:.1f}%) +/- {it_time_std:.6f}')
        print(f'loading step time average (IO + CPU to GPU transfer): {load_time:.6f} sec +/- {load_time_std:.6f}')
        display(Markdown(f'[Click here to display the log file]({log_out})'))
            
    else:
        overheat=Image.open('images/caroverheat.jpg')
        fig.add_layout_image(
        dict(
            source=overheat,
            x=0.25,
            y=0.5,
            sizex=1.1,
            sizey=1.1,
            xanchor='center',
            yanchor='middle'))
        if oom:
            fig.add_annotation(x=0.25, y=1.,
            text="CUDA Out of Memory",
            showarrow=False,
            align='center',
            font=dict(size=32, color='red'),
            xanchor='center')
            fig.show()
            display(Markdown(f'[Please check here the error log]({log_err})'))
        else:
            fig.add_annotation(x=0.25, y=1.,
            text="Some error",
            showarrow=False,
            align='center',
            font=dict(size=32, color='black'),
            xanchor='center')
            fig.show()
            display(Markdown(f'[Please check here the error log]({log_err})'))
            display(Markdown(f'[Or please check here if Job is finished]({log_out})'))
        
        
    

def GPU_underthehood(jobids, calcul_memo=True):
    mem = []
    bsize = []
    it_time = []
    load_time = []
    model = ''
    nparam = ''
    for i,out in enumerate(np.array([search_log(contains=j) for j in jobids]).reshape(-1)):
        oom=True
        with open(out, "r") as f:
            for line in f:
                if "Max Memory Allocated" in line: 
                    mem.append(float(line.split(' ')[-2]))
                    oom=False
                if "mini batch size" in line: 
                    bsize.append(line.split(' ')[3])
                if "Training performance" in line: 
                    it_time.append(float(line.split(' ')[-4]))
                if "Loading performance" in line: 
                    load_time.append(float(line.split(' ')[-4]))
                if i == 0 and 'model:' in line:
                    model = line.split(': ')[-1]
                if i == 0 and 'number of parameters:' in line:
                    nparam = line.split(': ')[-1]
                
                
        if oom:
            with open(np.array([search_log(contains=j, with_err=True)['stderr'] 
                                for j in jobids]).reshape(-1)[i],"r") as f:
                check=False
                for line in f:
                    if "CUDA out of memory" in line or "Out Of Memory" in line:
                        mem.append('OOM')
                        check=True
                        break
                assert check, 'erreur dans le code pas de OOM trouvé!!'
    
    it_time = np.array(it_time)
    load_time = np.array(load_time)
    memo = np.array([m for m in mem if m!='OOM'])
    mem2nan = np.array([m if m!='OOM' else np.nan for m in mem ])
    if calcul_memo: 
        diff_memo = np.diff(memo)
        bs_rate = (np.array(bsize[1:], dtype=int)/np.array(bsize[:-1], dtype=int))[:len(diff_memo)]
        model_mem = (memo[1:] - bs_rate * np.diff(memo))/2**30
        model_mem = model_mem[np.where(model_mem < 1.25*np.mean(model_mem))]
    for i in np.where(np.array(mem)=='OOM')[0]:
        it_time=np.insert(it_time, i, np.nan)
        load_time=np.insert(load_time, i, np.nan)
    thrpts = np.array(bsize,dtype=int)/np.array(it_time)
    thrpts_tot = np.array(bsize,dtype=int)/np.array(it_time+load_time)
    
    if calcul_memo: modelpart=Image.open('images/GPUmemModelpart.png')
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Max GPU Memory Allocated", "Throughput"))
    if calcul_memo: fig.add_trace(go.Bar(x = bsize,y = mem2nan/2**30, name = 'GPU Memory'), 1, 1)
    else: fig.add_trace(go.Bar(x = np.arange(len(bsize))+1,y = mem2nan/2**30, name = 'GPU Memory'), 1, 1)
    fig.add_hline(36, col=1, opacity=0.)
    fig.add_hline(32,line_dash="dot", line_color='red', col=1)
    fig.add_hline(16,line_dash="dot", line_color='orange', col=1, opacity=0.5)
    if calcul_memo:
        fig.add_trace(go.Scatter(x = bsize,y = thrpts, name = 'GPU Throughput', line_color='green'), 1, 2)
        fig.add_trace(go.Scatter(x = bsize,y = thrpts_tot, name = 'Global Throughput', line_color='green', line = dict(width=1, dash='dash')), 1, 2)
    else:
        fig.add_trace(go.Scatter(x = np.arange(len(bsize))+1,y = thrpts, name = 'GPU Throughput', line_color='green'), 1, 2)
        fig.add_trace(go.Scatter(x = np.arange(len(bsize))+1,y = thrpts_tot, name = 'Global Throughput', line_color='green', line = dict(width=1, dash='dash')), 1, 2)

    
    fig.update_layout(
    showlegend=True,
    autosize=False, 
    width=1100,
    height=800,
    margin={'b':400},
    font={'size':14})
    
    fig.update_yaxes(title_text='GBytes', col=1)
    fig.update_yaxes(title_text='Sequences/s', col=2)
    if calcul_memo: fig.update_xaxes(title_text='Batch size')
    
    if calcul_memo: fig.add_layout_image(
        dict(
            source=modelpart,
            x=0.5,
            y=-.75,
            sizex=1.,
            sizey=1.,
            xanchor='center',
            yanchor='middle'))
    
    if calcul_memo: fig.add_annotation(x=0.15, y=-1.05,
            text=f"{np.mean(model_mem):.3f} GB",
            showarrow=False,
            align='center',
            font=dict(size=30),
            xanchor='center',
            xref='paper',
            yref='paper')
    
    fig.add_annotation(x=0.5, y=1.25,
            text=f"{model}: {int(nparam)/1e6:.1f} Million parameters",
            showarrow=False,
            align='center',
            font=dict(size=24),
            xanchor='center',
            xref='paper',
            yref='paper')

    fig.show()
    
    for i, bs in enumerate(bsize):
        if mem[i] == 'OOM':
            print(f'Batch size per GPU: {bs} CUDA out of memory')
        else:
            print(f'Batch size per GPU: {bs} Max GPU Memory Allocated: {mem[i]/2**30:.2f} GB, Troughput: {int(bs)/it_time[i]:.3f} Sequences/second')
        
    if calcul_memo: print(f'Memory occupancy by Model part : {np.mean(model_mem):.3f} +/- {np.std(model_mem):.3f} GB')
    
    

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
                accs.append(float(line.split(',')[-2].split(' ')[-1]))
            elif "Step" in line and "Loss:" in line:
                train_accs.append(float(line.split(',')[-2].split(':')[-1]))
            elif "global batch size" in line:
                bsize = line.split(' ')[3]
            elif "(" in line and opt == None:
                opt = line.split(' ')[1]
            elif "lr:" in line:
                lr = line.split(' ')[-1][:-1]
            elif "weight_decay:" in line and wd == None:
                wd = line.split(' ')[-1][:-1]
            elif "Training complete in:" in line:
                time_train = line[:-1]
            elif "model:" in line:
                model = line.split()[-1]
            elif "image batch shape" in line:
                isize = line.split()[-1][:-2]
    return accs, train_accs, f'{opt} bs: {bsize} is: {isize} lr: {lr} wd: {wd} top-1: {max(accs)}', time_train, model

def plot_accuracy(jobids):
    jobids = np.array(jobids).reshape(-1,1)
    fig, ax = plt.subplots(figsize=(12,7))
    times = []
    colors = ['blue', 'orange', 'green', 'cyan', 'purple', 'pink', 'brown', 'gold']
    for i, j in enumerate(jobids):
        acc, train_acc, label, time_train, model = val_acc(j)
        ax.plot(np.arange(len(acc))+1, acc, label=label, color=colors[i%8])
        ax.plot((np.arange(len(train_acc))+1)/(len(train_acc)/len(acc)), train_acc, '--', alpha=0.35, color=colors[i%8])
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
    plt.ylim(0.95*min(trace), 1.05*max(trace))
    plt.show()

def time_train(jobid):
    time = None
    with open(search_log(contains=jobid[0])[0], "r") as f:
        for line in f:
            if "Training complete" in line: 
                time = line.split(' ')[-1].split('\n')[0]
    return pd.Timedelta(time) / pd.Timedelta('1 hour')

def imagenet_starter(jobid, lr=None, moment=0.9, weight_decay=5e-4, jour2=False):
    log_out = search_log(contains=jobid[0])[0]
    load_time = 0
    with open(log_out, "r") as f:
        for line in f:
            if "Training performance" in line: 
                it_time = float(line.split(' ')[-4])
            if "Loading performance" in line: 
                load_time = float(line.split(' ')[-4])
            if "batch per epoch" in line: 
                n_batch = float(line.split(' ')[-1])   
            if "image batch shape" in line:
                isize = line.split()[-1][:-2]
            if "mini batch size:" in line:
                bs = int(line.split()[-1])
            if ">>> Training on " in line:
                gpu = line.split()[-2]
           
    el_epochs = round(1800 / ((it_time+load_time)*n_batch/(32/int(gpu)) + 20))
    if lr==None:
        lr = bs / 512
    file = 'dlojz_imagenetrace.py' if jour2 else 'dlojz.py'
    return f'{file} -b {bs} -e {el_epochs} --image-size {isize} --lr {lr} --mom {moment} --wd {weight_decay}'
            
def plot_time(jobid):
    steps_times = []
    validation_times = []
    with open(search_log(contains=jobid[0])[0], "r") as f:
        for line in f:
            if ", Time:" in line: 
                steps_times.append(float(line.split(', ')[2].split()[-1]))
            if ">>> Validation complete in:" in line: 
                steps_times.append(float(line.split(':')[-1]))
                validation_times.append(float(line.split(':')[-1]))
    fig, ax = plt.subplots(figsize=(18,5))
    plt.bar((np.arange(len(steps_times))+1)/(len(steps_times)/len(validation_times)),steps_times, width=0.2, label='Training')
    plt.bar(np.arange(len(validation_times))+1,validation_times, width=0.2, label='Validation', color='purple')
    plt.legend(fontsize=15)
    plt.title('Measured times during training loop', fontsize=18)
    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('Tme(s)', fontsize=15)
    plt.show()
           
                
def pipe_memory(jobid, n_gpu=4):
    mem=[np.nan]*n_gpu
    log_out = search_log(contains=jobid[0])[0]
    with open(log_out, "r") as f:
        for line in f:
            if "Max Memory Allocated" in line: 
                mem[0]=float(line.split()[-2])/2**30
            if "MaxMemory for GPU:" in line:
                mem[int(line.split()[2].split(':')[-1])%n_gpu]=float(line.split()[3])/2**30
                
    plt.bar(np.arange(n_gpu), mem)
    plt.title('Pipeline Parallelism Memory Sharing', fontsize=14)
    plt.xlabel('GPU device number', fontsize=14)
    plt.xticks(np.arange(n_gpu))
    plt.ylabel('Max Memory Allocated (GB)', fontsize=14)
    plt.show()
    
def turbo_profiler(jobid, dataloader_info=False):
    log_out = search_log(contains=jobid[0])[0]
    with open(log_out, "r") as f:
        for line in f:
            if "Training complete" in line: 
                time = line.split(' ')[-1].split('\n')[0]
                training_time = float(time.split(':')[1])*60 + float(time.split(':')[2])
            if "Training performance" in line: 
                it_time = float(line.split(' ')[-4])
            if "Loading performance" in line:
                load_time = float(line.split(' ')[-4])
            if "JSON" in line:
                perf = json.loads(line.split('>>>JSON ')[-1])
            if dataloader_info and "DATALOADER" in line:
                num_workers = line.split(' ')[1]
                persistent_workers = line.split(' ')[2]
                pin_memory = line.split(' ')[3]
                non_blocking = line.split(' ')[4]
                prefetch_factor = line.split(' ')[5]
                drop_last = line.split(' ')[6]
            if dataloader_info and "VmHWM" in line:
                cpu_mem_usage = int(line.split(' ')[-2])/2**20
                
    print(f"\033[1m>>> Turbo Profiler >>>\033[0m Training complete in {training_time} s")
    pd.DataFrame(perf).plot(kind='bar', figsize=(18, 4))
    if dataloader_info:
        plt.title(f'>>> Turbo Profiler >>> CPU Memory Usage: {cpu_mem_usage} GB', fontsize=16)
    else:
        plt.title('>>> Turbo Profiler >>>', fontsize=16)
    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('Time in seconds', fontsize=14)
    plt.ylim(top=8)
    axes = plt.gca()
    plt.show()

    
    if dataloader_info: 
        dataloader_trial = pd.DataFrame({"jobid":[str(jobid[0])],
                                         "num_workers":[int(num_workers)],
                                         "persistent_workers":[str(persistent_workers)],
                                         "pin_memory":[str(pin_memory)],
                                         "non_blocking":[str(non_blocking)],
                                         "prefetch_factor":[int(prefetch_factor)],
                                         "drop_last":[str(drop_last)],
                                         "loading_time":[float(load_time)],
                                         "CPU_memory_usage(GB)":[float(cpu_mem_usage)]})
                                         #"training_time":[float(training_time)]})
                                         #"forward_backward_time":[float(it_time)],
                                         #"iteration_time":[float(it_time)+float(load_time)],
        return dataloader_trial


def comm_profiler(jobid, n_display=None):
    # jobid can either be a list, a tuple, an int, or a string
    if isinstance(jobid, (list, tuple)):
        jobid = jobid[0]
    if isinstance(jobid, int):
        jobid = str(jobid)

    coll_dict = {}
    coll_dict['local_rank'] = []
    coll_dict['coll_operation'] = []
    coll_dict['opCount'] = []
    coll_dict['comm'] = []
    coll_dict['Count'] = []
    coll_dict['datatype'] = []
    coll_dict['op'] = []
    coll_dict['root'] = []
    coll_dict['stream'] = []
    coll_dict['sendbuff'] = []
    coll_dict['recvbuff'] = []
    coll_dict['train_step'] = []
    log_out = search_log(contains=jobid)[0]
    with open(log_out, "r") as f:
        for line in f:
            if ">>> Training on" in line:
                n_rank = int(line.split()[-2])
                break
    steps = ['init.']*n_rank
    
    comm_rank = {}

    with open(log_out, "r") as f:
        for line in f:
            if "NCCL INFO" in line and "opCount" in line and "datatype" in line: 
                coll_trace = line.split()
                coll_dict['local_rank'].append(int(coll_trace[1][1:-1]))
                coll_dict['coll_operation'].append(coll_trace[4][:-1])
                coll_dict['opCount'].append(int(coll_trace[6], 16))
                comm = coll_trace[20]
                coll_dict['comm'].append(comm)
                coll_dict['Count'].append(np.int64(coll_trace[12]))
                coll_dict['datatype'].append(np.int32(coll_trace[14]))
                coll_dict['op'].append(np.int32(coll_trace[16]))
                coll_dict['root'].append(np.int32(coll_trace[18]))
                coll_dict['stream'].append(coll_trace[23])
                coll_dict['sendbuff'].append(coll_trace[8])
                coll_dict['recvbuff'].append(coll_trace[10])
                coll_dict['train_step'].append(steps[comm_rank[comm]])
                
            elif "Init COMPLETE" in line:
                trace = line.split()
                comm_rank[trace[-16]] = int(trace[-14])
                
            elif "Train step" in line:
                step = int(line.split()[2])
                rank = int(line.split()[-1])
                steps[rank] = 'init.' if step==0 else 'valid.' if step==100 else f'step {step}'
                
                        
    df = pd.DataFrame(coll_dict)
    df = df[df.train_step.isin(['init.', 'valid.']+[f'step {i}'for i in np.arange(10)+1])]
    df['global_rank'] = [comm_rank[c] for c in df.comm]
    df = df.set_index('opCount').sort_index()
    
    df.loc[df[df.datatype.isin([2,3,7])].index, 'Count'] *= 4
    df.loc[df[df.datatype.isin([4,5,8])].index, 'Count'] *= 8
    df.loc[df[df.datatype.isin([6,9])].index, 'Count'] *= 2
    
    dico = {}
    for r in np.sort(df.global_rank.unique()):
        dico[f'rank: {r}'] = df[df.global_rank==r].Count
    nccldtype = { 0: 'ncclInt8',
              1: 'ncclUint8',
              2: 'ncclInt32',
              3: 'ncclUint32',
              4: 'ncclInt64',
              5: 'ncclUint64',
              6: 'ncclFloat16',
              7: 'ncclFloat32',
              8: 'ncclFloat64',
              9: 'ncclBfloat16',
              10: 'ncclNumTypes'
            }
    dico['operations'] = df[df.global_rank==0].train_step + ' - ' + df[df.global_rank==0].coll_operation + ' - (' + df[df.global_rank==0].datatype.replace(nccldtype) + ')'
    dfplot = pd.DataFrame(dico)
    dfplot = dfplot.set_index('operations')
    if n_display:
        dfplot.iloc[:n_display].plot.bar(figsize=(18, 3), rot=90, title=f'Collective Communication Profiler - Nbr of operations: {len(df)}', ylabel='communications Bytes')
        dfplot.iloc[-n_display:].plot.bar(figsize=(18, 3), rot=90, title=f'Collective Communication Profiler - Nbr of operations: {len(df)}', ylabel='communications Bytes')
    else:
        dfplot.plot.bar(figsize=(18, 3), rot=90, title=f'Collective Communication Profiler - Nbr of operations: {len(df)}', ylabel='communications Bytes')
    dfplot = dfplot.groupby('operations', sort=False).sum()
    dfplot.plot.bar(figsize=(18, 3), rot=90, title=f'Aggregate Collective Communication Profiler - global count: {df.Count.sum()} Bytes', ylabel='communications Bytes')
    
    plt.show()
    
    
def BatchNorm_view(jobid, model, labels=None):
    layers_w = [k for k in model.state_dict().keys() if 'bn' in k and 'weight' in k]
    layers_b = [k for k in model.state_dict().keys() if 'bn' in k and 'bias' in k]
    layers_rmean = [k for k in model.state_dict().keys() if 'bn' in k and 'running_mean' in k]
    layers_rvar = [k for k in model.state_dict().keys() if 'bn' in k and 'running_var' in k]
    
    data_w = []
    data_b = []
    data_rmean = []
    data_rvar = []
    
    for j in jobid:
        checkpoint = torch.load(glob(f'checkpoints/{j}_*')[0], map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint, strict=False)
        data_w.append([(model.state_dict()[l].mean()).item() for l in layers_w])
        data_b.append([(model.state_dict()[l].mean()).item() for l in layers_b])
        data_rmean.append([(model.state_dict()[l].mean()).item() for l in layers_rmean])
        data_rvar.append([(model.state_dict()[l].mean()).item() for l in layers_rvar])
    
    
    bnlayers = [s[:-7] for s in layers_w]
    fig, ax = plt.subplots(2,2,figsize=(24,12))
    ax[0,0].set_title('running_mean', fontsize=16)
    ax[1,0].set_title('running_var', fontsize=16)
    ax[1,0].set_xticks(np.arange(len(bnlayers)))
    ax[1,0].set_xticklabels(bnlayers, rotation='vertical', fontsize=12)
    ax[0,1].set_title('weight', fontsize=16)
    ax[1,1].set_title('bias', fontsize=16)
    ax[1,1].set_xticks(np.arange(len(bnlayers)))
    ax[1,1].set_xticklabels(bnlayers, rotation='vertical', fontsize=12)
    for i in range(len(jobid)):
        ax[0,0].plot(data_rmean[i])
        ax[1,0].plot(data_rvar[i])
        if labels:
            ax[1,1].plot(data_b[i], label=labels[i])
            ax[0,1].plot(data_w[i], label=labels[i])
        else:
            ax[1,1].plot(data_b[i])
            ax[0,1].plot(data_w[i])
    if labels:
        ax[1,1].legend(fontsize=14)
        ax[0,1].legend(fontsize=14)
    plt.show()
    

                
