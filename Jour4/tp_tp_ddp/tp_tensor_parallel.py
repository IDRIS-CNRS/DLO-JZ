#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import idr_torch
import os
import time
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from pathlib import Path
from torch.cuda.amp import autocast, GradScaler
from transformers import BertTokenizer

import pcomm
from dataset import get_dataloaders
from tp_model import Transformer
from setup import seed_everything, parse_cli, write_metrics

seed_everything(seed=os.environ["USER"])
cfg = parse_cli()
assert idr_torch.world_size % cfg.tp == 0

dist.init_process_group(
    backend='nccl',
    init_method="env://",
    rank=idr_torch.rank,
    world_size=idr_torch.world_size,
    timeout=datetime.timedelta(seconds=15),
)
pcomm.init(tp=cfg.tp)

torch.cuda.set_device(idr_torch.local_rank)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer(Path.cwd() / "data" / "vocab.txt", do_lower_case=True)

model = Transformer(
    vocab_size=tokenizer.vocab_size,
    num_layers=cfg.layers,
    hidden_dim=cfg.dim,
    intermediate_dim=cfg.dim * 4,
    num_heads=cfg.heads,
    dropout_prob=0.1,
)

model = model.to(device)

###### BALISE 11b ######

###### FIN BALISE 11b ######

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(), lr=3e-05,
    betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01
)

train_loader, val_loader = get_dataloaders(tokenizer, cfg.bsz, 2 * cfg.bsz, samples=cfg.samples)


###### TRAINING ######
BEGINNING = time.perf_counter()
model.train()
scaler = GradScaler()
for epoch in range(cfg.epochs):

    for i, (input_ids, attention_mask, labels) in enumerate(train_loader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        with autocast():
            probs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(probs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()


###### EVALUATION ######
total_loss = torch.tensor(0., device=device)
accuracy = torch.tensor(0., device=device)
model.eval()
for (input_ids, attention_mask, labels) in val_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            probs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(probs, labels)
        total_loss += loss.detach()
        accuracy += (torch.argmax(probs.detach(), dim=-1) == labels).sum()

N_samples = len(val_loader) * val_loader.batch_size
total_loss /= N_samples
accuracy /= N_samples

sync_loss = torch.clone(total_loss)
sync_acc = torch.clone(accuracy)

###### BALISE 11c ######

###### FIN BALISE 11c ######


END = time.perf_counter()
duration = END - BEGINNING

write_metrics(total_loss, accuracy, sync_loss, sync_acc, device, duration)
