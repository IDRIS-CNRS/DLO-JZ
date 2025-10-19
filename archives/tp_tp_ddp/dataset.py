#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from transformers import PreTrainedTokenizer

import pcomm


def load_df() -> tuple[pd.DataFrame, pd.DataFrame]:
    folder = Path.cwd() / "data"
    train_dataframe = pd.read_csv(folder / "dataset_train.csv")
    val_dataframe = pd.read_csv(folder / "dataset_val.csv")
    return train_dataframe, val_dataframe


class DatasetIMDB(Dataset):

    def __init__(self, df, tokenizer):
        self.df = df
        self.length = df.shape[0]
        self.tokenizer = tokenizer

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        result = self.tokenizer(sample['review'], truncation=True, padding="max_length", max_length=512, return_tensors='pt')
        return (
            result["input_ids"],
            result["attention_mask"],
            torch.tensor(sample['label'])
        )

def collate_fn(samples):
    input_ids, attention_mask, labels = zip(*samples)
    return (
        torch.cat(input_ids, dim=0),
        torch.cat(attention_mask, dim=0),
        torch.stack(labels, dim=0)
    )

def get_dataloader(tokenizer: PreTrainedTokenizer, df: pd.DataFrame, bsz: int, samples: int) -> DataLoader:
    dataset = DatasetIMDB(df=df, tokenizer=tokenizer)
    dataset = Subset(dataset, range(samples))
    ###### BALISE 11a ######
    num_replicas = 1
    rank = 0
    ###### FIN BALISE 11a ######
    sampler = DistributedSampler(
        dataset, num_replicas=num_replicas, rank=rank, shuffle=True
    )
    return DataLoader(
        dataset,
        batch_size=bsz,
        collate_fn=collate_fn,
        num_workers=8,
        # sampler=sampler, # BALISE 11 - À DÉCOMMENTER
    )

def get_dataloaders(tokenizer: PreTrainedTokenizer, train_bsz: int, val_bsz: int, samples: int) -> tuple[DataLoader, DataLoader]:
    train_df, val_df = load_df()
    train_loader = get_dataloader(tokenizer, train_df, train_bsz, samples)
    val_loader = get_dataloader(tokenizer, val_df, val_bsz, 128)
    return train_loader, val_loader
