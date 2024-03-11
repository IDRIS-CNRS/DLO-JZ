# dataset.py

def get_dataloader(tokenizer: PreTrainedTokenizer, df: pd.DataFrame, bsz: int, samples: int) -> DataLoader:
    dataset = DatasetIMDB(df=df, tokenizer=tokenizer)
    dataset = Subset(dataset, range(samples))
    ###### BALISE 11a ######
    num_replicas = pcomm.dp_degree
    rank = pcomm.dp_rank
    ###### FIN BALISE 11a ######
    sampler = DistributedSampler(
        dataset, num_replicas=num_replicas, rank=rank, shuffle=True
    )
    return DataLoader(
        dataset,
        batch_size=bsz,
        collate_fn=collate_fn,
        num_workers=8,
        sampler=sampler,
    )


# tp_tensor_parallel.py

###### BALISE 11b ######
model = DistributedDataParallel(model, process_group=pcomm.dp_grp)
###### FIN BALISE 11b ######

###### BALISE 11c ######
dist.all_reduce(sync_loss, group=pcomm.dp_grp)
sync_loss /= pcomm.dp_degree
dist.all_reduce(sync_acc, group=pcomm.dp_grp)
sync_acc /= pcomm.dp_degree
###### FIN BALISE 11c ######