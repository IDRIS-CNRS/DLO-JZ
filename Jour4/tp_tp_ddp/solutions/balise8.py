def init(tp: int):
    global tp_grp, tp_rank, tp_degree, dp_grp, dp_rank, dp_degree


    ###### BALISE 8 ######
    tp_indices = list(range(
        idr_torch.rank - idr_torch.rank % tp,
        idr_torch.rank - idr_torch.rank % tp + tp,
    ))
    tp_grp = dist.new_group(
        ranks=tp_indices,
        timeout=datetime.timedelta(seconds=15),
        backend="nccl",
        use_local_synchronization=True,
    )
    tp_rank = dist.get_group_rank(tp_grp, idr_torch.rank)
    tp_degree = dist.get_world_size(tp_grp)
    ###### FIN BALISE 8 ######