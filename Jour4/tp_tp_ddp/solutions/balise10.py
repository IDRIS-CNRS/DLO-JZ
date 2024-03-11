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


    ###### BALISE 10 ######
    dp_indices = list(range(idr_torch.rank % tp, idr_torch.world_size, tp))
    dp_grp = dist.new_group(
        ranks=dp_indices,
        timeout=datetime.timedelta(seconds=15),
        backend="nccl",
        use_local_synchronization=True,
    )
    dp_rank = dist.get_group_rank(dp_grp, idr_torch.rank)
    dp_degree = dist.get_world_size(dp_grp)
    assert tp_degree * dp_degree == idr_torch.world_size
    ###### FIN BALISE 10 ######