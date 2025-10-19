#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import idr_torch
import os
import torch.distributed as dist
import torch
import torch.nn as nn
from argparse import ArgumentParser, Namespace
from textwrap import indent
from typing import Optional

import model
import pcomm
import tp_model
from setup import seed_everything


torch.cuda.set_device(idr_torch.local_rank)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def rand(
    size: tuple[int, ...], dtype: torch.dtype, high: Optional[int] = None
) -> torch.Tensor:
    if dtype is torch.float or dtype is torch.double:
        return torch.randn(size, device=device, dtype=dtype)
    elif dtype is torch.long or dtype is torch.int:
        assert high is not None
        return torch.randint(low=0, high=high, size=size, device=device, dtype=dtype)


def display(tensor: torch.Tensor) -> str:
    return indent(str(tensor), "\t\t")


def init_pg() -> None:
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=idr_torch.rank,
        world_size=idr_torch.world_size,
        timeout=datetime.timedelta(seconds=15),
    )

def parse_cli() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("classname", type=str)
    cfg = parser.parse_args()
    return cfg


def split_dim(src: torch.Tensor, dst: torch.Tensor, dim: int = 0) -> None:
    try:
        size_per_rank = src.size()[dim] // pcomm.tp_degree
        weight = torch.narrow(src, dim, size_per_rank * pcomm.tp_rank, size_per_rank)
        with torch.no_grad():
            dst.copy_(weight)
    except RuntimeError:
        print("Les tailles ne matchent pas bien, impossible de copier les poids.")
        exit(0)


def test_forward(
    base_mod: nn.Module,
    dist_mod: nn.Module,
    size_input: tuple[int, ...],
    in_dtype: torch.dtype = torch.float,
    vocab_size: Optional[int] = None,
    attention_size: tuple[int, ...] = None,
) -> None:
    x = rand(size_input, dtype=in_dtype, high=vocab_size)
    kwargs = dict()
    if attention_size is not None:
        attn_mask = rand(attention_size, dtype=torch.long, high=2)
        kwargs["attention_mask"] = attn_mask
    if idr_torch.master:
        print("####### Forward test #######")
        y = base_mod(x, **kwargs)
        print(f"Base module output:\n{display(y)}\n-------------------")
    dist.barrier()
    z = dist_mod(x, **kwargs)
    print(f"Rank {pcomm.tp_rank} | dist module output:\n{display(z)}")
    dist.barrier()


def test_backward(
    base_mod: nn.Module,
    dist_mod: nn.Module,
    size_input: tuple[int, ...],
    size_label: tuple[int, ...],
    in_dtype: torch.dtype = torch.float,
    out_dtype: torch.dtype = torch.float,
    vocab_size: Optional[int] = None,
    attention_size: tuple[int, ...] = None,
) -> None:
    x = rand(size_input, dtype=in_dtype, high=vocab_size)
    label = rand(size_label, dtype=out_dtype, high=vocab_size)
    kwargs = dict()
    if attention_size is not None:
        attn_mask = rand(attention_size, dtype=torch.long, high=2)
        kwargs["attention_mask"] = attn_mask
    criterion = nn.MSELoss()
    base_opt = torch.optim.SGD(base_mod.parameters(), lr=0.1)
    dist_opt = torch.optim.SGD(dist_mod.parameters(), lr=0.1)

    if idr_torch.master:
        y = base_mod(x, **kwargs)
        print("####### Backward test #######")
        loss = criterion(y, label)
        loss.backward()
        base_opt.step()
        
        string = "Base Module:\n"
        for name,param in base_mod.named_parameters():
            string += f"\t{name} grad:\n{display(param.grad)}\n"
        string += "-" * 18
        print(string)
    dist.barrier()

    y = dist_mod(x, **kwargs)
    loss = criterion(y, label)
    loss.backward()
    dist_opt.step()
    string = f"Rank {pcomm.tp_rank}\n"
    for name,param in dist_mod.named_parameters():
        string += f"\t{name} grad:\n{display(param.grad)}\n"
    string += "-" * 18
    print(string)
    dist.barrier()

    if idr_torch.master:
        print("=" * 18)
        print("####### Step test #######")
        string = "Base Module:\n"
        for name,param in base_mod.named_parameters():
            string += f"\t{name}:\n{display(param.data)}\n"
        string += "-" * 18
        print(string)
    dist.barrier()

    string = f"Rank {pcomm.tp_rank}\n"
    for name,param in dist_mod.named_parameters():
        string += f"\t{name}:\n{display(param.data)}\n"
    string += "-" * 18
    print(string)
    dist.barrier()
            


def test_forward_backward(
    base_mod: nn.Module,
    dist_mod: nn.Module,
    size_in: tuple[int, ...],
    size_out: tuple[int, ...],
    in_dtype: torch.dtype = torch.float,
    out_dtype: torch.dtype = torch.float,
    vocab_size: Optional[int] = None,
    attention_size: tuple[int, ...] = None,
) -> None:
    test_forward(
        base_mod, dist_mod, size_in, in_dtype,
        vocab_size=vocab_size, attention_size=attention_size,
    )
    if idr_torch.master:
        print("=" * 18)
    test_backward(
        base_mod, dist_mod, size_in, size_out, in_dtype, out_dtype,
        vocab_size=vocab_size, attention_size=attention_size,
    )


def test_ColRowLinearPair() -> None:
    bsz = 2
    in_dim = 2
    mid_dim = 4
    out_dim = in_dim
    pw_mid = [nn.GELU()]
    base_module = model.FeedForwardBlock(in_dim, mid_dim, out_dim, pw_mid)
    dist_module = tp_model.ColRowLinearPair(in_dim, mid_dim, out_dim, pw_mid)
    split_dim(base_module.first_layer.weight, dist_module.first_linear.weight, dim=0)
    split_dim(base_module.second_layer.weight, dist_module.second_linear.weight, dim=1)

    base_module = base_module.to(device)
    dist_module = dist_module.to(device)

    test_forward_backward(base_module, dist_module, (bsz, in_dim), (bsz, out_dim))


def test_ColWiseLinear() -> None:
    bsz = 2
    in_dim = 4
    out_dim = 8
    base_module = nn.Linear(in_dim, out_dim, bias=False)
    dist_module = tp_model.ColWiseLinear(in_dim, out_dim)
    split_dim(base_module.weight, dist_module.linear.weight, dim=0)
    
    base_module = base_module.to(device)
    dist_module = dist_module.to(device)

    test_forward_backward(base_module, dist_module, (bsz, in_dim), (bsz, out_dim))


def test_MHSA() -> None:
    bsz = 2
    seq_dim = 2
    hidden_dim = 4
    num_heads = 2

    base_module = model.MultiHeadSelfAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout_prob=0.0)
    dist_module = tp_model.MultiHeadSelfAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout_prob=0.0)

    split_dim(base_module.output.weight, dist_module.output.weight, dim=1)
    split_dim(base_module.q.weight, dist_module.q.weight, dim=0)
    split_dim(base_module.k.weight, dist_module.k.weight, dim=0)
    split_dim(base_module.v.weight, dist_module.v.weight, dim=0)

    base_module = base_module.to(device)
    dist_module = dist_module.to(device)

    test_forward_backward(
        base_module,
        dist_module,
        (bsz, seq_dim, hidden_dim),
        (bsz, seq_dim, hidden_dim),
        attention_size=(bsz, seq_dim),
    )


def test_Embedding() -> None:
    bsz = 2
    seq_dim = 2
    vocab_size = 6
    hidden_dim = 4
    base_module = nn.Embedding(vocab_size, hidden_dim)
    dist_module = tp_model.Embedding(vocab_size, hidden_dim)
    split_dim(base_module.weight, dist_module.mod.weight, dim=1)

    base_module = base_module.to(device)
    dist_module = dist_module.to(device)

    test_forward_backward(
        base_module,
        dist_module, 
        (bsz, seq_dim),
        (bsz, seq_dim, hidden_dim),
        in_dtype=torch.long,
        out_dtype=torch.float,
        vocab_size=vocab_size,
    )


def test():
    seed_everything(seed=os.environ["USER"])
    init_pg()
    cfg = parse_cli()
    pcomm.init(tp=idr_torch.world_size)

    if cfg.classname == "ColWiseLinear":
        test_ColWiseLinear()
    elif cfg.classname == "ColRowLinearPair":
        test_ColRowLinearPair()
    elif cfg.classname == "MultiHeadSelfAttention":
        test_MHSA()
    elif cfg.classname == "Embedding":
        test_Embedding()
    else:
        print("Invalid classname.")


if __name__ == "__main__":
    test()
