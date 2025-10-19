#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import idr_torch
import numpy as np
import os
import random
import torch
import uuid
from argparse import ArgumentParser, Namespace
from pathlib import Path


os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

_sentinel = object()
BASE_SEED = 123456789
TMP_DIR: Path = Path.cwd() / ".tmp"


def seed_everything(seed=_sentinel) -> None:
    if seed is _sentinel:
        seed = BASE_SEED
    seed = uuid.uuid5(uuid.NAMESPACE_DNS, str(seed)).int % (2 ** 32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)


def make_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--bsz", type=int, default=1)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--execid", type=str)
    parser.add_argument("--tp", type=int, default=1)
    return parser


def parse_cli() -> Namespace:
    parser = make_parser()
    cfg = parser.parse_args()
    global TMP_DIR
    TMP_DIR /= cfg.execid
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    return cfg


def write_metrics(
    loss: float, accuracy: float, sync_loss: float, sync_acc: float,
    device: torch.device, duration: float,
):
    rank = idr_torch.rank
    maxmem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    totmem = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    with (TMP_DIR / f"output_rank_{rank}.dlojz").open("wb") as f:
        torch.save({
            "loss": loss,
            "accuracy": accuracy,
            "maxmem": maxmem,
            "totmem": totmem,
        }, f)
    if rank == 0:
        with (TMP_DIR / "output.dlojz").open("wb") as f:
            torch.save({
                "loss": sync_loss,
                "accuracy": sync_acc,
                "duration": duration,
            }, f)


def read_metrics(execid: str):
    TMP_DIR = Path.cwd() / ".tmp" / execid
    rank_files: list[Path] = list(TMP_DIR.glob("output_rank_*.dlojz"))
    rank_files.sort()
    print("-----------------")
    for file in rank_files:
        try:
            number = int(file.name.split("output_rank_")[1].split(".dlojz")[0])
            with open(file, "rb") as f:
                d = torch.load(f, map_location="cpu")
            print(f"Rank {number} - Loss: {d['loss']:.8f} | Acc: {d['accuracy']:.2%} | Conso mémoire: {d['maxmem']:.2f} / {d['totmem']:.2f} GB")
        except (EOFError, FileNotFoundError):
            pass

    print("-----------------")
    try:
        with (TMP_DIR / "output.dlojz").open("rb") as f:
            d = torch.load(f, map_location="cpu")
        print(f"Loss finale: {d['loss']:.8f} | Acc finale: {d['accuracy']:.2%} | Durée (entraînement et validation): {d['duration']:.2f}s")
        print("\033[96m\033[1m" + u'\u2550' * 100 + "\033[0m")
    except FileNotFoundError:
        print("\033[91mLe script ne s'est pas terminé correctement, pas de métrique synchronisée.")
        print(u'\u2550' * 100 + "\033[0m")
