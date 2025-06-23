import os
import datasets
import functools
import idr_torch
import time
import torch
import torch.distributed as dist
from argparse import ArgumentParser, Namespace, BooleanOptionalAction
from pathlib import Path
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn import CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingLR, LinearLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchmetrics.aggregation import RunningMean
from torchmetrics.text import Perplexity
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from dlojz_chrono import Chronometer      

if idr_torch.rank == 0:
    print(f">>> Training on {idr_torch.nnodes} nodes and {idr_torch.world_size} processes")

def parse_args() -> Namespace:
    parser = ArgumentParser()

    # Memory related arguments
    parser.add_argument('--bsz', "--batch-size", dest="batch_size", default=4, type=int, help='batch size per GPU')
    parser.add_argument('--seq-len', default=1024, type=int, help='sequence length of each sample per GPU')

    # SIGINT
    parser.add_argument('--test', default=False, action='store_true', help='Test 50 iterations')
    parser.add_argument('--test-nsteps', default='50', type=int, help='the number of steps in test mode')

    # JIT related arguments
    parser.add_argument("--compile", default=False, action=BooleanOptionalAction, help="whether or not to compile model")
    parser.add_argument("--compile-warmup-steps", default=10, type=int, help="number of steps to warm up compilation")

    # DataLoader related arguments
    parser.add_argument('--num-workers', default=4, type=int, help='num workers in dataloader')
    parser.add_argument('--persistent-workers', default=False, action=BooleanOptionalAction, help='activate persistent workers in dataloader')
    parser.add_argument('--pin-memory', default=True, action=BooleanOptionalAction, help='activate pin memory option in dataloader')
    parser.add_argument('--non-blocking', default=True, action=BooleanOptionalAction, help='activate asynchronuous GPU transfer')
    parser.add_argument('--prefetch-factor', default=3, type=int, help='prefectch factor in dataloader')
    parser.add_argument('--drop-last', default=False, action=BooleanOptionalAction, help='activate drop_last option in dataloader')

    # Training related arguments
    parser.add_argument("--lr-warmup-ratio", default=0.1, type=float, help="linear warmup of learning rate before cosine annealing")
    parser.add_argument("--lr", "--learning-rate", dest="learning_rate", type=float, default=1e-5, help="learning rate for adamw")
    parser.add_argument("--wd", "--weight-decay", dest="weight_decay", type=float, default=0.1, help="weight decay for adamw")

    # Other
    parser.add_argument("--nccl-profile", default=False, action=BooleanOptionalAction, help="whether or not to profile nccl communications. Clutters stdout heavily.")

    return parser.parse_args()


args = parse_args()
if args.nccl_profile:
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,COLL"

chrono = Chronometer(True, idr_torch.rank)

torch.set_float32_matmul_precision('high')

dist.init_process_group(
    backend="nccl",
    rank=idr_torch.rank,
    world_size=idr_torch.world_size,
)

DSDIR = Path(os.environ["DSDIR"])
model_path = DSDIR / "HuggingFace_Models" / "meta-llama" / "Llama-3.2-3B-Instruct"
dataset_path = DSDIR / "HuggingFace" / "hieunguyenminh" / "roleplay"

torch.cuda.set_device(idr_torch.local_rank)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#### Initialize the model and its tokenizer
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
num_parameters = sum(param.numel() for param in model.parameters())
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token_id = tokenizer.eos_token_id
####

#### Distribute the Model
model = FSDP(
    module=model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    use_orig_params=True,
    device_id=idr_torch.local_rank,
    auto_wrap_policy=functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LlamaDecoderLayer},
    ),
)
####

#### JIT
if args.compile:
    pass
####

if idr_torch.rank == 0:
    print(f"model: {model}")
    print(f"number of parameters: {num_parameters}")


#### Data Loading
def collate_fn(batch):
    tokenized = tokenizer(
        batch,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=args.seq_len + 1,
    )
    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask
    return input_ids[..., :-1], attention_mask[..., :-1], input_ids[..., 1:]

roleplay_dataset = datasets.load_from_disk(str(dataset_path))
train_dataset = roleplay_dataset["train"]["text"]

sampler = DistributedSampler(
    dataset=train_dataset,
    rank=idr_torch.rank,
    num_replicas=idr_torch.world_size,
    shuffle=True,
)

dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    collate_fn=collate_fn,
    pin_memory=args.pin_memory,
    drop_last=args.drop_last,
    persistent_workers=args.persistent_workers,
    prefetch_factor=args.prefetch_factor,
    sampler=sampler,
)
####


#### Training step
criterion = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = AdamW(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

if idr_torch.rank == 0:
    print(f'global batch size: {args.batch_size * idr_torch.world_size} - mini batch size: {args.batch_size}')
    print(f"DATALOADER {args.num_workers} {args.persistent_workers} {args.pin_memory} {args.non_blocking} {args.prefetch_factor} {args.drop_last} ")
    print(f"Optimizer: {optimizer}")

lr_warmup_iters = int(len(dataloader) * args.lr_warmup_ratio)  # * args.epochs
warmup_lr_scheduler = LinearLR(optimizer, start_factor=1e-9, end_factor=1, total_iters=lr_warmup_iters)
annealing_lr_scheduler = CosineAnnealingLR(optimizer, T_max=len(dataloader) - lr_warmup_iters, eta_min=0.)
lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_lr_scheduler, annealing_lr_scheduler], milestones=[lr_warmup_iters])

loss_metric = RunningMean(window=5).to(device)
perplexity = Perplexity(ignore_index=tokenizer.pad_token_id).to(device)
####

#### Compile warmup
for i, (input_ids, attention_mask, labels) in enumerate(dataloader, start=1):
    if not args.compile or i > args.compile_warmup_steps:
        break
    input_ids = input_ids.to(device, non_blocking=args.non_blocking)
    attention_mask = attention_mask.to(device, non_blocking=args.non_blocking)
    labels = labels.to(device, non_blocking=args.non_blocking)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids, attention_mask=attention_mask).logits
        bsz, seq_len, vocab_size = logits.shape
        loss = criterion(logits.view(bsz * seq_len, vocab_size), labels.view(bsz * seq_len))
    loss.backward()
####



#### Training loop
chrono.start()
chrono.dataload()
if idr_torch.rank == 0: chrono.tac_time(clear=True)
for i, (input_ids, attention_mask, labels) in enumerate(dataloader, start=1):
    if args.test and i > args.test_nsteps: break
    print(f'Train step {i} - rank {idr_torch.rank}')

    input_ids = input_ids.to(device, non_blocking=args.non_blocking)
    attention_mask = attention_mask.to(device, non_blocking=args.non_blocking)
    labels = labels.to(device, non_blocking=args.non_blocking)

    chrono.dataload()
    chrono.training()
    chrono.forward()

    optimizer.zero_grad()

    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits: torch.Tensor = model(input_ids, attention_mask=attention_mask).logits
        bsz, seq_len, vocab_size = logits.shape
        loss: torch.Tensor = criterion(logits.view(bsz * seq_len, vocab_size), labels.view(bsz * seq_len))
        loss /= idr_torch.world_size


    loss_metric.update(loss)
    perplexity.update(logits, labels)

    chrono.forward()
    chrono.backward()
    loss.backward()
    optimizer.step()
    lr_scheduler.step()
    chrono.backward()
    chrono.training()

    if i % 5 == 0:
        L = loss_metric.compute()
        perp = perplexity.compute()
        last_lr = lr_scheduler.get_last_lr()[0]
        if idr_torch.rank == 0:
            print(f"Step {i} / {args.test_nsteps if args.test else len(dataloader)} | Loss: {L.item():.3f} | Perplexity: {perp.item():.3f} | LR: {last_lr:0.3e} | Wall: {chrono.tac_time()}")

    chrono.dataload()
####

chrono.display()
dist.barrier()
if idr_torch.rank == 0:
    print(f">>> Number of batch per epoch: {len(dataloader)}")
    print(f'Max Memory Allocated {torch.cuda.max_memory_allocated()} Bytes')
else:
    print(f'MaxMemory for GPU:{idr_torch.rank} {torch.cuda.max_memory_allocated()} Bytes')

dist.barrier()
dist.destroy_process_group()
