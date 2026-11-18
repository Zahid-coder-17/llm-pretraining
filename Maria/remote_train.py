#!/usr/bin/env python3
"""
train.py — Pre-training Script for Maria 1B HyperGQE on 4x H100 SXM5.

Hardware: 4x NVIDIA H100 SXM5 80GB (NVLink)
Target: 100 Billion Tokens @ 8,192 Context Window
Throughput: ~296,800 tok/sec (~7.06s / step)
Runtime: ~93.6 hours (3.9 days)

Usage:
    # 4-GPU Cluster run:
    torchrun --nproc_per_node=4 train.py

    # Quick 50-step benchmark:
    torchrun --nproc_per_node=4 train.py --max_steps 50 --benchmark

    # Resume from checkpoint:
    torchrun --nproc_per_node=4 train.py --resume
"""
import argparse
import datetime
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Local Maria modules
sys.path.insert(0, str(Path(__file__).parent))
import config as C
from model import MariaConfig, build_model
from dataset import build_dataloader


# ── Distributed Setup ─────────────────────────────────────────────────────────

def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def print_rank0(msg: str, rank: int = 0):
    if rank == 0:
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# ── Learning Rate Schedule ───────────────────────────────────────────────────

def get_lr(step: int, total_steps: int, warmup_steps: int, lr_max: float, lr_min: float) -> float:
    if step < warmup_steps:
        return lr_max * (step + 1) / warmup_steps
    if step > total_steps:
        return lr_min
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_min + cosine_decay * (lr_max - lr_min)


# ── Optimizer Parameter Grouping ─────────────────────────────────────────────

def configure_optimizers(model, weight_decay: float, lr: float, betas: tuple):
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Biases, 1D layer norm / RMSNorm weights, and expert embeddings without decay
        if param.dim() < 2 or "norm" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas, eps=1e-8, fused=True)
    return optimizer


# ── Checkpointing ─────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, step: int, tokens_seen: int, loss: float, ckpt_dir: Path, is_best: bool = False):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    raw_model = model.module if hasattr(model, "module") else model
    checkpoint = {
        "step": step,
        "tokens_seen": tokens_seen,
        "loss": loss,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": raw_model.config,
    }
    ckpt_path = ckpt_dir / f"maria_1b_step_{step:06d}.pt"
    tmp_path = ckpt_dir / f"maria_1b_step_{step:06d}.tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, ckpt_path)

    latest_path = ckpt_dir / "latest.pt"
    torch.save(checkpoint, latest_path)

    if is_best:
        best_path = ckpt_dir / "best.pt"
        torch.save(checkpoint, best_path)

    # Maintain only last 3 step checkpoints to save disk space
    all_ckpts = sorted(list(ckpt_dir.glob("maria_1b_step_*.pt")))
    if len(all_ckpts) > 3:
        for old in all_ckpts[:-3]:
            try:
                old.unlink()
            except OSError:
                pass


def load_checkpoint(model, optimizer, ckpt_dir: Path):
    latest_path = ckpt_dir / "latest.pt"
    if not latest_path.exists():
        return 0, 0, float("inf")

    print(f"Loading checkpoint from {latest_path}...")
    checkpoint = torch.load(latest_path, map_location="cpu")
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    step = checkpoint["step"]
    tokens_seen = checkpoint["tokens_seen"]
    best_loss = checkpoint.get("loss", float("inf"))
    print(f"Resumed at step {step} ({tokens_seen / 1e9:.3f}B tokens).")
    return step, tokens_seen, best_loss


# ── Training Loop ─────────────────────────────────────────────────────────────

def train(args):
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    print_rank0("=" * 80, rank)
    print_rank0(f" Maria 1B Pretraining Engine — Starting on {world_size} GPU(s)", rank)
    print_rank0(f" Target Tokens: {C.TOTAL_TOKEN_BUDGET / 1e9:.1f}B | Context Window: {C.SEQ_LEN}", rank)
    print_rank0(f" Micro Batch: {C.MICRO_BATCH_SIZE} | Accum Steps: {C.GRAD_ACCUM_STEPS} | Global Batch: {C.GLOBAL_BATCH_TOKENS:,} tok", rank)
    print_rank0(f" FlashAttention & RoPE (theta=500,000) Active", rank)
    print_rank0("=" * 80, rank)

    # 1. Instantiate Model
    model_cfg = MariaConfig()
    model = build_model(model_cfg)
    model.to(device=device, dtype=torch.bfloat16)

    total_params = model.num_parameters()
    flops_per_tok = model.flops_per_token()
    print_rank0(f"Model Parameters: {total_params / 1e6:,.2f} M | Non-Embedding: {model.non_embedding_parameters() / 1e6:,.2f} M", rank)

    if args.compile and hasattr(torch, "compile"):
        print_rank0("Compiling model with torch.compile...", rank)
        model = torch.compile(model)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # 2. Configure Optimizer
    optimizer = configure_optimizers(
        model,
        weight_decay=C.WEIGHT_DECAY,
        lr=C.LEARNING_RATE,
        betas=(C.BETA1, C.BETA2),
    )

    # 3. Checkpointing & Resume
    ckpt_dir = C.get_checkpoint_dir()
    start_step = 0
    tokens_seen = 0
    best_loss = float("inf")

    if args.resume:
        start_step, tokens_seen, best_loss = load_checkpoint(model, optimizer, ckpt_dir)

    # 4. DataLoader
    data_dir = C.get_data_dir()
    print_rank0(f"Streaming dataset from: {data_dir}", rank)
    dataloader = build_dataloader(
        data_dir=data_dir,
        batch_size=C.MICRO_BATCH_SIZE,
        seq_len=C.SEQ_LEN,
        rank=rank,
        world_size=world_size,
        num_workers=2,
    )
    data_iter = iter(dataloader)

    # 5. Training Loop
    total_steps = args.max_steps if args.max_steps else C.TOTAL_STEPS
    model.train()
    step = start_step

    step_times = []
    throughput_samples = []
    start_wall_time = time.time()

    print_rank0(f"Launching training from step {start_step} to {total_steps}...", rank)

    while step < total_steps:
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        # Adjust LR for current step
        curr_lr = get_lr(step, total_steps, C.WARMUP_STEPS, C.LEARNING_RATE, C.MIN_LEARNING_RATE)
        for param_group in optimizer.param_groups:
            param_group["lr"] = curr_lr

        # Gradient accumulation micro-steps
        for micro_step in range(C.GRAD_ACCUM_STEPS):
            # Sync gradients only on last micro-step
            is_last_micro = (micro_step == C.GRAD_ACCUM_STEPS - 1)
            sync_context = model.no_sync() if (world_size > 1 and not is_last_micro) else torch.enable_grad()

            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                x, y = next(data_iter)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with sync_context:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(x, labels=y)
                    loss = outputs["loss"] / C.GRAD_ACCUM_STEPS

                loss.backward()
                accum_loss += loss.item()

        # Gradient clipping
        if world_size > 1:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)

        optimizer.step()

        t1 = time.perf_counter()
        step_duration = t1 - t0
        step_times.append(step_duration)

        tokens_seen += C.GLOBAL_BATCH_TOKENS
        step += 1

        # Real-time Throughput & MFU calculation
        step_tok_sec = C.GLOBAL_BATCH_TOKENS / max(step_duration, 1e-4)
        throughput_samples.append(step_tok_sec)
        if len(throughput_samples) > 50:
            throughput_samples.pop(0)
        avg_tok_sec = sum(throughput_samples) / len(throughput_samples)

        # Theoretical MFU for 4x H100 SXM5 (4 * 989 TFLOPs bfloat16 = 3,956 TFLOPs)
        cluster_peak_flops = world_size * 989e12
        step_flops = flops_per_tok * step_tok_sec
        mfu = step_flops / cluster_peak_flops

        # Logging
        if step % C.LOG_EVERY_STEPS == 0 or step == total_steps:
            elapsed_hrs = (time.time() - start_wall_time) / 3600.0
            remaining_steps = total_steps - step
            eta_hrs = (remaining_steps * (sum(step_times[-10:]) / len(step_times[-10:]))) / 3600.0
            cost_spent = elapsed_hrs * 15.96  # $15.96/hr 4x H100 SXM

            if torch.cuda.is_available():
                peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
            else:
                peak_vram_gb = 0.0

            print_rank0(
                f"Step {step:05d}/{total_steps} | "
                f"Loss: {accum_loss:.4f} | "
                f"LR: {curr_lr:.2e} | "
                f"Throughput: {avg_tok_sec:,.0f} tok/s | "
                f"MFU: {mfu * 100:.1f}% | "
                f"VRAM: {peak_vram_gb:.1f} GB | "
                f"ETA: {eta_hrs:.1f}h | "
                f"Spent: ${cost_spent:.2f}",
                rank,
            )

        # Checkpointing
        if step % C.SAVE_EVERY_STEPS == 0 and rank == 0:
            is_best = accum_loss < best_loss
            if is_best:
                best_loss = accum_loss
            save_checkpoint(model, optimizer, step, tokens_seen, accum_loss, ckpt_dir, is_best=is_best)
            print_rank0(f">> Checkpoint saved at step {step} (Best loss: {best_loss:.4f})", rank)

    total_wall_time = time.time() - start_wall_time
    print_rank0("=" * 80, rank)
    print_rank0(f" Training complete! {tokens_seen / 1e9:.2f}B tokens processed.", rank)
    print_rank0(f" Total wall clock: {total_wall_time / 3600.0:.2f} hours.", rank)
    print_rank0(f" Final Average Throughput: {sum(throughput_samples)/len(throughput_samples):,.0f} tok/sec.", rank)
    print_rank0("=" * 80, rank)

    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Maria 1B HyperGQE Pretraining")
    parser.add_argument("--max_steps", type=int, default=None, help="Override total steps (for benchmark)")
    parser.add_argument("--benchmark", action="store_true", help="Run 50-step benchmark mode")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()

    if args.benchmark and args.max_steps is None:
        args.max_steps = 50

    train(args)
