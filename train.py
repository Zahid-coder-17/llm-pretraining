#!/usr/bin/env python3
"""
train.py — Pre-training script for GQA / GQE / HyperGQE.

IDENTICAL config for all 3 architectures. Only --arch changes.

Run:
    python train.py --arch gqa
    python train.py --arch gqe
    python train.py --arch hyper_gqe

Restartable: automatically resumes from latest checkpoint if --resume not given.
"""
import argparse, json, logging, math, os, random, subprocess, sys, time
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, str(Path(__file__).parent))
import config as C
from model   import build_model, PretrainConfig
from dataset import build_dataloader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ── Training hyperparameters ──────────────────────────────────────────────────
# These are FIXED and identical for all 3 architectures.

class TrainConfig:
    # Optimizer
    learning_rate       = 3e-4
    weight_decay        = 0.1
    beta1               = 0.9
    beta2               = 0.95
    grad_clip           = 1.0

    # Schedule
    warmup_steps        = 500
    # Total steps computed from token budget at runtime

    # Batch (RTX 3050 6GB safe)
    batch_size          = 2           # per-GPU micro batch (RTX 3050 6GB safe)
    grad_accum_steps    = 32          # effective batch = 2 * 32 = 64
    # effective tokens per step = 64 * 512 = 32,768

    # Training budget
    total_train_tokens  = 500_000_000

    # Checkpointing
    save_every_steps    = 100
    eval_every_steps    = 250
    log_every_steps     = 10

    # Mixed precision
    dtype               = "bfloat16"   # bfloat16 > float16 for training stability

    # Router loss coefficient (for GQE/HyperGQE)
    router_loss_coeff   = 0.01

    @property
    def total_steps(self) -> int:
        tokens_per_step = self.batch_size * self.grad_accum_steps * C.SEQ_LEN
        return math.ceil(self.total_train_tokens / tokens_per_step)

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum_steps



# ── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int):
    """Lock all sources of randomness for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Deterministic CUDA ops (slight perf cost, required for reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    # PyTorch >= 1.11 global determinism flag
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        pass
    log.info(f"Seed locked: {seed} (Python/NumPy/Torch/CUDA)")


def get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "no-git"


def save_run_config(arch: str, tc, mc, ckpt_dir: Path, seed: int = 42):
    """Save complete run configuration for reproducibility."""
    import platform, torch
    cfg = {
        "arch":            arch,
        "seed":            seed,
        "vocab_size":      mc.vocab_size,
        "hidden_size":     mc.hidden_size,
        "num_layers":      mc.num_hidden_layers,
        "num_heads":       mc.num_attention_heads,
        "num_kv_heads":    mc.num_key_value_heads,
        "intermediate":    mc.intermediate_size,
        "seq_len":         C.SEQ_LEN,
        "batch_size":      tc.batch_size,
        "grad_accum":      tc.grad_accum_steps,
        "effective_batch": tc.effective_batch_size,
        "learning_rate":   tc.learning_rate,
        "weight_decay":    tc.weight_decay,
        "beta1":           tc.beta1,
        "beta2":           tc.beta2,
        "grad_clip":       tc.grad_clip,
        "warmup_steps":    tc.warmup_steps,
        "total_steps":     tc.total_steps,
        "total_tokens":    tc.total_train_tokens,
        "dtype":           tc.dtype,
        "git_hash":        get_git_hash(),
        "python_version":  platform.python_version(),
        "torch_version":   torch.__version__,
        "cuda_version":    torch.version.cuda,
        "cudnn_version":   str(torch.backends.cudnn.version()),
        "platform":        platform.platform(),
        "data_dir":        str(C.DATA_DIR),
        "checkpoint_dir":  str(ckpt_dir),
        "run_started_at":  time.strftime("%Y-%m-%dT%H:%M:%S"),
        "datasets":        {k: {
                               "lang":          v["lang"],
                               "train_tokens":  v["train_tokens"],
                               "val_tokens":    v["val_tokens"],
                               "test_tokens":   v["test_tokens"],
                           } for k, v in C.DATASETS.items()},
    }
    out = ckpt_dir / "run_config.json"
    out.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    log.info(f"Run config saved: {out}")
    return cfg

def cosine_lr(step: int, warmup: int, total: int, lr: float) -> float:
    if step < warmup:
        return lr * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def set_lr(optimizer, lr: float):
    for g in optimizer.param_groups:
        g["lr"] = lr


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scaler, step: int, loss: float,
                    arch: str, ckpt_dir: Path):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step_{step:07d}.pt"
    torch.save({
        "step":           step,
        "arch":           arch,
        "model":          model.state_dict(),
        "optimizer":      optimizer.state_dict(),
        "scaler":         scaler.state_dict(),
        "loss":           loss,
    }, path)
    # Keep only last 3 checkpoints to save disk space
    all_ckpts = sorted(ckpt_dir.glob("step_*.pt"))
    for old in all_ckpts[:-3]:
        old.unlink(missing_ok=True)
    log.info(f"  Checkpoint saved: {path.name}")
    return path


def load_checkpoint(model, optimizer, scaler, ckpt_dir: Path):
    ckpts = sorted(ckpt_dir.glob("step_*.pt"))
    if not ckpts:
        return 0, float("inf")
    path = ckpts[-1]
    log.info(f"Resuming from {path}")
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    return ckpt["step"], ckpt["loss"]


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, val_loader, device, dtype, max_batches: int = 50) -> dict:
    model.eval()
    total_loss = 0.0
    total_tok  = 0
    batches    = 0

    for batch in val_loader:
        if batches >= max_batches:
            break
        x = batch.to(device)
        with autocast(dtype=dtype, enabled=dtype != torch.float32):
            out = model(x, labels=x)
        loss = out["loss"]
        total_loss += loss.item() * x.numel()
        total_tok  += x.numel()
        batches    += 1

    avg_loss   = total_loss / max(total_tok, 1)
    perplexity = math.exp(min(avg_loss, 20))   # cap at e^20 to avoid overflow
    model.train()
    return {"val_loss": avg_loss, "perplexity": perplexity, "batches": batches}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch",   required=True, choices=["gqa", "gqe", "hyper_gqe"])
    parser.add_argument("--seed",   type=int, default=42,
                        help="Training seed. Run 3 times: --seed 42, --seed 123, --seed 456")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint (auto-detected if omitted)")
    parser.add_argument("--subset", default=None,
                        help="Path to downsampling manifest JSON")
    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    set_seed(args.seed)

    tc  = TrainConfig()
    mc  = PretrainConfig()
    mc.gradient_checkpointing = True   # mandatory for RTX 3050

    dtype = torch.bfloat16 if tc.dtype == "bfloat16" else torch.float16
    amp_enabled = device.type == "cuda"

    ckpt_dir = Path("E:/pretrain_checkpoints") / f"{args.arch}_seed{args.seed}"
    log_file  = ckpt_dir / "train_log.jsonl"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(args.arch, tc, mc, ckpt_dir, seed=args.seed)

    # Log config
    log.info(f"Architecture:      {args.arch.upper()}")
    log.info(f"Total steps:       {tc.total_steps:,}")
    log.info(f"Effective batch:   {tc.effective_batch_size} sequences "
             f"({tc.effective_batch_size * C.SEQ_LEN:,} tokens/step)")
    log.info(f"Warmup steps:      {tc.warmup_steps}")
    log.info(f"Precision:         {tc.dtype}")

    # Build model
    log.info(f"Building {args.arch} model...")
    model = build_model(args.arch, mc).to(device)
    if amp_enabled:
        model = model.to(dtype)

    n_params = model.num_parameters()
    log.info(f"Parameters: {n_params/1e6:.2f}M")

    # Optimizer (separate weight decay groups)
    decay_params    = [p for n, p in model.named_parameters()
                       if p.requires_grad and p.ndim >= 2]
    no_decay_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and p.ndim < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay_params,    "weight_decay": tc.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=tc.learning_rate, betas=(tc.beta1, tc.beta2), fused=device.type == "cuda")

    scaler = GradScaler(enabled=amp_enabled and tc.dtype == "float16")

    # Resume
    start_step = 0
    best_val_loss = float("inf")
    if (ckpt_dir / "step_0000001.pt").parent.exists():
        start_step, _ = load_checkpoint(model, optimizer, scaler, ckpt_dir)

    # DataLoaders
    subset_path = Path(args.subset) if args.subset else None
    train_loader = build_dataloader(
        "train", lang="both", batch_size=tc.batch_size,
        num_workers=1, shuffle=True, subset_manifest=subset_path
    )
    val_loader = build_dataloader(
        "val", lang="both", batch_size=tc.batch_size,
        num_workers=1, shuffle=False
    )

    # Training loop
    model.train()
    optimizer.zero_grad()

    step         = start_step
    accum_step   = 0
    total_loss   = 0.0
    t_start      = time.time()
    tokens_seen  = step * tc.effective_batch_size * C.SEQ_LEN

    log.info(f"Starting training from step {step}/{tc.total_steps}")

    train_iter = iter(train_loader)

    while step < tc.total_steps:
        # Fetch batch (restart iterator if exhausted)
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        x = batch.to(device)

        # Forward
        with autocast(dtype=dtype, enabled=amp_enabled):
            out  = model(x, labels=x)
            loss = out["loss"] / tc.grad_accum_steps

        # NaN guard — skip bad batch instead of crashing
        if torch.isnan(loss) or torch.isinf(loss):
            log.warning(f"NaN/Inf loss at step {step}, skipping batch")
            optimizer.zero_grad()
            accum_step = 0
            continue

        # Backward
        scaler.scale(loss).backward()
        total_loss  += loss.item() * tc.grad_accum_steps
        accum_step  += 1

        if accum_step < tc.grad_accum_steps:
            continue

        # Optimizer step
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)

        lr = cosine_lr(step, tc.warmup_steps, tc.total_steps, tc.learning_rate)
        set_lr(optimizer, lr)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        step        += 1
        accum_step   = 0
        tokens_seen += tc.effective_batch_size * C.SEQ_LEN

        # Logging
        if step % tc.log_every_steps == 0:
            elapsed = time.time() - t_start
            tok_s   = tokens_seen / max(elapsed, 1)
            log.info(f"step {step:6d}/{tc.total_steps} | "
                     f"loss {total_loss/tc.log_every_steps:.4f} | "
                     f"lr {lr:.2e} | {tok_s/1e3:.1f}K tok/s | "
                     f"{tokens_seen/1e6:.0f}M tok seen")
            with open(log_file, "a") as f:
                f.write(json.dumps({
                    "step": step, "loss": round(total_loss/tc.log_every_steps, 5),
                    "lr": lr, "tokens_seen": tokens_seen,
                }) + "\n")
            total_loss = 0.0

        # Evaluation
        if step % tc.eval_every_steps == 0:
            val_metrics = evaluate(model, val_loader, device, dtype)
            log.info(f"  [EVAL] val_loss={val_metrics['val_loss']:.4f}  "
                     f"ppl={val_metrics['perplexity']:.2f}")
            with open(log_file, "a") as f:
                f.write(json.dumps({"step": step, "eval": val_metrics}) + "\n")
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                best_path = ckpt_dir / "best.pt"
                torch.save({"step": step, "arch": args.arch,
                            "model": model.state_dict(),
                            "val_loss": best_val_loss}, best_path)
                log.info(f"  [BEST] Saved to {best_path}")

        # Checkpoint
        if step % tc.save_every_steps == 0:
            save_checkpoint(model, optimizer, scaler, step,
                            total_loss, args.arch, ckpt_dir)

    # Final checkpoint and eval
    save_checkpoint(model, optimizer, scaler, step, total_loss, args.arch, ckpt_dir)
    final_metrics = evaluate(model, val_loader, device, dtype, max_batches=200)
    log.info(f"Training complete. Final val_loss={final_metrics['val_loss']:.4f}  "
             f"ppl={final_metrics['perplexity']:.2f}")


if __name__ == "__main__":
    main()






