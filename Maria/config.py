#!/usr/bin/env python3
"""
config.py - Master Configuration for Maria 1B HyperGQE Pretraining (50B Budget).
"""
from pathlib import Path
import os

PROJECT_NAME        = "Maria"
MODEL_NAME          = "Maria-1B-HyperGQE"
ARCHITECTURE        = "hyper_gqe"

NUM_GPUS            = 4
GPU_TYPE            = "NVIDIA H100 SXM5 80GB"
INTERCONNECT        = "NVLink 900 GB/s"

# ── 50B Token Budget & Steps ─────────────────────────────────────────────────
TOTAL_TOKEN_BUDGET  = 50_000_000_000   # 50 Billion Tokens
SEQ_LEN             = 8192             # 8k Context Window

MICRO_BATCH_SIZE    = 2                # 2 * 8192 = 16,384 tokens per GPU
GRAD_ACCUM_STEPS    = 32               # 4 GPUs * 2 * 32 * 8192 = 2,097,152 tokens
GLOBAL_BATCH_TOKENS = NUM_GPUS * MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS * SEQ_LEN # 2,097,152
TOTAL_STEPS         = 23_842           # ceil(50B / 2,097,152)

# Budget Auto-Stop Guard (Graceful shutdown when spend hits $1,400)
MAX_BUDGET_USD      = 1400.0           # Stop and save final checkpoint at $1,400
HOURLY_COST_USD     = 15.96            # RunPod 4x H100 On-Demand rate

# ── Optimizer & Schedule ─────────────────────────────────────────────────────
LEARNING_RATE       = 3.0e-4
MIN_LEARNING_RATE   = 3.0e-5
WARMUP_STEPS        = 800              # ~1.68B tokens warmup
WEIGHT_DECAY        = 0.1
BETA1               = 0.9
BETA2               = 0.95
GRAD_CLIP           = 1.0
DTYPE               = "bfloat16"
ROUTER_LOSS_COEFF   = 0.01

SAVE_EVERY_STEPS    = 250
EVAL_EVERY_STEPS    = 500
LOG_EVERY_STEPS     = 5

POD_DATA_DIR        = Path("/workspace/final_115B")
POD_CHECKPOINT_DIR  = Path("/workspace/checkpoints/maria_1b")
LOCAL_DATA_DIR      = Path("E:/final_115B") if Path("E:/final_115B").exists() else Path("E:/pretrain_data")
LOCAL_CHECKPOINT_DIR= Path("E:/Maria/checkpoints")

def get_data_dir() -> Path:
    if POD_DATA_DIR.exists():
        return POD_DATA_DIR
    return LOCAL_DATA_DIR

def get_checkpoint_dir() -> Path:
    if Path("/workspace").exists():
        d = POD_CHECKPOINT_DIR
    else:
        d = LOCAL_CHECKPOINT_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d
