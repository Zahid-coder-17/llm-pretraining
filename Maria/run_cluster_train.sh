#!/usr/bin/env bash
# ==============================================================================
# run_cluster_train.sh — Maria 1B HyperGQE Cluster Launch Script
# Target Hardware: 4x NVIDIA H100 SXM5 80GB on RunPod
# ==============================================================================
set -e

echo "======================================================================"
echo "          MARIA 1B HYPERGQE PRETRAINING CLUSTER LAUNCHER"
echo "======================================================================"
date

# 1. Hardware & System Verification
echo ">>> Checking GPU Setup..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# 2. Check Persistent Volume Mount
DATA_DIR="/workspace/final_115B"
echo ">>> Checking Dataset Volume: $DATA_DIR"
if [ -d "$DATA_DIR" ]; then
    NUM_FILES=$(find "$DATA_DIR" -name "*.parquet" | wc -l)
    echo "    Found $NUM_FILES Parquet files on persistent volume."
else
    echo "    WARNING: /workspace/final_115B not found. Fallback to local data."
fi

# 3. Create Logging & Checkpoint Directories
mkdir -p /workspace/checkpoints/maria_1b
mkdir -p /workspace/logs

# 4. Set High-Performance Environment Flags
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=3
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# 5. Execute 50-Step Warmup / Verification Benchmark
echo ">>> Running 50-step warm-up benchmark to verify FlashAttention & throughput..."
torchrun --nproc_per_node=4 train.py --benchmark 2>&1 | tee /workspace/logs/maria_benchmark.log

echo ">>> Benchmark complete. Reviewing peak VRAM and tok/sec above."
echo ">>> Launching Full 100B Token Pretraining..."

# 6. Execute Full Training
torchrun --nproc_per_node=4 train.py 2>&1 | tee -a /workspace/logs/maria_pretrain.log

echo ">>> Training run finished successfully at $(date)."
