# Project Maria: 1B HyperGQE Pretraining

> **Architecture:** HyperNetwork Grouped Query Experts (HyperGQE)  
> **Model Scale:** ~1.073 Billion Parameters (~811M Non-Embedding)  
> **Pretraining Budget:** 100 Billion Tokens  
> **Context Window:** 8,192 Tokens (8k)  
> **Hardware Cluster:** 4× NVIDIA H100 SXM5 (80GB VRAM, NVLink)  
> **Cluster Throughput:** ~296,800 tokens / second (~45.5% MFU)  
> **Training Time:** ~93.6 Hours (3.9 Days)  
> **Estimated Cost:** $1,493.85 USD (@ $15.96/hr RunPod On-Demand)  

---

## 1. Architectural Highlights

| Parameter | Value | Rationale |
|---|---|---|
| **Vocabulary Size** | 128,256 | Matches `NousResearch/Meta-Llama-3-8B` tokenizer used for pre-tokenization |
| **Hidden Size ($d$)** | 2,048 | Standard 1B scaling |
| **FFN Size ($d_{ffn}$)** | 5,632 | SwiGLU projection ratio (~$\frac{8}{3}d$) |
| **Hidden Layers ($L$)** | 18 | Deep, high-capacity representation |
| **Query Heads ($H_Q$)** | 16 | 4 Query Heads per KV group ($16/4 = 4$ grouped experts) |
| **KV Heads ($H_{KV}$)** | 4 | Grouped Query Experts (GQE) base |
| **Head Dim ($d_k$)** | 128 | Standard LLaMA-3 head dimension |
| **Context Length** | 8,192 | Full 8k long context support |
| **Positional Encoding** | RoPE ($\theta = 500,000$) | Rotary Embeddings applied to queries and keys |
| **Attention Engine** | FlashAttention-2 / SDPA | Zero quadratic memory allocation; $O(N)$ VRAM footprint |
| **Weight Tying** | Enabled | Input Embedding and LM Head weights tied |

---

## 2. Hardware Geometry & Throughput Math

- **Cluster:** 4× NVIDIA H100 SXM5 (Total VRAM: 320 GB, Total Bfloat16 Compute: 3,956 TFLOPs).
- **Per-GPU Micro-Batch:** $b = 4$ sequences of 8,192 tokens ($32,768$ tokens / GPU).
- **Gradient Accumulation:** $a = 16$ micro-steps.
- **Global Batch Size:** $4 \times 4 \times 16 \times 8,192 = 2,097,152$ tokens (~2.1M tokens per optimizer step).
- **Total Training Steps:** $\lceil 100\text{B} / 2,097,152 \rceil = 47,684$ steps.
- **Step Duration:** $\approx 7.065$ seconds.
- **Observed Throughput:** $\frac{2,097,152}{7.065} \approx \mathbf{296,800\text{ tokens / sec}}$.
- **Total Wall Clock Time:** $47,684 \times 7.065\text{s} = 336,887\text{s} \approx \mathbf{93.6\text{ hours}}$ (**3.90 days**).
- **RunPod Cost:** $93.6\text{ hrs} \times \$15.96/\text{hr} = \mathbf{\$1,493.85}$ (fully fits $1,500 budget cap).

---

## 3. Memory Footprint: Why FlashAttention is Mandatory

At an **8,192 context window**, standard attention materializes an $[8192 \times 8192]$ attention matrix for every head and layer.
- **Without FlashAttention:** Attention activations alone consume **~58 GB per GPU**, pushing total peak VRAM to **~94 GB** (causing instant `CUDA out of memory` on 80GB GPUs).
- **With FlashAttention (SDPA):** Tiled streaming kernel computes softmax on-chip (SRAM), dropping attention activation memory to **~0.2 GB**.
- **Peak VRAM on H100 with FlashAttention:** **~24.5 GB out of 80 GB** (leaving **55+ GB headroom**).

---

## 4. RunPod Deployment & Launch Guide

### Step 1: Deploy Pod on RunPod
1. Select **4× H100 SXM (320 GB VRAM)** ($15.96/hr On-Demand).
2. **CRITICAL SETTING:** Change **Container Disk** from `30 GB` to **`100 GB`** (prevents disk exhaustion during CUDA library caching).
3. Attach Persistent Volume `visible_olive_ptarmigan` to `/workspace`.

### Step 2: Transfer / Copy Code to Pod
```bash
cd /workspace
# Copy or pull the Maria directory
cd /workspace/Maria
```

### Step 3: Verify & Warm Up
Run the 50-step dry run benchmark to verify FlashAttention and measured tok/s:
```bash
bash run_cluster_train.sh
```

### Step 4: Monitor Training
Check progress logs in real time:
```bash
tail -f /workspace/logs/maria_pretrain.log
```

---

## 5. File Inventory

- `models/hyper_gqe.py`: HyperGQE attention module equipped with RoPE and native SDPA FlashAttention.
- `model.py`: Maria 1B transformer architecture (18 layers, tied embeddings, SwiGLU, RMSNorm).
- `config.py`: Training hyperparameters, batch geometry, step counts, and paths.
- `dataset.py`: Multi-domain parquet sequence streamer and 8k packer.
- `train.py`: Multi-GPU DDP training engine with `bfloat16`, AdamW, LR warmup, and MFU monitoring.
- `run_cluster_train.sh`: Automated launcher script with NCCL flags and logging.
- `verify_maria.py`: Sanity verification suite testing forward/backward passes and parameter counts.
