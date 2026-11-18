import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, "/workspace/Maria")
from model import MariaConfig, build_model

device = torch.device("cuda:0")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def chunked_cross_entropy(hidden, lm_head_weight, labels, chunk_size=2048):
    total_tokens = hidden.shape[0]
    total_loss = 0.0
    for i in range(0, total_tokens, chunk_size):
        end = min(i + chunk_size, total_tokens)
        logits_chunk = F.linear(hidden[i:end], lm_head_weight)
        loss_chunk = F.cross_entropy(logits_chunk, labels[i:end], reduction="sum")
        total_loss += loss_chunk
    return total_loss / total_tokens

print(f"Testing Aggressive Throughput Pipeline on {torch.cuda.get_device_name(0)}...")

cfg = MariaConfig()
cfg.gradient_checkpointing = False  # NO GRADIENT CHECKPOINTING!

model = build_model(cfg).to(device, dtype=torch.bfloat16)

# High-density batch: micro-batch 4, seq_len 8192 = 32,768 tokens per GPU
bsz = 4
slen = 8192
total_tok = bsz * slen

x = torch.randint(0, cfg.vocab_size, (bsz, slen), device=device)
y = torch.randint(0, cfg.vocab_size, (bsz, slen), device=device)

# Warmup
for _ in range(2):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        h = model.embed(x)
        for b in model.blocks:
            h = b(h)
        h = model.norm(h)
        shift_h = h[:, :-1, :].reshape(-1, cfg.hidden_size)
        shift_y = y[:, 1:].reshape(-1)
        loss = chunked_cross_entropy(shift_h, model.lm_head.weight, shift_y, chunk_size=4096)
        loss.backward()
        model.zero_grad(set_to_none=True)
torch.cuda.synchronize()

# Benchmark 10 micro-steps
t0 = time.perf_counter()
N = 10
for _ in range(N):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        h = model.embed(x)
        for b in model.blocks:
            h = b(h)
        h = model.norm(h)
        shift_h = h[:, :-1, :].reshape(-1, cfg.hidden_size)
        shift_y = y[:, 1:].reshape(-1)
        loss = chunked_cross_entropy(shift_h, model.lm_head.weight, shift_y, chunk_size=4096)
        loss.backward()
        model.zero_grad(set_to_none=True)
torch.cuda.synchronize()
t1 = time.perf_counter()

dt = (t1 - t0) / N
tok_s_1gpu = total_tok / dt
mem = torch.cuda.max_memory_allocated() / (1024**3)

print("=" * 65)
print(f"Aggressive Micro-Step Time: {dt*1000:.1f} ms | VRAM: {mem:.1f} GB")
print(f"Single GPU Throughput:      {tok_s_1gpu:,.0f} tok/s")
print(f"4x H100 Cluster Throughput: {tok_s_1gpu * 4:,.0f} tok/s")
time_100b_hrs = 100e9 / (tok_s_1gpu * 4 * 3600)
cost_100b = time_100b_hrs * 15.96
print(f"Time for 100B Tokens:       {time_100b_hrs:.1f} Hours ({time_100b_hrs/24:.2f} Days)")
print(f"Total Cost for 100B Tokens: ${cost_100b:,.2f} USD")
print("=" * 65)
