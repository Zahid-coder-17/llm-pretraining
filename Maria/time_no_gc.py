import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, "/workspace/Maria")
from model import MariaConfig, build_model

device = torch.device("cuda:0")

def chunked_cross_entropy(hidden, lm_head_weight, labels, chunk_size=2048):
    total_tokens = hidden.shape[0]
    total_loss = 0.0
    for i in range(0, total_tokens, chunk_size):
        end = min(i + chunk_size, total_tokens)
        logits_chunk = F.linear(hidden[i:end], lm_head_weight)
        loss_chunk = F.cross_entropy(logits_chunk, labels[i:end], reduction="sum")
        total_loss += loss_chunk
    return total_loss / total_tokens

cfg = MariaConfig()
cfg.gradient_checkpointing = False
model = build_model(cfg).to(device, dtype=torch.bfloat16)

bsz = 2
slen = 8192
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
        loss = chunked_cross_entropy(shift_h, model.lm_head.weight, shift_y, chunk_size=2048)
        loss.backward()
        model.zero_grad(set_to_none=True)
torch.cuda.synchronize()

# Benchmark 10 steps
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
        loss = chunked_cross_entropy(shift_h, model.lm_head.weight, shift_y, chunk_size=2048)
        loss.backward()
        model.zero_grad(set_to_none=True)
torch.cuda.synchronize()
t1 = time.perf_counter()

dt = (t1 - t0) / N
tok_s = (bsz * slen) / dt
print(f"NO-GC Step Time: {dt*1000:6.1f} ms | Single GPU: {tok_s:7,.0f} tok/s | 4-GPU: {tok_s*4:7,.0f} tok/s")
