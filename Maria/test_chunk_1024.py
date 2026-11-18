import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, "/workspace/Maria")
from model import MariaConfig, build_model

device = torch.device("cuda:0")

def chunked_cross_entropy(hidden, lm_head_weight, labels, chunk_size=1024):
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

# Try with bsz=3 (24,576 tokens) or bsz=2 (16,384 tokens)
for bsz in [2, 3]:
    slen = 8192
    total_tok = bsz * slen
    x = torch.randint(0, cfg.vocab_size, (bsz, slen), device=device)
    y = torch.randint(0, cfg.vocab_size, (bsz, slen), device=device)
    try:
        # Warmup
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            h = model.embed(x)
            for b in model.blocks:
                h = b(h)
            h = model.norm(h)
            shift_h = h[:, :-1, :].reshape(-1, cfg.hidden_size)
            shift_y = y[:, 1:].reshape(-1)
            loss = chunked_cross_entropy(shift_h, model.lm_head.weight, shift_y, chunk_size=1024)
            loss.backward()
            model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        
        t0 = time.perf_counter()
        N = 5
        for _ in range(N):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                h = model.embed(x)
                for b in model.blocks:
                    h = b(h)
                h = model.norm(h)
                shift_h = h[:, :-1, :].reshape(-1, cfg.hidden_size)
                shift_y = y[:, 1:].reshape(-1)
                loss = chunked_cross_entropy(shift_h, model.lm_head.weight, shift_y, chunk_size=1024)
                loss.backward()
                model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / N
        mem = torch.cuda.max_memory_allocated() / (1024**3)
        tok_s = total_tok / dt
        print(f"Batch {bsz}: dt={dt*1000:.1f}ms | VRAM={mem:.1f}GB | Single GPU={tok_s:,.0f} tok/s | 4-GPU={tok_s*4:,.0f} tok/s")
    except torch.OutOfMemoryError as e:
        print(f"Batch {bsz} OOM: {e}")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
