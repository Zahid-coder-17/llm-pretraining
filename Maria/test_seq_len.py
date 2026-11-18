import time
import torch
import sys
sys.path.insert(0, "/workspace/Maria")
from model import MariaConfig, build_model

device = torch.device("cuda:0")

for slen in [8192, 4096]:
    cfg = MariaConfig()
    cfg.gradient_checkpointing = True
    model = build_model(cfg).to(device, dtype=torch.bfloat16)
    
    bsz = 4 if slen == 4096 else 2
    x = torch.randint(0, cfg.vocab_size, (bsz, slen), device=device)
    y = torch.randint(0, cfg.vocab_size, (bsz, slen), device=device)
    
    # Warmup
    for _ in range(2):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(x, labels=y)
            out["loss"].backward()
            model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    
    t0 = time.perf_counter()
    N = 6
    for _ in range(N):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(x, labels=y)
            out["loss"].backward()
            model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    
    dt = (t1 - t0) / N
    mem = torch.cuda.max_memory_allocated() / (1024**3)
    tok_s = (bsz * slen) / dt
    print(f"SeqLen: {slen} | Batch: {bsz} | Time: {dt*1000:5.1f} ms | VRAM: {mem:4.1f} GB | Single GPU: {tok_s:7,.0f} tok/s | 4-GPU: {tok_s*4:7,.0f} tok/s")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
