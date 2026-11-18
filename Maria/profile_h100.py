import time
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, "/workspace/Maria")
from model import MariaConfig, build_model

device = torch.device("cuda:0")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print(f"Profiling on {torch.cuda.get_device_name(0)}...")

for gc in [True, False]:
    cfg = MariaConfig()
    cfg.gradient_checkpointing = gc
    model = build_model(cfg).to(device, dtype=torch.bfloat16)
    
    # Batch size 2, seq_len 8192
    x = torch.randint(0, cfg.vocab_size, (2, 8192), device=device)
    y = torch.randint(0, cfg.vocab_size, (2, 8192), device=device)
    
    # Warmup
    for _ in range(3):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(x, labels=y)
            out["loss"].backward()
            model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    
    # Timed run
    t0 = time.perf_counter()
    N = 10
    for _ in range(N):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(x, labels=y)
            out["loss"].backward()
            model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    
    dt = (t1 - t0) / N
    mem = torch.cuda.max_memory_allocated() / (1024**3)
    # 2 * 8192 = 16,384 tokens per micro-batch
    tok_per_sec = (2 * 8192) / dt
    print(f"Gradient Checkpointing = {gc:<5} | Step Time: {dt*1000:6.1f} ms | VRAM: {mem:4.1f} GB | Single GPU Throughput: {tok_per_sec:7,.0f} tok/s | 4-GPU Projected: {tok_per_sec*4:7,.0f} tok/s")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()



