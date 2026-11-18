import time
import torch
import sys
sys.path.insert(0, "/workspace/Maria")
from model import MariaConfig, build_model

device = torch.device("cuda:0")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print(f"Testing torch.compile on {torch.cuda.get_device_name(0)}...")

cfg = MariaConfig()
cfg.gradient_checkpointing = True
model = build_model(cfg).to(device, dtype=torch.bfloat16)

# Compile model
print("Compiling model with torch.compile...")
compiled_model = torch.compile(model)

x = torch.randint(0, cfg.vocab_size, (2, 8192), device=device)
y = torch.randint(0, cfg.vocab_size, (2, 8192), device=device)

# Warmup / Compilation steps
print("Running warm-up & kernel compilation...")
for i in range(3):
    t0 = time.time()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = compiled_model(x, labels=y)
        out["loss"].backward()
        compiled_model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    print(f"Warm-up {i+1} done in {time.time()-t0:.2f}s")

# Timed run
print("Timing compiled model (10 steps)...")
t0 = time.perf_counter()
N = 10
for _ in range(N):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = compiled_model(x, labels=y)
        out["loss"].backward()
        compiled_model.zero_grad(set_to_none=True)
torch.cuda.synchronize()
t1 = time.perf_counter()

dt = (t1 - t0) / N
tok_per_sec = (2 * 8192) / dt
print(f"Compiled Step Time: {dt*1000:6.1f} ms | Single GPU: {tok_per_sec:7,.0f} tok/s | 4-GPU Projected: {tok_per_sec*4:7,.0f} tok/s")
