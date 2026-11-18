import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, "/workspace/Maria")
from model import MariaConfig, build_model

device = torch.device("cuda:0")

def chunked_cross_entropy(hidden, lm_head_weight, labels, chunk_size=2048):
    # hidden: (total_tokens, hidden_size)
    # labels: (total_tokens,)
    total_tokens = hidden.shape[0]
    total_loss = 0.0
    for i in range(0, total_tokens, chunk_size):
        end = min(i + chunk_size, total_tokens)
        h_chunk = hidden[i:end]
        l_chunk = labels[i:end]
        logits_chunk = F.linear(h_chunk, lm_head_weight)
        loss_chunk = F.cross_entropy(logits_chunk, l_chunk, reduction="sum")
        total_loss += loss_chunk
    return total_loss / total_tokens

print(f"Testing Chunked Cross-Entropy on {torch.cuda.get_device_name(0)}...")

cfg = MariaConfig()
cfg.gradient_checkpointing = False  # NO GRADIENT CHECKPOINTING!

model = build_model(cfg).to(device, dtype=torch.bfloat16)

# Test if forward + backward fits in memory without GC using chunked loss
bsz = 2
slen = 8192
x = torch.randint(0, cfg.vocab_size, (bsz, slen), device=device)
y = torch.randint(0, cfg.vocab_size, (bsz, slen), device=device)

try:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        # Forward transformer blocks without GC
        h = model.embed(x)
        for b in model.blocks:
            h = b(h)
        h = model.norm(h)
        
        # Shift
        shift_h = h[:, :-1, :].reshape(-1, cfg.hidden_size)
        shift_y = y[:, 1:].reshape(-1)
        
        loss = chunked_cross_entropy(shift_h, model.lm_head.weight, shift_y, chunk_size=2048)
        loss.backward()
    
    mem = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"SUCCESS! Peak VRAM without Gradient Checkpointing: {mem:.2f} GB")
except torch.OutOfMemoryError as e:
    print(f"OOM without Gradient Checkpointing: {e}")
