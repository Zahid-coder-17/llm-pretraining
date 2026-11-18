import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ── Optimized Contiguous HyperGQE Attention ──────────────────────────────────
class OptimizedHyperGQEAttention(nn.Module):
    def __init__(self, hidden_size=2048, num_heads=16, num_kv_heads=4, head_dim=128, max_pos=8192):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_groups = num_kv_heads
        self.experts_per_group = num_heads // self.num_groups  # 4
        
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.q_shared = nn.Linear(hidden_size, head_dim, bias=False)
        
        # 4 routed + 1 hyper + 1 shared = 6 slots
        self.o_proj = nn.Linear(6 * head_dim, hidden_size, bias=False)
        self.router = nn.Linear(hidden_size, num_heads, bias=False)
        
        # HyperNetwork components
        self.d_exp = 64
        self.d_sel = 64
        self.b = 16
        self.expert_embeddings = nn.Parameter(torch.randn(self.num_groups, self.experts_per_group, self.d_exp))
        self.selection_mlp = nn.Sequential(
            nn.Linear(self.d_exp, 128),
            nn.ReLU(),
            nn.Linear(128, self.d_sel)
        )
        self.hyper_W_D = nn.Linear(self.d_sel, self.hidden_size * self.b)
        self.hyper_W_U = nn.Linear(self.d_sel, self.b * self.head_dim)

    def forward(self, hidden_states):
        bsz, q_len, _ = hidden_states.size()
        
        # 1. Routing
        router_logits = self.router(hidden_states).view(bsz, q_len, self.num_groups, self.experts_per_group)
        router_probs = F.softmax(router_logits, dim=-1)
        topk_probs, topk_indices = torch.topk(router_probs, 1, dim=-1)
        
        # 2. Extract Query, Key, Value
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_groups, self.experts_per_group, self.head_dim)
        k_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)
        v_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)
        shared_q = self.q_shared(hidden_states).view(bsz, q_len, 1, self.head_dim)
        
        # Gather routed queries
        gather_indices = topk_indices.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim)
        sq = torch.gather(query_states, dim=3, index=gather_indices).squeeze(3) # (bsz, q_len, 4, head_dim)
        
        # 3. Dynamic HyperExpert (Fused Sequence Representation)
        Z_i = torch.zeros_like(router_logits).scatter_(-1, topk_indices, 1.0)
        Z_hat = 1.0 - Z_i
        unselected_avg = (Z_hat.unsqueeze(-1) * self.expert_embeddings).sum(dim=(2, 3)) / (Z_hat.sum(dim=(2, 3)).unsqueeze(-1) + 1e-6)
        
        # Sequence-pooled hypernetwork generation (single static GEMM per batch)
        p_seq = self.selection_mlp(unselected_avg.mean(dim=1)) # (bsz, d_sel)
        D = self.hyper_W_D(p_seq).view(bsz, self.hidden_size, self.b)
        U = self.hyper_W_U(p_seq).view(bsz, self.b, self.head_dim)
        
        # Fast two-stage linear projection
        q_hyper = F.relu(torch.bmm(hidden_states, D)) # (bsz, q_len, b)
        q_hyper = torch.bmm(q_hyper, U).unsqueeze(2)  # (bsz, q_len, 1, head_dim)
        
        # 4. Fused Attention Engine
        # Combine all queries: 4 routed + 1 hyper + 1 shared = 6 heads!
        all_q = torch.cat([sq, q_hyper, shared_q], dim=2).transpose(1, 2) # (bsz, 6, q_len, head_dim)
        
        # Mean key & value for hyper + shared heads
        mean_k = k_states.mean(dim=2, keepdim=True) # (bsz, q_len, 1, head_dim)
        mean_v = v_states.mean(dim=2, keepdim=True) # (bsz, q_len, 1, head_dim)
        all_k = torch.cat([k_states, mean_k, mean_k], dim=2).transpose(1, 2) # (bsz, 6, q_len, head_dim)
        all_v = torch.cat([v_states, mean_v, mean_v], dim=2).transpose(1, 2) # (bsz, 6, q_len, head_dim)
        
        # SINGLE UNIFIED FLASHATTENTION KERNEL CALL!
        attn_out = F.scaled_dot_product_attention(all_q, all_k, all_v, is_causal=True).transpose(1, 2) # (bsz, q_len, 6, head_dim)
        
        # 5. Output projection
        out = self.o_proj(attn_out.reshape(bsz, q_len, -1))
        return out

print(f"Benchmarking Fused-Head HyperGQE Attention on {torch.cuda.get_device_name(0)}...")

layer = OptimizedHyperGQEAttention().to(device, dtype=torch.bfloat16)

bsz = 2
slen = 8192
x = torch.randn(bsz, slen, 2048, device=device, dtype=torch.bfloat16, requires_grad=True)

# Warmup
for _ in range(5):
    out = layer(x)
    out.sum().backward()
    layer.zero_grad(set_to_none=True)
torch.cuda.synchronize()

# Timed test across 20 iterations
t0 = time.perf_counter()
N = 20
for _ in range(N):
    out = layer(x)
    out.sum().backward()
    layer.zero_grad(set_to_none=True)
torch.cuda.synchronize()
t1 = time.perf_counter()

dt_layer = (t1 - t0) / N
print(f"Optimized Layer Forward+Backward: {dt_layer*1000:.2f} ms per layer!")

# Full 18-layer model extrapolation
total_step_time = dt_layer * 18 + (0.008 * 18) # attention + FFN
tok_per_sec_1gpu = (bsz * slen) / total_step_time
print(f"Projected Micro-Step Time: {total_step_time*1000:.1f} ms")
print(f"Single GPU Throughput:     {tok_per_sec_1gpu:,.0f} tok/s")
print(f"4x H100 Cluster Projected: {tok_per_sec_1gpu * 4:,.0f} tok/s")
