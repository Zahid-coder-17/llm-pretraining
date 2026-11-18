import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Rotary Position Embeddings ───────────────────────────────────────────────

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 8192, base: float = 500000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(self.max_position_embeddings)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int):
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype, device=x.device),
            self.sin_cached[:seq_len].to(dtype=x.dtype, device=x.device),
        )


# ── High-Throughput Fused HyperGQE Attention ─────────────────────────────────

class HyperGQEAttention(nn.Module):
    """
    Fused-Head HyperGQE Attention Engine:
      - 4 Routed Query Experts + 1 Dynamic HyperExpert + 1 Shared Head (6 active heads)
      - Fused single-pass FlashAttention across all heads
      - Fast sequence-level HyperNetwork adapter GEMMs
      - Native 8k RoPE (theta=500,000)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        
        self.num_groups = self.num_key_value_heads
        self.experts_per_group = self.num_heads // self.num_groups
        self.top_k = 1
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.q_shared = nn.Linear(self.hidden_size, self.head_dim, bias=config.attention_bias)
        
        # 4 routed + 1 hyper + 1 shared = 6 slots
        self.num_output_slots = 6
        self.o_proj = nn.Linear(self.num_output_slots * self.head_dim, self.hidden_size, bias=config.attention_bias)
        
        self.router = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        self.router_loss = 0.0

        # HyperMoE specific components
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

        rope_theta = getattr(config, "rope_theta", 500000.0)
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=rope_theta
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        # 1. Routing
        router_logits = self.router(hidden_states).view(bsz, q_len, self.num_groups, self.experts_per_group)
        router_probs = F.softmax(router_logits, dim=-1)
        topk_probs, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        # 2. Query, Key, Value extraction
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_groups, self.experts_per_group, self.head_dim)
        k_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        v_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        shared_q = self.q_shared(hidden_states).view(bsz, q_len, 1, self.head_dim)

        # Gather selected routed queries
        gather_indices = topk_indices.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim)
        sq = torch.gather(query_states, dim=3, index=gather_indices).squeeze(3) # (bsz, q_len, 4, head_dim)

        # 3. Dynamic HyperExpert (Fast In-RAM Adapter)
        Z_i = torch.zeros_like(router_logits).scatter_(-1, topk_indices, 1.0)
        Z_hat = 1.0 - Z_i
        unselected_avg = (Z_hat.unsqueeze(-1) * self.expert_embeddings).sum(dim=(2, 3)) / (Z_hat.sum(dim=(2, 3)).unsqueeze(-1) + 1e-6)
        
        p_seq = self.selection_mlp(unselected_avg.mean(dim=1))
        D = self.hyper_W_D(p_seq).view(bsz, self.hidden_size, self.b)
        U = self.hyper_W_U(p_seq).view(bsz, self.b, self.head_dim)
        
        q_hyper = F.relu(torch.bmm(hidden_states, D))
        q_hyper = torch.bmm(q_hyper, U).unsqueeze(2) # (bsz, q_len, 1, head_dim)

        # 4. RoPE Positional Encoding
        cos, sin = self.rotary_emb(hidden_states, seq_len=q_len)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # Combine 6 queries: 4 routed + 1 hyper + 1 shared = 6 heads
        all_q = torch.cat([sq, q_hyper, shared_q], dim=2).transpose(1, 2) # (bsz, 6, q_len, head_dim)
        
        # Mean key & value for hyper + shared heads
        mean_k = k_states.mean(dim=2, keepdim=True)
        mean_v = v_states.mean(dim=2, keepdim=True)
        all_k = torch.cat([k_states, mean_k, mean_k], dim=2).transpose(1, 2) # (bsz, 6, q_len, head_dim)
        all_v = torch.cat([v_states, mean_v, mean_v], dim=2).transpose(1, 2) # (bsz, 6, q_len, head_dim)

        # Apply RoPE
        all_q, all_k = apply_rotary_pos_emb(all_q, all_k, cos, sin)

        # 5. SINGLE FUSED FLASHATTENTION CALL
        is_causal = (attention_mask is None)
        attn_out = F.scaled_dot_product_attention(
            all_q, all_k, all_v,
            attn_mask=attention_mask,
            is_causal=is_causal
        ).transpose(1, 2) # (bsz, q_len, 6, head_dim)

        # 6. Output Projection
        output = self.o_proj(attn_out.reshape(bsz, q_len, -1))

        # Auxiliary router load balancing loss
        me = router_probs.mean(dim=(0, 1))
        loss = 0.0
        for g in range(self.num_groups):
            loss = loss + torch.var(me[g]) / (torch.mean(me[g])**2 + 1e-6)
        self.router_loss = loss

        return output, router_logits
