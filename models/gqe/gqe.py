import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GQEAttention(nn.Module):
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
        
        self.num_output_slots = (self.top_k * self.num_groups) + 1 + 1
        self.o_proj = nn.Linear(self.num_output_slots * self.head_dim, self.hidden_size, bias=config.attention_bias)
        
        self.router = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        self.router_loss = 0.0

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_embeddings: tuple = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        router_logits = self.router(hidden_states)
        router_logits = router_logits.view(bsz, q_len, self.num_groups, self.experts_per_group)
        
        router_probs = F.softmax(router_logits, dim=-1)
        topk_probs, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_groups, self.experts_per_group, self.head_dim)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        
        shared_query_states = self.q_shared(hidden_states).view(bsz, q_len, 1, self.head_dim)

        if position_embeddings is not None:
            pass 

        gather_indices = topk_indices.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim)
        selected_queries = torch.gather(query_states, dim=3, index=gather_indices)
        
        sq = selected_queries.squeeze(3).transpose(1, 2) 
        sk = key_states.transpose(1, 2)
        sv = value_states.transpose(1, 2)
        
        attn_weights = torch.matmul(sq, sk.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, sv) 
        attn_output = attn_output.transpose(1, 2) 
        
        hard_routed = attn_output 
        
        flat_probs = topk_probs.squeeze(-1)
        renorm_probs = flat_probs / (flat_probs.sum(dim=-1, keepdim=True) + 1e-6)
        
        weighted_sum = (attn_output * renorm_probs.unsqueeze(-1)).sum(dim=2, keepdim=True)
        
        mean_k = sk.mean(dim=1, keepdim=True) 
        mean_v = sv.mean(dim=1, keepdim=True) 
        shared_q = shared_query_states.transpose(1, 2) 
        
        shared_attn_weights = torch.matmul(shared_q, mean_k.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            shared_attn_weights = shared_attn_weights + attention_mask
        shared_attn_weights = F.softmax(shared_attn_weights, dim=-1)
        shared_attn_output = torch.matmul(shared_attn_weights, mean_v).transpose(1, 2) 
        
        final_slots = torch.cat([hard_routed, weighted_sum, shared_attn_output], dim=2)
        final_output = final_slots.view(bsz, q_len, -1)
        output = self.o_proj(final_output)
        
        me = router_probs.mean(dim=(0, 1)) 
        loss = 0.0
        for g in range(self.num_groups):
            loss += torch.var(me[g]) / (torch.mean(me[g])**2 + 1e-6)
        self.router_loss = loss
        
        return output, router_logits
