#!/usr/bin/env python3
"""
Maria — 1B Parameter HyperGQE Language Model with FlashAttention & RoPE.
"""
import math
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hyper_gqe import HyperGQEAttention


# ── Maria 1B Configuration ───────────────────────────────────────────────────

class MariaConfig:
    vocab_size: int              = 128_256
    hidden_size: int             = 2048
    intermediate_size: int       = 5632
    num_hidden_layers: int       = 18
    num_attention_heads: int     = 16
    num_key_value_heads: int     = 4
    max_position_embeddings: int = 8192
    rope_theta: float            = 500_000.0
    attention_bias: bool         = False
    layer_norm_eps: float        = 1e-5
    dropout: float               = 0.0
    tie_weights: bool            = True
    gradient_checkpointing: bool = True  # Activates gradient checkpointing for 8k context

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


# ── Primitives ───────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * rms) * self.weight


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network with SiLU gating."""
    def __init__(self, config: MariaConfig):
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up   = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ── Transformer Block ────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, config: MariaConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, config.layer_norm_eps)
        self.attn      = HyperGQEAttention(config)
        self.ffn_norm  = RMSNorm(config.hidden_size, config.layer_norm_eps)
        self.ffn       = SwiGLUFFN(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        normed = self.attn_norm(x)
        attn_out, _ = self.attn(normed, attention_mask=attention_mask)
        x = residual + attn_out

        residual = x
        x = residual + self.ffn(self.ffn_norm(x))
        return x


def _block_forward(block, x, mask):
    """Wrapper for activation gradient checkpointing."""
    return block(x, attention_mask=mask)


# ── Full Maria Language Model ────────────────────────────────────────────────

class MariaLM(nn.Module):
    def __init__(self, config: MariaConfig = None):
        super().__init__()
        self.config = config or MariaConfig()

        self.embed = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.drop  = nn.Dropout(self.config.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(self.config)
            for _ in range(self.config.num_hidden_layers)
        ])

        self.norm    = RMSNorm(self.config.hidden_size, self.config.layer_norm_eps)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

        if self.config.tie_weights:
            self.lm_head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self):
        std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

    def get_router_aux_loss(self) -> torch.Tensor:
        """Accumulate auxiliary router load balancing loss across all layers."""
        loss = torch.tensor(0.0)
        for block in self.blocks:
            attn = block.attn
            if hasattr(attn, "router_loss") and isinstance(attn.router_loss, torch.Tensor):
                loss = loss.to(attn.router_loss.device)
                loss = loss + attn.router_loss
        return loss

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        bsz, seq_len = input_ids.shape

        x = self.drop(self.embed(input_ids))

        # Forward through transformer layers with gradient checkpointing
        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    _block_forward, block, x, attention_mask, use_reentrant=False
                )
            else:
                x = block(x, attention_mask=attention_mask)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # Shift tokens for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            aux_loss = self.get_router_aux_loss()
            loss = lm_loss + 0.01 * aux_loss.to(lm_loss.device)

        return {"loss": loss, "logits": logits}

    def num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def non_embedding_parameters(self) -> int:
        embed_params = self.embed.weight.numel()
        return self.num_parameters(trainable_only=False) - embed_params

    def flops_per_token(self) -> float:
        N = self.non_embedding_parameters()
        return 6.0 * N + (12.0 * self.config.num_hidden_layers * self.config.hidden_size)


def build_model(config: MariaConfig = None) -> MariaLM:
    return MariaLM(config=config or MariaConfig())
