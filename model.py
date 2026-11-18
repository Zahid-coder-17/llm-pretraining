#!/usr/bin/env python3
"""
model.py ? Shared transformer backbone for GQA / GQE / HyperGQE pre-training.

The ONLY difference between the 3 model variants is the attention class plugged in.
Everything else (FFN, RMSNorm, embeddings, LM head) is byte-for-byte identical.

Usage:
    from model import TransformerLM, PretrainConfig
    from models.gqa.gqa       import GQAAttention
    from models.gqe.gqe       import GQEAttention
    from models.hyper_gqe.hyper_gqe import HyperGQEAttention

    cfg   = PretrainConfig()
    model = TransformerLM(cfg, attention_cls=GQAAttention)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ?? Config ???????????????????????????????????????????????????????????????????

class PretrainConfig:
    """
    ~128M-parameter config compatible with all 3 attention modules.
    Fits comfortably in 6 GB VRAM with bfloat16 + gradient checkpointing.
    """
    vocab_size              = 32_000
    hidden_size             = 1280         # Scaled up for ~500M parameters
    intermediate_size       = 5120         # FFN width (4x hidden_size)
    num_hidden_layers       = 24           # Doubled layers
    num_attention_heads     = 20           # 20 heads
    num_key_value_heads     = 5            # 4 heads per KV group
    max_position_embeddings = 512
    attention_bias          = False
    layer_norm_eps          = 1e-5
    dropout                 = 0.0          # 0 for pre-training
    tie_weights             = True         # tie embedding ? LM head (saves 24.6M params)
    gradient_checkpointing  = False        # set True to save VRAM

    # Derived (do not set manually)
    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads


# ?? Primitives ????????????????????????????????????????????????????????????????

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network (same as TinyLlama / LLaMA-2)."""
    def __init__(self, config: PretrainConfig):
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up   = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ?? Causal mask helper ????????????????????????????????????????????????????????

def make_causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Returns additive causal mask of shape (1, 1, seq_len, seq_len)."""
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


# ?? Transformer Block ????????????????????????????????????????????????????????

class TransformerBlock(nn.Module):
    def __init__(self, config: PretrainConfig, attention_cls):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, config.layer_norm_eps)
        self.attn      = attention_cls(config)
        self.ffn_norm  = RMSNorm(config.hidden_size, config.layer_norm_eps)
        self.ffn       = SwiGLUFFN(config)

    def forward(
        self,
        x:              torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # Attention (pre-norm + residual)
        residual  = x
        normed    = self.attn_norm(x)
        attn_out, _ = self.attn(normed, attention_mask=attention_mask)
        x = residual + attn_out

        # FFN (pre-norm + residual)
        residual = x
        x = residual + self.ffn(self.ffn_norm(x))
        return x


def _block_forward(block, x, mask):
    """Wrapper for gradient checkpointing."""
    return block(x, attention_mask=mask)


# ?? Full Language Model ????????????????????????????????????????????????????????

class TransformerLM(nn.Module):
    def __init__(self, config: PretrainConfig, attention_cls):
        super().__init__()
        self.config       = config
        self.attention_cls = attention_cls

        self.embed     = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embed = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.drop      = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(config, attention_cls)
            for _ in range(config.num_hidden_layers)
        ])

        self.norm    = RMSNorm(config.hidden_size, config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_weights:
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
        """Collects auxiliary router losses from GQE / HyperGQE blocks."""
        loss = torch.tensor(0.0)
        for block in self.blocks:
            attn = block.attn
            if hasattr(attn, "router_loss") and isinstance(attn.router_loss, torch.Tensor):
                loss = loss.to(attn.router_loss.device)
                loss = loss + attn.router_loss
        return loss

    def forward(
        self,
        input_ids:      torch.Tensor,           # (bsz, seq_len)
        attention_mask: torch.Tensor = None,    # additive causal mask
        labels:         torch.Tensor = None,    # (bsz, seq_len) shifted inside
    ):
        bsz, seq_len = input_ids.shape
        device       = input_ids.device
        dtype        = self.embed.weight.dtype

        # Build causal mask if not provided
        if attention_mask is None:
            attention_mask = make_causal_mask(seq_len, device, dtype)

        # Embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.drop(self.embed(input_ids) + self.pos_embed(positions))

        # Transformer blocks
        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    _block_forward, block, x, attention_mask, use_reentrant=False
                )
            else:
                x = block(x, attention_mask=attention_mask)

        x      = self.norm(x)
        logits = self.lm_head(x)           # (bsz, seq_len, vocab_size)

        loss = None
        if labels is not None:
            # Shift: predict token t+1 from token t
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            # Add auxiliary router loss (0 for GQA, non-zero for GQE/HyperGQE)
            aux_loss = self.get_router_aux_loss()
            loss = lm_loss + 0.01 * aux_loss.to(lm_loss.device)

        return {"loss": loss, "logits": logits}

    def num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# ?? Factory ???????????????????????????????????????????????????????????????????

ARCH_MAP = {
    "gqa":       ("models.gqa.gqa",               "GQAAttention"),
    "gqe":       ("models.gqe.gqe",               "GQEAttention"),
    "hyper_gqe": ("models.hyper_gqe.hyper_gqe",   "HyperGQEAttention"),
}


def build_model(arch: str, config: PretrainConfig = None) -> TransformerLM:
    if arch not in ARCH_MAP:
        raise ValueError(f"Unknown arch '{arch}'. Choose from: {list(ARCH_MAP)}")
    if config is None:
        config = PretrainConfig()

    module_path, cls_name = ARCH_MAP[arch]
    import importlib
    mod = importlib.import_module(module_path)
    attention_cls = getattr(mod, cls_name)

    return TransformerLM(config, attention_cls)


# ?? CLI quick test ?????????????????????????????????????????????????????????????

if __name__ == "__main__":
    import sys
    arch = sys.argv[1] if len(sys.argv) > 1 else "gqa"
    cfg  = PretrainConfig()
    cfg.gradient_checkpointing = False

    print(f"\nBuilding {arch.upper()} model...")
    model = build_model(arch, cfg)
    total = model.num_parameters(trainable_only=False)
    train = model.num_parameters(trainable_only=True)
    print(f"  Total params:     {total/1e6:.2f}M")
    print(f"  Trainable params: {train/1e6:.2f}M")

    # Forward pass test
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(x, labels=y)
    print(f"  Loss: {out['loss'].item():.4f}")
    print(f"  Logits shape: {out['logits'].shape}")
    print(f"  OK.")

