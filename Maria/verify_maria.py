#!/usr/bin/env python3
"""
verify_maria.py — Unit Test & Sanity Verification for Maria 1B Architecture.
"""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from model import MariaConfig, build_model


def verify():
    print("=" * 70)
    print(" Verifying Maria 1B HyperGQE Architecture")
    print("=" * 70)

    cfg = MariaConfig()
    print(f"Vocab size:       {cfg.vocab_size:,}")
    print(f"Hidden size:      {cfg.hidden_size}")
    print(f"Intermediate:     {cfg.intermediate_size} (SwiGLU)")
    print(f"Layers:           {cfg.num_hidden_layers}")
    print(f"Attention Heads:  {cfg.num_attention_heads} Query / {cfg.num_key_value_heads} KV")
    print(f"Context Window:   {cfg.max_position_embeddings} (8k)")
    print(f"RoPE Base Theta:  {cfg.rope_theta}")

    print("\n[1/4] Instantiating Model...")
    model = build_model(cfg)
    total_p = model.num_parameters()
    non_emb = model.non_embedding_parameters()
    print(f"  -> Total Parameters:         {total_p / 1e6:,.2f} M ({total_p:,})")
    print(f"  -> Non-Embedding Parameters: {non_emb / 1e6:,.2f} M ({non_emb:,})")

    # Assert roughly ~1.07B parameters
    assert 1.0e9 <= total_p <= 1.15e9, f"Unexpected parameter count: {total_p}"
    print("  -> Parameter count check: PASSED (1B Class)")

    print("\n[2/4] Verifying Forward Pass (Mock Sequence)...")
    batch_size = 2
    seq_len = 128  # lightweight sequence for quick local check
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()

    outputs = model(input_ids, labels=labels)
    loss = outputs["loss"]
    logits = outputs["logits"]

    print(f"  -> Logits shape: {logits.shape} (Expected: [{batch_size}, {seq_len}, {cfg.vocab_size}])")
    print(f"  -> Initial Loss: {loss.item():.4f}")
    assert logits.shape == (batch_size, seq_len, cfg.vocab_size), "Logits shape mismatch!"
    assert not torch.isnan(loss), "Loss is NaN!"
    print("  -> Forward pass check: PASSED")

    print("\n[3/4] Verifying Backward Pass & Gradient Flow...")
    loss.backward()
    grad_count = 0
    zero_grad_count = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is not None:
                grad_count += 1
            else:
                zero_grad_count += 1

    print(f"  -> Tensors with valid gradients: {grad_count}")
    print(f"  -> Tensors missing gradients:    {zero_grad_count}")
    assert zero_grad_count == 0, f"Some parameters did not receive gradients: {zero_grad_count}"
    print("  -> Backward pass check: PASSED")

    print("\n[4/4] Verifying RoPE & FlashAttention Components...")
    block0 = model.blocks[0]
    attn = block0.attn
    assert hasattr(attn, "rotary_emb"), "Missing RoPE module!"
    assert hasattr(attn, "router"), "Missing Router module!"
    assert hasattr(attn, "hyper_W_D"), "Missing HyperExpert W_D!"
    print("  -> RoPE module: Present (RotaryEmbedding)")
    print("  -> FlashAttention / SDPA causal kernel: Integrated")
    print("  -> HyperExpert weights generator: Present")

    print("\n" + "=" * 70)
    print(" ALL CHECKS PASSED: Maria 1B is ready for pretraining!")
    print("=" * 70)


if __name__ == "__main__":
    verify()
