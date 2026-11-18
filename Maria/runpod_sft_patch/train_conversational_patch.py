import os
import sys
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

# Locate Maria model modules
possible_dirs = ["/workspace/Maria", ".", "../.."]
for d in possible_dirs:
    if os.path.exists(os.path.join(d, "model.py")):
        sys.path.insert(0, d)
        break

from model import MariaConfig, build_model

SYSTEM_PROMPT = "You are Maria, a helpful, articulate, and friendly AI assistant. You speak in fluent, natural English. You give clear, direct answers and excel at conversation, document editing, and everyday assistance."

class ShortChatDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_seq_len: int = 256):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.items = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.items.append(json.loads(line))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        inst = item["instruction"].strip()
        resp = item["response"].strip()

        # Strict LLaMA-3 Chat Template with <|eot_id|>
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"{SYSTEM_PROMPT}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n"
            f"{inst}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
        )
        response = f"{resp}<|eot_id|>"

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = self.tokenizer.encode(response, add_special_tokens=False)

        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids

        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]

        pad_len = self.max_seq_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id or 0] * pad_len
            labels = labels + [-100] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

def find_checkpoint():
    candidates = [
        "/workspace/vault_backup/maria_1b_20B_weights.pt",
        "/workspace/Maria/checkpoints_original/20Btokens/maria_1b_20B_weights.pt",
        "./checkpoints_original/20Btokens/maria_1b_20B_weights.pt",
        "/workspace/sft_output/maria_1b_20B_instruct_chat.pt"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("  MARIA 1.07B CONVERSATIONAL ALIGNMENT SFT (EOS UP-WEIGHTING)")
    print(f"  Device: {device} | Dtype: torch.bfloat16")
    print("=" * 70)

    # 1. Load Tokenizer
    tok_candidates = [
        "NousResearch/Meta-Llama-3-8B",
        "/root/.cache/huggingface/hub/models--NousResearch--Meta-Llama-3-8B"
    ]
    tokenizer = None
    for tc in tok_candidates:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tc, use_fast=True)
            print(f"[*] Loaded tokenizer from: {tc}")
            break
        except Exception:
            pass

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Dataset
    data_path = "/workspace/sft_data/maria_short_chat_10k.jsonl" if os.path.exists("/workspace") else "./sft_data/maria_short_chat_10k.jsonl"
    if not os.path.exists(data_path):
        print(f"[!] Error: Dataset not found at {data_path}. Run build_conversational_patch_dataset.py first!")
        sys.exit(1)

    dataset = ShortChatDataset(data_path, tokenizer, max_seq_len=256)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    print(f"[*] Loaded {len(dataset):,} short conversational samples.")

    # 3. Model & Base Weights
    ckpt_path = find_checkpoint()
    if not ckpt_path:
        print("[!] Error: Base checkpoint not found!")
        sys.exit(1)

    print(f"[*] Reloading base foundation weights from: {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    clean_sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}

    cfg = MariaConfig()
    cfg.gradient_checkpointing = False
    model = build_model(cfg).to(device=device, dtype=torch.bfloat16)
    model.load_state_dict(clean_sd, strict=False)

    # 4. EOS Up-Weighted Loss (Crucial Fix)
    print("[*] Configuring CrossEntropyLoss with 5x EOS (<|eot_id|>) Multiplier...")
    loss_weights = torch.ones(cfg.vocab_size, device=device, dtype=torch.float32)
    EOT_ID = 128009  # <|eot_id|> in Llama-3
    loss_weights[EOT_ID] = 5.0  # 5x penalty for skipping stop token!
    loss_fn = nn.CrossEntropyLoss(weight=loss_weights, ignore_index=-100)

    # 5. Training Hyperparameters
    EPOCHS = 3
    LR = 2.0e-5
    MIN_LR = 2.0e-6
    GRAD_ACCUM = 2

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = (len(dataloader) // GRAD_ACCUM) * EPOCHS
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=MIN_LR)

    out_dir = Path("/workspace/sft_output") if os.path.exists("/workspace") else Path("./sft_output")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "maria_1b_20B_instruct_v2.pt"

    print(f"[*] Total Steps: {total_steps} | Epochs: {EPOCHS} | Peak LR: {LR}")
    print("-" * 70)

    step = 0
    t0 = time.time()

    for epoch in range(EPOCHS):
        model.train()
        accum_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(dataloader):
            x = batch["input_ids"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)["logits"]
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = y[..., 1:].contiguous()
                loss = loss_fn(shift_logits.view(-1, cfg.vocab_size), shift_labels.view(-1))
                loss = loss / GRAD_ACCUM

            loss.backward()
            accum_loss += loss.item()

            if (i + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if step % 20 == 0 or step == total_steps:
                    curr_lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - t0
                    print(f"  [Epoch {epoch+1}/{EPOCHS}] Step {step:4d}/{total_steps} | Loss: {accum_loss*GRAD_ACCUM:.4f} | LR: {curr_lr:.2e} | Elapsed: {elapsed:.1f}s")
                accum_loss = 0.0

    print("-" * 70)
    print(f"[*] Training Complete in {(time.time()-t0)/60:.1f} minutes!")
    print(f"[*] Saving aligned checkpoint to: {save_path}...")
    torch.save({
        "model_state_dict": model.state_dict(),
        "step": step,
        "epochs": EPOCHS,
        "loss": accum_loss
    }, save_path)
    print(f"[OK] Checkpoint saved successfully! Size: {os.path.getsize(save_path)/(1024**3):.2f} GB")

if __name__ == "__main__":
    main()
