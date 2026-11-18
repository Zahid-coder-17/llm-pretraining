import os
import sys
import torch
from transformers import AutoTokenizer

possible_dirs = ["/workspace/Maria", ".", "../.."]
for d in possible_dirs:
    if os.path.exists(os.path.join(d, "model.py")):
        sys.path.insert(0, d)
        break

from model import MariaConfig, build_model

SYSTEM_PROMPT = "You are Maria, a helpful, articulate, and friendly AI assistant. You speak in fluent, natural English. You give clear, direct answers and excel at conversation, document editing, and everyday assistance."

device = "cuda:0" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B")

ckpt_candidates = [
    "/workspace/sft_output/maria_1b_20B_instruct_v2.pt",
    "./sft_output/maria_1b_20B_instruct_v2.pt",
    "/workspace/sft_output/maria_1b_20B_instruct_chat.pt"
]

ckpt_path = None
for c in ckpt_candidates:
    if os.path.exists(c):
        ckpt_path = c
        break

print(f"Loading checkpoint from: {ckpt_path}...")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
clean_sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}

cfg = MariaConfig()
cfg.gradient_checkpointing = False
model = build_model(cfg).to(device=device, dtype=torch.bfloat16)
model.load_state_dict(clean_sd, strict=False)
model.eval()

test_prompts = [
    ("1. Greeting", "Hi!"),
    ("2. Chitchat", "How are you?"),
    ("3. Factual Arithmetic", "What is 2 + 2?"),
    ("4. Empathy Support", "I had a difficult day."),
    ("5. Compound Greeting", "Hi, how are you?")
]

print("\n" + "=" * 70)
print("  POST-ALIGNMENT VERIFICATION TEST ON MARIA INSTRUCT V2")
print("=" * 70)

for label, p in test_prompts:
    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"{p}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )
    input_ids = tok.encode(prompt, return_tensors="pt").to(device)
    curr = input_ids
    stopped_by_eot = False

    with torch.inference_mode():
        for _ in range(60):
            out = model(curr)
            logits = out["logits"][:, -1, :].clone()
            probs = torch.softmax(logits / 0.4, dim=-1)
            next_t = torch.multinomial(probs, num_samples=1)
            val = next_t.item()
            if val in {128001, 128009}:
                stopped_by_eot = True
                break
            curr = torch.cat([curr, next_t], dim=1)

    gen = curr[0, input_ids.shape[1]:]
    res = tok.decode(gen, skip_special_tokens=True).strip()
    status_icon = "[OK - Stopped by EOS]" if stopped_by_eot else "[!] Hit max tokens"
    print(f"\n{label} -> User: '{p}'")
    print(f"Maria: '{res}' {status_icon}")

print("\n" + "=" * 70)
print("Verification complete!")
