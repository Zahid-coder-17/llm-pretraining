import os
import sys
import torch
from transformers import AutoTokenizer

sys.path.insert(0, '/workspace/Maria')
from model import MariaConfig, build_model

SYSTEM_PROMPT = "You are Maria, a helpful, articulate, and friendly AI assistant. You speak in fluent, natural English. You give clear, direct answers and excel at conversation, document editing, and everyday assistance."
device = "cuda:0" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B")
ckpt_path = "/workspace/sft_output/maria_1b_20B_instruct_v2.pt"

ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
clean_sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}

cfg = MariaConfig()
cfg.gradient_checkpointing = False
model = build_model(cfg).to(device=device, dtype=torch.bfloat16)
model.load_state_dict(clean_sd, strict=False)
model.eval()

prompt_text = "Hi maria, how's ur day"
prompt = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    + SYSTEM_PROMPT + "<|eot_id|>"
    + "<|start_header_id|>user<|end_header_id|>\n"
    + prompt_text + "<|eot_id|>"
    + "<|start_header_id|>assistant<|end_header_id|>\n"
)

input_ids = tok.encode(prompt, return_tensors="pt").to(device)
curr = input_ids
stopped = False

with torch.inference_mode():
    for _ in range(80):
        out = model(curr)
        logits = out["logits"][:, -1, :].clone()
        probs = torch.softmax(logits / 0.5, dim=-1)
        next_t = torch.multinomial(probs, num_samples=1)
        val = next_t.item()
        if val in {128001, 128009}:
            stopped = True
            break
        curr = torch.cat([curr, next_t], dim=1)

gen = curr[0, input_ids.shape[1]:]
res = tok.decode(gen, skip_special_tokens=True).strip()

print("========================================")
print(f"PROMPT: {prompt_text}")
print(f"RESPONSE: {res}")
print(f"STOPPED_BY_EOS: {stopped}")
print("========================================")
