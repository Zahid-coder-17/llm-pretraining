import os
os.environ['HF_HOME'] = './cache'
os.environ['HF_DATASETS_CACHE'] = './cache'
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from datasets import load_dataset
from model import build_model, PretrainConfig
from torch.amp import autocast
from torch.optim import AdamW
import time

# --- HYPERPARAMETERS FOR A100 80GB ---
BATCH_SIZE = 32
MAX_SEQ_LEN = 512
LEARNING_RATE = 1e-5
EPOCHS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SYSTEM_PROMPT = "System: You are MARIA, a helpful AI assistant.\n"

print("Downloading conversational dataset (ShareGPT format converted to Alpaca for phase 1.5)...")
# For this phase 1.5, we'll continue using Alpaca or upgrade to a ShareGPT-style dataset.
raw_dataset = load_dataset("tatsu-lab/alpaca", split="train[:50000]", cache_dir="./cache")

print("Loading SentencePiece tokenizer...")
sp = spm.SentencePieceProcessor(model_file="tokenizer.model")
EOS_ID = 3

class SFTDataset(Dataset):
    def __init__(self, hf_dataset, sp_tokenizer, max_length):
        self.data = []
        for row in hf_dataset:
            instruction = row['instruction']
            input_text = row['input']
            output = row['output']
            
            if input_text:
                prompt = f"{SYSTEM_PROMPT}User: {instruction}\nInput: {input_text}\nMARIA: "
            else:
                prompt = f"{SYSTEM_PROMPT}User: {instruction}\nMARIA: "
                
            full_text = prompt + output
            tokens = sp_tokenizer.encode(full_text)
            tokens.append(EOS_ID) 
            
            if len(tokens) <= max_length:
                self.data.append(tokens)
                
        print(f"Loaded {len(self.data)} valid training examples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]
        padded = tokens + [0] * (MAX_SEQ_LEN - len(tokens))
        x = torch.tensor(padded[:-1], dtype=torch.long)
        y = torch.tensor(padded[1:], dtype=torch.long)
        y[y == 0] = -100
        return x, y

print("Formatting data for MARIA...")
dataset = SFTDataset(raw_dataset, sp, MAX_SEQ_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Loading HyperGQE base model...")
mc = PretrainConfig()
model = build_model("hyper_gqe", mc).to(DEVICE)
ckpt = torch.load("best.pt", map_location=DEVICE)
model.load_state_dict(ckpt["model"])

# --- THE FIX: RESTORING ADAMW ---
print("Initializing AdamW Optimizer for A100...")
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

print("\n--- Starting Instruction Tuning (MARIA) ---")
model.train()
scaler = torch.amp.GradScaler()

for epoch in range(EPOCHS):
    total_loss = 0
    t0 = time.time()
    
    for i, (x, y) in enumerate(dataloader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        optimizer.zero_grad()
        
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(x, labels=y)
            loss = outputs["loss"]
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        if i % 100 == 0:
            print(f"Step {i} | Loss: {loss.item():.4f}")
            
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} complete! Avg Loss: {avg_loss:.4f} | Time: {time.time()-t0:.1f}s")

print("Saving MARIA...")
torch.save({"model": model.state_dict()}, "MARIA_v2.pt")
print("Saved as MARIA_v2.pt!")
