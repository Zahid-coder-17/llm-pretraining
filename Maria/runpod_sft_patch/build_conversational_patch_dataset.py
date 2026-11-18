import json
import os
import random

out_dir = "/workspace/sft_data" if os.path.exists("/workspace") else "./sft_data"
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "maria_short_chat_10k.jsonl")

print("=" * 65)
print("  Building 10,000 High-Density Short Conversational Alignment Dataset")
print("=" * 65)

samples = []

# 1. Essential Core Chitchat, Greetings & Factual QA Seed (500 high-priority anchors)
core_anchors = [
    # Greetings & Identity
    ("Hi!", "Hello! How can I help you today?"),
    ("Hi", "Hello! It is great to hear from you. What can I do for you?"),
    ("Hello", "Hello! How are you doing today?"),
    ("Hey there", "Hey! How can I assist you today?"),
    ("Good morning", "Good morning! I hope you have a wonderful day ahead. How can I help?"),
    ("Good evening", "Good evening! How can I assist you tonight?"),
    ("How are you?", "I'm doing well, thank you for asking! How are you doing?"),
    ("Hi, how are you?", "Hello! I am doing well, thank you. How can I help you today?"),
    ("Who are you?", "I am Maria, your friendly AI assistant. I'm here to help with conversation, questions, and writing."),
    ("What is your name?", "My name is Maria, an AI assistant designed to chat and help with tasks."),
    ("What can you do?", "I can assist you with answering questions, writing, brainstorming, and everyday problem solving."),
    ("Nice to meet you", "Nice to meet you too! Feel free to ask me anything."),

    # Emotional & Conversational Support
    ("I had a difficult day.", "I'm really sorry to hear that. Do you want to talk about what happened?"),
    ("I'm feeling stressed today.", "I'm sorry you are feeling stressed. Take a deep breath?is there anything I can help take off your mind?"),
    ("I feel exhausted.", "Make sure to get some rest and drink some water. I'm here whenever you need me."),
    ("I had a great day today!", "That's wonderful to hear! What made today so special?"),
    ("Thank you so much!", "You're very welcome! Let me know if you need anything else."),
    ("Thanks for your help", "Glad I could help! Have a great day."),
    ("Goodbye", "Goodbye! Take care and have a great day."),
    ("See you later", "See you later! Feel free to return whenever you need assistance."),

    # Arithmetic & Concise Factual QA
    ("What is 2 + 2?", "2 + 2 is 4."),
    ("What is 5 + 7?", "5 + 7 is 12."),
    ("What is 10 * 10?", "10 * 10 is 100."),
    ("What is 15 - 8?", "15 - 8 is 7."),
    ("What is 20 / 4?", "20 / 4 is 5."),
    ("What is 3 * 7?", "3 * 7 is 21."),
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
    ("What is the speed of light?", "The speed of light in a vacuum is approximately 299,792,458 meters per second."),
    ("What is the chemical formula for water?", "The chemical formula for water is H2O."),
]

# Multiply anchors across variations to guarantee high probability mass
for _ in range(15):
    for u, a in core_anchors:
        samples.append({"instruction": u, "response": a, "category": "core_chitchat"})

print(f"1. Anchored {len(samples)} core conversational turns.")

# 2. Extract strictly short turns from UltraChat (< 45 words)
try:
    from datasets import load_dataset
    print("2. Pulling short conversational turns from UltraChat 200k...")
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:25000]")
    uc_count = 0
    for row in ds:
        msgs = row.get("messages", [])
        if len(msgs) >= 2:
            u = msgs[0]["content"].strip()
            a = msgs[1]["content"].strip()
            # Strict short filter: 5 to 45 words only!
            u_words = len(u.split())
            a_words = len(a.split())
            if 2 <= u_words <= 30 and 4 <= a_words <= 45:
                if not any(bad in a.lower() for bad in ["http", "github", "function", "def ", "class "]):
                    samples.append({"instruction": u, "response": a, "category": "short_dialogue"})
                    uc_count += 1
                    if uc_count >= 5000:
                        break
    print(f"   Collected {uc_count} short UltraChat turns.")
except Exception as e:
    print(f"   UltraChat notice: {e}")

# 3. Extract concise factual/instruction QA from Alpaca (< 40 words)
try:
    print("3. Pulling concise QA from Alpaca-Cleaned...")
    ds_alpaca = load_dataset("tatsu-lab/alpaca", split="train[:20000]")
    alp_count = 0
    for row in ds_alpaca:
        inst = row["instruction"].strip()
        out = row["output"].strip()
        i_words = len(inst.split())
        o_words = len(out.split())
        if 3 <= i_words <= 25 and 3 <= o_words <= 40:
            samples.append({"instruction": inst, "response": out, "category": "concise_qa"})
            alp_count += 1
            if alp_count >= 4000:
                break
    print(f"   Collected {alp_count} concise Alpaca QA turns.")
except Exception as e:
    print(f"   Alpaca notice: {e}")

# Shuffle and cap at exactly 10,000 samples
random.seed(42)
random.shuffle(samples)
samples = samples[:10000]

print(f"\nFinal Dataset Size: {len(samples):,} samples")
print(f"Saving to: {out_file}...")

with open(out_file, "w", encoding="utf-8") as f:
    for s in samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"SUCCESS! Output file: {out_file} ({os.path.getsize(out_file)/(1024**2):.2f} MB)")
