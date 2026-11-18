import torch
import sentencepiece as spm
from model import build_model, PretrainConfig
import sys

def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading tokenizer and MARIA_v1.pt...")
    sp = spm.SentencePieceProcessor(model_file="tokenizer.model")
    mc = PretrainConfig()
    model = build_model("hyper_gqe", mc).to(device)
    
    try:
        ckpt = torch.load("MARIA_v1.pt", map_location=device)
        model.load_state_dict(ckpt["model"])
        model.eval()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    prompts = [
        "Hello, who are you?",
        "What is the capital of France?",
        "import torch\ndef add_numbers(a, b):",
        "الذكاء الاصطناعي هو"
    ]
    
    print("Running tests...")
    with open("maria_test_results.txt", "w", encoding="utf-8") as f:
        for raw_prompt in prompts:
            prompt = f"System: You are MARIA, a helpful AI assistant.\nUser: {raw_prompt}\nMARIA: "
            input_ids = sp.encode(prompt)
            x = torch.tensor([input_ids], dtype=torch.long).to(device)
            
            generated = []
            for _ in range(50):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(x)["logits"]
                
                next_token_logits = logits[0, -1, :] / 0.8
                v, _ = torch.topk(next_token_logits, min(50, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[-1]] = -float('Inf')
                
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                if next_token.item() == 3: # EOS
                    break
                    
                generated.append(next_token.item())
                x = torch.cat((x, next_token.unsqueeze(0)), dim=1)
            
            output = sp.decode(generated)
            f.write(f"PROMPT: {raw_prompt}\nMARIA: {output}\n" + "-"*40 + "\n")
    print("Test complete. Results saved to maria_test_results.txt")

if __name__ == "__main__":
    test()
