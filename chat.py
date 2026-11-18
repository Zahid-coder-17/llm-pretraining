import torch
import sys
import sentencepiece as spm
from model import build_model, PretrainConfig

def load_model_and_tokenizer(checkpoint_path, device="cuda"):
    print(f"Loading tokenizer...")
    sp = spm.SentencePieceProcessor(model_file="tokenizer.model")
    
    print(f"Loading HyperGQE architecture from {checkpoint_path}...")
    mc = PretrainConfig()
    model = build_model("hyper_gqe", mc).to(device)
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    
    return model, sp

@torch.no_grad()
def generate_text(model, sp, prompt, max_new_tokens=50, temperature=0.7, top_k=50, device="cuda"):
    # Encode prompt
    input_ids = sp.encode(prompt)
    x = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    print(f"\nPrompt: {prompt}")
    print("Generating...", end="", flush=True)
    
    generated_tokens = []
    
    for _ in range(max_new_tokens):
        # Forward pass
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)["logits"]
        
        # Get logits for the last token
        next_token_logits = logits[0, -1, :]
        
        # Temperature scaling
        next_token_logits = next_token_logits / temperature
        
        # Top-K filtering
        v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
        next_token_logits[next_token_logits < v[-1]] = -float('Inf')
        
        # Sample
        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append to output and context
        generated_tokens.append(next_token.item())
        x = torch.cat((x, next_token.unsqueeze(0)), dim=1)
        
        # Stop if we hit max sequence length (512)
        if next_token.item() == 3:
            break
        if x.size(1) >= 512:
            break
            
    # Decode
    output_text = sp.decode(generated_tokens)
    print(f"\n\nHyperGQE (500M): {prompt}{output_text}\n")
    return output_text

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model, sp = load_model_and_tokenizer("best.pt", device)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
        
    print("\n" + "="*50)
    print("HyperGQE Interactive Chat (Type 'quit' to exit)")
    print("="*50)
    
    while True:
        prompt = input("\nYou: ")
        
        if prompt.lower() in ["quit", "exit"]:
            break
        if prompt.strip() == "":
            continue
            
        generate_text(model, sp, prompt, max_new_tokens=100, temperature=0.8, device=device)
