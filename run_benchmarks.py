import argparse
import json
import torch
from pathlib import Path
from lm_eval import evaluator, tasks
from lm_eval.api.model import LM
from lm_eval.models.huggingface import HFLM
from tokenizers import Tokenizer

# Import your custom architecture
from config import TOKENIZER_DIR
from model import build_model, PretrainConfig

class HyperGQEWrapper(HFLM):
    """
    A custom wrapper to trick EleutherAI's evaluation harness 
    into thinking your custom HyperGQE model is a standard HuggingFace model!
    """
    def __init__(self, checkpoint_path, device="cuda"):
        super().__init__()
        self._device = device
        
        print(f"Loading HyperGQE architecture from {checkpoint_path}...")
        mc = PretrainConfig()
        
        # Build your custom model and load the 622M trained weights
        self.hyper_model = build_model("hyper_gqe", mc).to(device)
        ckpt = torch.load(checkpoint_path, map_location=device)
        self.hyper_model.load_state_dict(ckpt["model"])
        self.hyper_model.eval()
        
        # Load your custom BPE Tokenizer
        tokenizer_path = TOKENIZER_DIR / "bpe_32k.json"
        self.custom_tokenizer = Tokenizer.from_file(str(tokenizer_path))
        
        print("Model and Tokenizer successfully loaded into RTX 3050!")

    @property
    def eot_token_id(self):
        # End of text token
        return self.custom_tokenizer.token_to_id("<|endoftext|>")

    @property
    def max_length(self):
        return 512

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return 4

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str, **kwargs):
        return self.custom_tokenizer.encode(string).ids

    def tok_decode(self, tokens):
        return self.custom_tokenizer.decode(tokens)

    def _model_call(self, inps):
        """Passes the input tensors into your custom model"""
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = self.hyper_model(inps)
            return outputs["logits"]

    def _model_generate(self, context, max_length, eos_token_id):
        # We only need loglikelihoods for multiple choice, not full generation for these tests
        raise NotImplementedError("Generative tasks not supported in zero-shot choice mode")


def run_evaluations(checkpoint_path, task_list):
    print(f"\n--- Starting Zero-Shot Evaluation on RTX 3050 ---")
    print(f"Tasks: {task_list}")
    
    # Initialize our custom wrapper
    lm = HyperGQEWrapper(checkpoint_path=checkpoint_path)
    
    # Run EleutherAI Harness
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=task_list,
        num_fewshot=0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=4,
    )
    
    # Print beautiful results
    print("\n" + "="*50)
    print("Zero-Shot Evaluation Results")
    print("="*50)
    for task_name, metrics in results["results"].items():
        acc = metrics.get("acc,none", metrics.get("acc_norm,none", 0.0))
        print(f"Domain: {task_name.upper():<15} | Accuracy: {acc*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="best.pt", help="Path to best.pt")
    args = parser.parse_args()
    
    # The domains you requested!
    domains = [
        "arc_easy",       # Knowledge
        "hellaswag",      # Common Sense
        "blimp_syntax",   # Linguistics
        "xcopa_ar"        # Arabic
    ]
    
    run_evaluations(args.ckpt, domains)
