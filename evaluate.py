#!/usr/bin/env python3
"""
evaluate.py — Evaluation script for pre-trained GQA / GQE / HyperGQE checkpoints.

Computes:
  - Perplexity on validation set
  - Perplexity on test set
  - Per-language perplexity (English vs Arabic)
  - Side-by-side comparison table when --all is used

Run:
    python evaluate.py --arch gqa
    python evaluate.py --arch gqe
    python evaluate.py --arch hyper_gqe
    python evaluate.py --all           # compare all 3 side-by-side
"""
import argparse, json, logging, math, sys, time
from pathlib import Path

import torch
from torch.cuda.amp import autocast

sys.path.insert(0, str(Path(__file__).parent))
import config as C
from model   import build_model, PretrainConfig
from dataset import build_dataloader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def load_best_checkpoint(arch: str, device: torch.device, seed: int = 42) -> torch.nn.Module:
    ckpt_dir = Path("E:/pretrain_checkpoints") / f"{arch}_seed{seed}"
    best_f   = ckpt_dir / "best.pt"
    if not best_f.exists():
        # fall back to latest step checkpoint
        ckpts = sorted(ckpt_dir.glob("step_*.pt"))
        if not ckpts:
            log.error(f"No checkpoint found for {arch} in {ckpt_dir}")
            sys.exit(1)
        best_f = ckpts[-1]

    log.info(f"[{arch}] Loading checkpoint: {best_f}")
    ckpt  = torch.load(best_f, map_location=device)
    mc    = PretrainConfig()
    model = build_model(arch, mc).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def compute_perplexity(
    model, loader, device, dtype,
    max_batches: int = None,
    desc: str = ""
) -> dict:
    total_nll = 0.0
    total_tok = 0
    batches   = 0

    for batch in loader:
        if max_batches and batches >= max_batches:
            break
        x = batch.to(device)
        with autocast(dtype=dtype, enabled=device.type == "cuda"):
            out = model(x, labels=x)
        loss = out["loss"]
        n    = x[:, 1:].numel()     # shifted tokens
        total_nll += loss.item() * n
        total_tok += n
        batches   += 1

        if batches % 50 == 0:
            ppl_so_far = math.exp(min(total_nll / max(total_tok, 1), 20))
            log.info(f"  {desc} batch {batches}: running ppl={ppl_so_far:.2f}")

    avg_nll    = total_nll / max(total_tok, 1)
    perplexity = math.exp(min(avg_nll, 20))
    return {
        "nll":         avg_nll,
        "perplexity":  perplexity,
        "tokens":      total_tok,
        "batches":     batches,
    }


def evaluate_arch(arch: str, device: torch.device, dtype: torch.dtype, seed: int = 42) -> dict:
    model, ckpt = load_best_checkpoint(arch, device, seed=seed)
    mc = PretrainConfig()
    results = {"arch": arch, "checkpoint_step": ckpt.get("step", "?")}

    # Full validation set
    log.info(f"[{arch}] Evaluating on validation set (both languages)...")
    val_loader = build_dataloader("val", lang="both", batch_size=8,
                                  num_workers=2, shuffle=False)
    val_metrics = compute_perplexity(model, val_loader, device, dtype, desc="val-both")
    results["val"] = val_metrics

    # Full test set
    log.info(f"[{arch}] Evaluating on test set (both languages)...")
    test_loader = build_dataloader("test", lang="both", batch_size=8,
                                   num_workers=2, shuffle=False)
    test_metrics = compute_perplexity(model, test_loader, device, dtype, desc="test-both")
    results["test"] = test_metrics

    # Per-language validation perplexity
    for lang, lang_name in [("en", "english"), ("ar", "arabic")]:
        try:
            lang_loader = build_dataloader("val", lang=lang, batch_size=8,
                                           num_workers=2, shuffle=False)
            m = compute_perplexity(model, lang_loader, device, dtype,
                                   desc=f"val-{lang_name}")
            results[f"val_{lang_name}"] = m
        except FileNotFoundError:
            log.warning(f"[{arch}] No {lang_name} val shards found")

    # Log summary
    log.info(f"\n{'='*50}")
    log.info(f"  Architecture:     {arch.upper()}")
    log.info(f"  Checkpoint step:  {results['checkpoint_step']}")
    log.info(f"  Val  perplexity:  {val_metrics['perplexity']:.3f}")
    log.info(f"  Test perplexity:  {test_metrics['perplexity']:.3f}")
    if "val_english" in results:
        log.info(f"  Val  EN ppl:      {results['val_english']['perplexity']:.3f}")
    if "val_arabic" in results:
        log.info(f"  Val  AR ppl:      {results['val_arabic']['perplexity']:.3f}")
    log.info(f"{'='*50}\n")

    # Save results
    out_f = Path("E:/pretrain_checkpoints") / f"{arch}_seed{seed}" / "eval_results.json"
    out_f.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info(f"[{arch}] Results saved to {out_f}")
    return results


def print_comparison_table(all_results: list):
    log.info("\n" + "="*70)
    log.info("COMPARISON TABLE — GQA vs GQE vs HyperGQE (Pre-trained from Scratch)")
    log.info("="*70)
    header = f"{'Architecture':<15} {'Val PPL':>10} {'Test PPL':>10} {'EN Val PPL':>12} {'AR Val PPL':>12}"
    log.info(header)
    log.info("-"*70)
    for r in all_results:
        arch    = r["arch"].upper()
        val_ppl = f"{r['val']['perplexity']:.3f}"
        tst_ppl = f"{r['test']['perplexity']:.3f}"
        en_ppl  = f"{r.get('val_english', {}).get('perplexity', float('nan')):.3f}"
        ar_ppl  = f"{r.get('val_arabic',  {}).get('perplexity', float('nan')):.3f}"
        log.info(f"{arch:<15} {val_ppl:>10} {tst_ppl:>10} {en_ppl:>12} {ar_ppl:>12}")
    log.info("="*70)
    log.info("Lower perplexity = better.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None,
                        help="Evaluate specific seed. Omit to evaluate all 3 seeds and report mean±std.")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--arch", choices=["gqa", "gqe", "hyper_gqe"])
    grp.add_argument("--all",  action="store_true",
                     help="Evaluate all 3 architectures and print comparison table")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16

    seed = args.seed
    if args.all and seed is None:
        # Full comparison: 3 seeds × 3 architectures
        summaries = []
        for arch in ["gqa", "gqe", "hyper_gqe"]:
            s = evaluate_with_seeds(arch, device, dtype)
            summaries.append(s)
        print_mean_std_table(summaries)
        out_f = Path("E:/pretrain_checkpoints") / "comparison_mean_std.json"
        out_f.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        log.info(f"Saved to {out_f}")
    elif args.all:
        all_results = []
        for arch in ["gqa", "gqe", "hyper_gqe"]:
            r = evaluate_arch(arch, device, dtype)
            all_results.append(r)
        print_comparison_table(all_results)

        # Save comparison
        out_f = Path("E:/pretrain_checkpoints") / "comparison_results.json"
        out_f.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        log.info(f"Comparison saved to {out_f}")
    else:
        evaluate_arch(args.arch, device, dtype)


if __name__ == "__main__":
    main()




import statistics as _stats

def evaluate_with_seeds(arch: str, device, dtype) -> dict:
    """Run evaluation across all 3 seeds and return mean +- std."""
    all_val_ppl  = []
    all_test_ppl = []
    all_en_ppl   = []
    all_ar_ppl   = []

    for seed in C.TRAIN_SEEDS:
        ckpt_dir = Path("E:/pretrain_checkpoints") / f"{arch}_seed{seed}"
        if not ckpt_dir.exists():
            log.warning(f"[{arch} seed={seed}] Checkpoint dir not found — skipping")
            continue
        r = evaluate_arch(arch, device, dtype, seed=seed)
        all_val_ppl.append(r["val"]["perplexity"])
        all_test_ppl.append(r["test"]["perplexity"])
        if "val_english" in r:
            all_en_ppl.append(r["val_english"]["perplexity"])
        if "val_arabic" in r:
            all_ar_ppl.append(r["val_arabic"]["perplexity"])

    def fmt(vals):
        if not vals: return "N/A"
        if len(vals) == 1: return f"{vals[0]:.3f}"
        return f"{_stats.mean(vals):.3f} ± {_stats.stdev(vals):.3f}"

    summary = {
        "arch":      arch,
        "seeds":     C.TRAIN_SEEDS,
        "val_ppl":   fmt(all_val_ppl),
        "test_ppl":  fmt(all_test_ppl),
        "en_val_ppl": fmt(all_en_ppl),
        "ar_val_ppl": fmt(all_ar_ppl),
        "raw": {
            "val_ppl": all_val_ppl, "test_ppl": all_test_ppl,
            "en_ppl": all_en_ppl,   "ar_ppl": all_ar_ppl,
        }
    }
    return summary


def print_mean_std_table(summaries: list):
    log.info("\n" + "="*80)
    log.info("FINAL COMPARISON TABLE — mean ± std across 3 seeds (42, 123, 456)")
    log.info("="*80)
    hdr = f"{'Architecture':<15} {'Val PPL':>18} {'Test PPL':>18} {'EN Val PPL':>18} {'AR Val PPL':>18}"
    log.info(hdr)
    log.info("-"*80)
    for s in summaries:
        log.info(f"{s['arch'].upper():<15} {s['val_ppl']:>18} {s['test_ppl']:>18} {s['en_val_ppl']:>18} {s['ar_val_ppl']:>18}")
    log.info("="*80)
    log.info("Lower perplexity = better language modeling.")

