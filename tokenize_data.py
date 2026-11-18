#!/usr/bin/env python3
"""
tokenize_data.py — Stage 6: Tokenize all splits and measure actual token counts.

Input:  data/splits/<dataset>/{train,val,test}.jsonl  +  data/tokenizer/tokenizer.model
Output: data/tokenized/<dataset>/{train,val,test}.jsonl  (token lists as JSON arrays)
        data/tokenized/<dataset>/token_counts.json

Run:  python tokenize_data.py [--dataset <name>]
"""
import argparse, json, logging, sys, time
from pathlib import Path

import sentencepiece as spm

sys.path.insert(0, str(Path(__file__).parent))
import config as C

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("tokenize_data.log")]
)
log = logging.getLogger(__name__)


def load_tokenizer() -> spm.SentencePieceProcessor:
    model_file = C.TOKENIZER_DIR / "tokenizer.model"
    if not model_file.exists():
        log.error("Tokenizer model not found. Run train_tokenizer.py first.")
        sys.exit(1)
    sp = spm.SentencePieceProcessor(model_file=str(model_file))
    return sp


def iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def tokenize_split(name: str, split: str, sp: spm.SentencePieceProcessor) -> dict:
    inp = C.SPLITS_DIR  / name / f"{split}.jsonl"
    out_dir = C.TOKENIZED_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{split}.jsonl"

    if out.exists():
        log.info(f"[{name}/{split}] Already tokenized — skipping.")
        counts_f = out_dir / "token_counts.json"
        if counts_f.exists():
            return json.loads(counts_f.read_text()).get(split, {})
        return {}

    if not inp.exists():
        log.warning(f"[{name}/{split}] Input not found — skipping.")
        return {}

    t0    = time.time()
    total_tokens = 0
    total_docs   = 0
    min_tok = float("inf")
    max_tok = 0
    raw_chars = 0

    tmp = out_dir / f"{split}.jsonl.tmp"

    with open(tmp, "w", encoding="utf-8") as fout:
        for doc in iter_jsonl(inp):
            text   = doc["text"]
            tokens = sp.encode(text, out_type=int)
            if not tokens:
                continue

            n = len(tokens)
            total_tokens += n
            total_docs   += 1
            raw_chars    += len(text)
            min_tok = min(min_tok, n)
            max_tok = max(max_tok, n)

            out_doc = {
                "tokens": tokens,
                "source": doc.get("source", name),
                "doc_id": doc.get("doc_id", ""),
                "lang":   doc.get("lang", ""),
            }
            fout.write(json.dumps(out_doc, ensure_ascii=False) + "\n")

            if total_docs % 100_000 == 0:
                log.info(f"[{name}/{split}] {total_docs:,} docs, {total_tokens/1e6:.1f}M tokens")

    tmp.rename(out)
    elapsed = time.time() - t0

    stats = {
        "documents":          total_docs,
        "actual_tokens":      total_tokens,
        "raw_chars":          raw_chars,
        "avg_tokens_per_doc": round(total_tokens / max(total_docs, 1), 1),
        "min_tokens_per_doc": min_tok if total_docs > 0 else 0,
        "max_tokens_per_doc": max_tok,
        "avg_chars_per_token": round(raw_chars / max(total_tokens, 1), 3),
        "elapsed_seconds":    round(elapsed, 1),
    }

    # Append to token_counts.json
    counts_f = out_dir / "token_counts.json"
    all_counts = json.loads(counts_f.read_text()) if counts_f.exists() else {}
    all_counts[split] = stats
    counts_f.write_text(json.dumps(all_counts, indent=2), encoding="utf-8")

    log.info(f"[{name}/{split}] Done: {total_docs:,} docs, "
             f"{total_tokens/1e6:.2f}M tokens in {elapsed:.0f}s")
    return stats


def tokenize_dataset(name: str, sp: spm.SentencePieceProcessor):
    for split in ("train", "val", "test"):
        tokenize_split(name, split, sp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args()
    C.ensure_dirs()
    C.TOKENIZED_DIR.mkdir(parents=True, exist_ok=True)

    sp = load_tokenizer()
    log.info(f"Tokenizer loaded, vocab size = {sp.get_piece_size()}")

    targets = [args.dataset] if args.dataset else list(C.DATASETS.keys())
    for name in targets:
        tokenize_dataset(name, sp)

    log.info("All tokenization complete.")


if __name__ == "__main__":
    main()
