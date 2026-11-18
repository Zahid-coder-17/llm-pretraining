#!/usr/bin/env python3
"""
pack_sequences.py — Stage 7: Pack tokenized documents into 512-token blocks.

Pipeline:
  doc tokens → append EOS (id=3) → fill 512-token blocks
  - No cross-document leaking (EOS separator is explicit)
  - Incomplete final block is DROPPED (not padded) for training
  - For val/test, final incomplete block is also dropped (documented policy)
  - Output: .bin files (uint16 numpy arrays) in shards

Output structure:
  data/train/english/shard_00000.bin
  data/train/arabic/shard_00000.bin
  data/validation/english/shard_00000.bin
  ...

Run:  python pack_sequences.py [--split train|val|test] [--dataset <name>]
"""
import argparse, json, logging, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import config as C

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("pack_sequences.log")]
)
log = logging.getLogger(__name__)

EOS_ID     = 3      # as configured in train_tokenizer.py
DTYPE      = np.uint16


def get_lang_dir(split_name: str, lang: str) -> Path:
    lang_str = "english" if lang == "en" else "arabic"
    if split_name == "train":
        return C.TRAIN_DIR / lang_str
    elif split_name == "val":
        return C.VAL_DIR / lang_str
    else:
        return C.TEST_DIR / lang_str


def iter_tokens(tokenized_file: Path):
    """Yield flat token lists from tokenized JSONL."""
    with open(tokenized_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
                tokens = doc.get("tokens", [])
                if tokens:
                    yield tokens
            except json.JSONDecodeError:
                continue


def pack_to_shards(name: str, split: str, lang: str):
    tokenized_f = C.TOKENIZED_DIR / name / f"{split}.jsonl"
    if not tokenized_f.exists():
        log.warning(f"[{name}/{split}] Tokenized file not found — skipping.")
        return 0, 0

    out_dir = get_lang_dir(split, lang)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if already packed by looking for index file
    index_f = out_dir / f"{name}_{split}_index.json"
    if index_f.exists():
        log.info(f"[{name}/{split}] Already packed — skipping.")
        idx = json.loads(index_f.read_text())
        return idx.get("total_sequences", 0), idx.get("shard_count", 0)

    t0 = time.time()
    buffer      = []          # accumulates tokens between EOS markers
    shard_buf   = []          # current shard token array
    shard_idx   = 0
    total_seqs  = 0
    shard_files = []

    def flush_shard():
        nonlocal shard_idx, shard_buf
        if not shard_buf:
            return
        arr = np.array(shard_buf, dtype=DTYPE)
        shard_name = f"{name}_{split}_shard_{shard_idx:05d}.bin"
        shard_path = out_dir / shard_name
        arr.tofile(shard_path)
        n_seqs = len(arr) // C.SEQ_LEN
        shard_files.append({
            "file":     shard_name,
            "tokens":   len(arr),
            "sequences": n_seqs,
        })
        log.info(f"[{name}/{split}] Flushed shard {shard_idx}: "
                 f"{len(arr)/1e6:.1f}M tokens, {n_seqs:,} sequences")
        shard_idx += 1
        shard_buf = []

    for doc_tokens in iter_tokens(tokenized_f):
        # Append EOS after each document
        combined = doc_tokens + [EOS_ID]
        buffer.extend(combined)

        # Extract complete 512-token blocks from buffer
        while len(buffer) >= C.SEQ_LEN:
            block = buffer[:C.SEQ_LEN]
            buffer = buffer[C.SEQ_LEN:]
            shard_buf.extend(block)
            total_seqs += 1

            # Flush shard when it reaches target size
            if len(shard_buf) >= C.SHARD_SIZE_TOKENS:
                flush_shard()

    # Drop incomplete final block (do NOT pad)
    dropped = len(buffer)
    if dropped > 0:
        log.info(f"[{name}/{split}] Dropped {dropped} tokens in incomplete final block (correct policy)")

    # Flush remaining shard
    if shard_buf:
        flush_shard()

    elapsed = time.time() - t0
    index   = {
        "dataset":         name,
        "split":           split,
        "lang":            lang,
        "total_sequences": total_seqs,
        "total_tokens":    total_seqs * C.SEQ_LEN,
        "shard_count":     shard_idx,
        "dropped_tokens":  dropped,
        "seq_len":         C.SEQ_LEN,
        "eos_id":          EOS_ID,
        "dtype":           "uint16",
        "final_block_policy": "dropped",
        "elapsed_seconds": round(elapsed, 1),
        "shards":          shard_files,
    }
    index_f.write_text(json.dumps(index, indent=2), encoding="utf-8")
    log.info(f"[{name}/{split}] Done: {total_seqs:,} sequences, {shard_idx} shards in {elapsed:.0f}s")
    return total_seqs, shard_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",   default=None, choices=["train", "val", "test"])
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args()
    C.ensure_dirs()

    splits   = [args.split] if args.split else ["train", "val", "test"]
    datasets = {args.dataset: C.DATASETS[args.dataset]} if args.dataset else C.DATASETS

    grand_total = 0
    for split in splits:
        for name, cfg in datasets.items():
            seqs, shards = pack_to_shards(name, split, cfg["lang"])
            grand_total += seqs
            log.info(f"  {name}/{split}: {seqs:,} sequences in {shards} shards")

    log.info(f"Total sequences packed: {grand_total:,}")


if __name__ == "__main__":
    main()
