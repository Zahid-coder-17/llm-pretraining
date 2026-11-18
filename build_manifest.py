#!/usr/bin/env python3
"""
build_manifest.py — Stage 8: Build dataset_manifest.json and dataset_manifest.csv.

Reads all index files from packed shards and token count files.
Reports target vs actual tokens for every dataset/split.

Run:  python build_manifest.py
"""
import csv, hashlib, json, logging, sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config as C

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("build_manifest.log")]
)
log = logging.getLogger(__name__)

SPLIT_MAP = {"train": "train", "val": "validation", "test": "test"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_token_counts(name: str, split: str) -> dict:
    counts_f = C.TOKENIZED_DIR / name / "token_counts.json"
    if counts_f.exists():
        data = json.loads(counts_f.read_text())
        return data.get(split, {})
    return {}


def load_shard_index(name: str, split: str, lang: str) -> dict:
    lang_str = "english" if lang == "en" else "arabic"
    if split == "train":
        d = C.TRAIN_DIR / lang_str
    elif split == "val":
        d = C.VAL_DIR / lang_str
    else:
        d = C.TEST_DIR / lang_str
    idx_f = d / f"{name}_{split}_index.json"
    return json.loads(idx_f.read_text()) if idx_f.exists() else {}


def main():
    C.ensure_dirs()
    C.MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "seed":                C.SEED,
        "tokenizer_vocab_size": C.VOCAB_SIZE,
        "sequence_length":     C.SEQ_LEN,
        "created_at":          datetime.now(timezone.utc).isoformat(),
        "datasets":            {},
        "totals":              {},
        "deduplication": {
            "exact_method":              "sha256",
            "near_duplicate_method":     "minhash_lsh",
            "near_duplicate_threshold":  C.NEAR_DUP_THRESHOLD,
            "minhash_permutations":      C.MINHASH_NUM_PERM,
            "shingle_type":              "char_5gram",
            "cross_dataset_deduplication": True,
        },
        "file_hashes": {},
    }

    totals = dict(
        train_tokens=0, validation_tokens=0, test_tokens=0,
        english_train_tokens=0, arabic_train_tokens=0,
    )
    csv_rows = []

    for name, cfg in C.DATASETS.items():
        lang = cfg["lang"]
        ds_entry = {"language": lang}

        for raw_split, manifest_split in SPLIT_MAP.items():
            tok_counts = load_token_counts(name, raw_split)
            shard_idx  = load_shard_index(name, raw_split, lang)

            actual_tokens = tok_counts.get("actual_tokens",
                            shard_idx.get("total_tokens", None))
            actual_docs   = tok_counts.get("documents", None)
            target_key    = f"{raw_split}_tokens"

            ds_entry[manifest_split] = {
                "target_tokens": cfg.get(target_key, 0),
                "actual_tokens": actual_tokens,   # None = not yet tokenized
                "documents":     actual_docs,
                "shard_count":   shard_idx.get("shard_count", None),
                "sequences_512": shard_idx.get("total_sequences", None),
                "avg_tok_per_doc": tok_counts.get("avg_tokens_per_doc", None),
                "min_tok_per_doc": tok_counts.get("min_tokens_per_doc", None),
                "max_tok_per_doc": tok_counts.get("max_tokens_per_doc", None),
            }

            if actual_tokens:
                if raw_split == "train":
                    totals["train_tokens"] += actual_tokens
                    if lang == "en":
                        totals["english_train_tokens"] += actual_tokens
                    else:
                        totals["arabic_train_tokens"]  += actual_tokens
                elif raw_split == "val":
                    totals["validation_tokens"] += actual_tokens
                elif raw_split == "test":
                    totals["test_tokens"]       += actual_tokens

            csv_rows.append({
                "dataset":       name,
                "language":      lang,
                "split":         manifest_split,
                "target_tokens": cfg.get(target_key, 0),
                "actual_tokens": actual_tokens,
                "documents":     actual_docs,
                "shard_count":   shard_idx.get("shard_count", ""),
            })

        manifest["datasets"][name] = ds_entry

    manifest["totals"] = totals

    # Hash tokenizer and this manifest before writing
    tok_model = C.TOKENIZER_DIR / "tokenizer.model"
    if tok_model.exists():
        manifest["file_hashes"]["tokenizer.model"] = sha256_file(tok_model)

    # Write JSON manifest
    manifest_json = C.MANIFESTS_DIR / "dataset_manifest.json"
    manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["file_hashes"]["dataset_manifest.json"] = sha256_file(manifest_json)
    # rewrite with self-hash
    manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info(f"Wrote {manifest_json}")

    # Write CSV manifest
    manifest_csv = C.MANIFESTS_DIR / "dataset_manifest.csv"
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        w.writeheader()
        w.writerows(csv_rows)
    log.info(f"Wrote {manifest_csv}")

    # Print summary
    log.info("=== MANIFEST SUMMARY ===")
    for k, v in totals.items():
        log.info(f"  {k}: {v/1e6:.1f}M tokens" if isinstance(v, int) else f"  {k}: {v}")

    log.info("Manifest build complete.")


if __name__ == "__main__":
    main()
