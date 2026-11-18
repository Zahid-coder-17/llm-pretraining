
#!/usr/bin/env python3
"""
split_data.py — Stage 4: Train/Validation/Test splitting.

Rules (from spec):
  - Reserve TEST documents first (until test token target reached)
  - Reserve VALIDATION documents next
  - Remaining documents = training pool
  - Select training documents until train token target reached
  - NEVER split a document across splits
  - Shuffle deterministically with seed 42
  - Token counts estimated via CHARS_PER_TOKEN (actuals measured after tokenization)

Input:  data/deduped/<dataset>/deduped.jsonl
Output: data/splits/<dataset>/{train,val,test}.jsonl
        data/splits/<dataset>/split_stats.json

Run:  python split_data.py [--dataset <name>]
"""
import argparse, json, logging, random, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config as C

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("split_data.log")]
)
log = logging.getLogger(__name__)


def estimate_tokens(text: str, lang: str) -> int:
    return max(1, int(len(text) / C.CHARS_PER_TOKEN.get(lang, 4.0)))


def iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def split_dataset(name: str, cfg: dict):
    inp     = C.DEDUPED_DIR / name / "deduped.jsonl"
    out_dir = C.SPLITS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_f = out_dir / "train.jsonl"
    val_f   = out_dir / "val.jsonl"
    test_f  = out_dir / "test.jsonl"
    stats_f = out_dir / "split_stats.json"

    if train_f.exists() and val_f.exists() and test_f.exists() and stats_f.exists():
        log.info(f"[{name}] Already split — skipping.")
        return

    if not inp.exists():
        log.error(f"[{name}] deduped.jsonl not found. Run deduplicate.py first.")
        sys.exit(1)

    lang = cfg["lang"]
    test_target  = cfg["test_tokens"]
    val_target   = cfg["val_tokens"]
    train_target = cfg["train_tokens"]

    log.info(f"[{name}] Loading all doc IDs for shuffle (memory: doc_ids only)...")
    t0 = time.time()

    # Pass 1: read all docs into memory by doc_id + estimated tokens
    # We need to shuffle before splitting. Store (doc_id, est_tokens) pairs.
    doc_info = []
    total_docs = 0
    for doc in iter_jsonl(inp):
        et = estimate_tokens(doc["text"], lang)
        doc_info.append((doc["doc_id"], et))
        total_docs += 1

    log.info(f"[{name}] {total_docs:,} docs loaded for shuffling.")

    rng = random.Random(C.SEED)
    rng.shuffle(doc_info)

    # Assign splits: test first, then val, then train
    test_ids  = set()
    val_ids   = set()
    train_ids = set()

    test_tok = val_tok = train_tok = 0

    for doc_id, et in doc_info:
        if test_tok < test_target:
            test_ids.add(doc_id)
            test_tok += et
        elif val_tok < val_target:
            val_ids.add(doc_id)
            val_tok += et
        elif train_tok < train_target:
            train_ids.add(doc_id)
            train_tok += et
        else:
            break  # All targets satisfied

    log.info(f"[{name}] Split sizes (estimated tokens): "
             f"test={test_tok/1e6:.1f}M, val={val_tok/1e6:.1f}M, train={train_tok/1e6:.1f}M")

    # Pass 2: write split files
    tmp_train = out_dir / "train.jsonl.tmp"
    tmp_val   = out_dir / "val.jsonl.tmp"
    tmp_test  = out_dir / "test.jsonl.tmp"

    counts = {"train": 0, "val": 0, "test": 0, "unused": 0}

    with open(tmp_train, "w", encoding="utf-8") as ftr, \
         open(tmp_val,   "w", encoding="utf-8") as fva, \
         open(tmp_test,  "w", encoding="utf-8") as fte:

        for doc in iter_jsonl(inp):
            did = doc["doc_id"]
            if did in test_ids:
                fte.write(json.dumps(doc, ensure_ascii=False) + "\n")
                counts["test"] += 1
            elif did in val_ids:
                fva.write(json.dumps(doc, ensure_ascii=False) + "\n")
                counts["val"] += 1
            elif did in train_ids:
                ftr.write(json.dumps(doc, ensure_ascii=False) + "\n")
                counts["train"] += 1
            else:
                counts["unused"] += 1

    tmp_train.rename(train_f)
    tmp_val.rename(val_f)
    tmp_test.rename(test_f)

    elapsed = time.time() - t0
    stats = {
        "dataset": name,
        "train_docs": counts["train"],
        "val_docs":   counts["val"],
        "test_docs":  counts["test"],
        "unused_docs": counts["unused"],
        "est_train_tokens": train_tok,
        "est_val_tokens":   val_tok,
        "est_test_tokens":  test_tok,
        "target_train_tokens": train_target,
        "target_val_tokens":   val_target,
        "target_test_tokens":  test_target,
        "elapsed_seconds": round(elapsed, 1),
        "note": "Token counts are estimates. Actual counts measured after tokenize_data.py",
    }
    stats_f.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log.info(f"[{name}] Split done in {elapsed:.0f}s: {counts}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args()
    C.ensure_dirs()
    C.SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    targets = {args.dataset: C.DATASETS[args.dataset]} if args.dataset else C.DATASETS
    for name, cfg in targets.items():
        split_dataset(name, cfg)


if __name__ == "__main__":
    main()
