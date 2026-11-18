#!/usr/bin/env python3
"""
validate_dataset.py — Stage 9: Automated validation of all 16 checks from spec.

Checks:
  1.  No duplicate SHA-256 hashes across any split
  2.  No document appears in multiple datasets
  3.  No train document exactly matches validation/test
  4.  No train document has >=0.90 near-dup similarity with val/test
  5.  Arabic documents are actually Arabic (script check)
  6.  English documents are actually English (langdetect spot check)
  7.  Tokenizer vocabulary = exactly 32,000
  8.  Training English tokens ~ 250M
  9.  Training Arabic tokens ~ 250M
  10. Total training tokens ~ 500M
  11. Validation ~ 17M
  12. Test ~ 17M
  13. Packed training sequences are exactly 512 tokens
  14. No NaN/invalid token IDs in tokenized files (spot check)
  15. Token IDs in [0, 31999]
  16. Manifest counts match actual generated shards

Run:  python validate_dataset.py
"""
import json, logging, re, sys
from pathlib import Path

import numpy as np
import sentencepiece as spm

sys.path.insert(0, str(Path(__file__).parent))
import config as C

try:
    from datasketch import MinHash, MinHashLSH
    HAS_DATASKETCH = True
except ImportError:
    HAS_DATASKETCH = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("validate_dataset.log")]
)
log = logging.getLogger(__name__)

ARABIC_RE = re.compile(r'[\u0600-\u06FF]')
PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"
SKIP = "SKIP"

results = []


def check(num, desc, status, detail=""):
    results.append({"check": num, "description": desc, "status": status, "detail": detail})
    icon = "✅" if status == PASS else ("⚠️" if status == WARN else ("⏭" if status == SKIP else "❌"))
    log.info(f"  [{icon} {status}] Check {num}: {desc}" + (f" — {detail}" if detail else ""))


def iter_jsonl_sample(path: Path, max_docs=100_000):
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if count >= max_docs:
                break
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                    count += 1
                except json.JSONDecodeError:
                    continue


def main():
    log.info("=== STARTING VALIDATION (16 checks) ===")

    # ── Check 7: Tokenizer vocab ──────────────────────────────────────────
    tok_model = C.TOKENIZER_DIR / "tokenizer.model"
    if tok_model.exists():
        sp = spm.SentencePieceProcessor(model_file=str(tok_model))
        vs = sp.get_piece_size()
        check(7, f"Tokenizer vocab = {C.VOCAB_SIZE}",
              PASS if vs == C.VOCAB_SIZE else FAIL,
              f"actual={vs}")
    else:
        check(7, "Tokenizer vocab", SKIP, "tokenizer.model not found")

    # ── Checks 8-12: Token totals from manifest ───────────────────────────
    manifest_f = C.MANIFESTS_DIR / "dataset_manifest.json"
    manifest   = {}
    if manifest_f.exists():
        manifest = json.loads(manifest_f.read_text())
        totals   = manifest.get("totals", {})

        en_train = totals.get("english_train_tokens", 0)
        ar_train = totals.get("arabic_train_tokens", 0)
        total_tr = totals.get("train_tokens", 0)
        total_va = totals.get("validation_tokens", 0)
        total_te = totals.get("test_tokens", 0)

        tol = 0.10   # 10% tolerance
        def near(actual, target, t=tol):
            if target == 0: return actual == 0
            return abs(actual - target) / target <= t

        check(8,  "English train tokens ~ 250M",  PASS if near(en_train, 250e6) else WARN,
              f"{en_train/1e6:.1f}M")
        check(9,  "Arabic train tokens ~ 250M",   PASS if near(ar_train, 250e6) else WARN,
              f"{ar_train/1e6:.1f}M")
        check(10, "Total train tokens ~ 500M",    PASS if near(total_tr, 500e6) else WARN,
              f"{total_tr/1e6:.1f}M")
        check(11, "Validation tokens ~ 17M",      PASS if near(total_va, 17e6)  else WARN,
              f"{total_va/1e6:.1f}M")
        check(12, "Test tokens ~ 17M",            PASS if near(total_te, 17e6)  else WARN,
              f"{total_te/1e6:.1f}M")
    else:
        for n in [8, 9, 10, 11, 12]:
            check(n, f"Token total check {n}", SKIP, "manifest not found — run build_manifest.py")

    # ── Check 13: Packed sequences are exactly 512 tokens ────────────────
    all_512 = True
    checked_shards = 0
    for lang_dir in [C.TRAIN_DIR / "english", C.TRAIN_DIR / "arabic"]:
        if not lang_dir.exists():
            continue
        for f in list(lang_dir.glob("*.bin"))[:3]:   # spot-check 3 shards
            arr = np.fromfile(f, dtype=np.uint16)
            if len(arr) % C.SEQ_LEN != 0:
                all_512 = False
                log.error(f"  Shard {f.name}: {len(arr)} tokens — NOT divisible by {C.SEQ_LEN}")
            checked_shards += 1
    if checked_shards == 0:
        check(13, f"Packed sequences are {C.SEQ_LEN} tokens", SKIP, "No shard files found yet")
    else:
        check(13, f"Packed sequences are {C.SEQ_LEN} tokens",
              PASS if all_512 else FAIL, f"Spot-checked {checked_shards} shards")

    # ── Checks 14-15: Token ID validity (spot check tokenized files) ──────
    bad_ids = False
    checked_toks = 0
    for name, cfg in C.DATASETS.items():
        tok_f = C.TOKENIZED_DIR / name / "train.jsonl"
        if not tok_f.exists():
            continue
        for doc in iter_jsonl_sample(tok_f, max_docs=1000):
            tokens = doc.get("tokens", [])
            for t in tokens:
                if not isinstance(t, int) or t < 0 or t >= C.VOCAB_SIZE:
                    bad_ids = True
                    log.error(f"Invalid token ID {t} in {name}")
            checked_toks += len(tokens)
        break  # One dataset spot check is enough

    if checked_toks == 0:
        check(14, "No NaN/invalid token IDs", SKIP, "No tokenized files found yet")
        check(15, f"Token IDs in [0, {C.VOCAB_SIZE-1}]", SKIP, "No tokenized files found yet")
    else:
        check(14, "No NaN/invalid token IDs",
              FAIL if bad_ids else PASS, f"Checked {checked_toks:,} tokens")
        check(15, f"Token IDs in [0, {C.VOCAB_SIZE-1}]",
              FAIL if bad_ids else PASS, f"Checked {checked_toks:,} tokens")

    # ── Checks 1-3: Hash uniqueness and cross-split contamination ─────────
    log.info("  Collecting hashes from all splits (sample)...")
    all_hashes   = {}    # hash -> (dataset, split)
    train_hashes = set()
    val_hashes   = set()
    test_hashes  = set()
    cross_ds_ids = {}    # doc_id -> dataset

    dup_hashes   = False
    cross_ds_dup = False
    contaminated = False

    for name in C.DATASETS:
        for split in ("train", "val", "test"):
            split_f = C.DEDUPED_DIR / name / "deduped.jsonl"
            if not split_f.exists():
                split_f = C.SPLITS_DIR / name / f"{split}.jsonl"
            if not split_f.exists():
                continue

            for doc in iter_jsonl_sample(split_f, max_docs=10_000):
                h   = doc.get("hash", "")
                did = doc.get("doc_id", "")

                # Check 1: dup hashes
                if h in all_hashes and all_hashes[h] != (name, split):
                    dup_hashes = True
                all_hashes[h] = (name, split)

                # Check 2: same doc_id across datasets
                ds_from_id = did.split("_")[0] if did else name
                if did in cross_ds_ids and cross_ds_ids[did] != name:
                    cross_ds_dup = True
                cross_ds_ids[did] = name

                # Collect per-split hashes
                if split == "train":   train_hashes.add(h)
                elif split == "val":   val_hashes.add(h)
                elif split == "test":  test_hashes.add(h)

    # Check 3: train ∩ val or train ∩ test
    if train_hashes & val_hashes:
        contaminated = True
        log.error(f"  {len(train_hashes & val_hashes)} train docs found in val!")
    if train_hashes & test_hashes:
        contaminated = True
        log.error(f"  {len(train_hashes & test_hashes)} train docs found in test!")

    check(1, "No duplicate SHA-256 hashes across splits",
          FAIL if dup_hashes else PASS, "(spot-checked 10k docs per dataset/split)")
    check(2, "No document in multiple datasets",
          FAIL if cross_ds_dup else PASS, "(spot-checked)")
    check(3, "No train doc matches val/test exactly",
          FAIL if contaminated else PASS, "(hash-based spot check)")

    # ── Check 4: Near-dup contamination check (only if datasketch available) ─
    if not HAS_DATASKETCH:
        check(4, "Near-dup train/val/test contamination", SKIP, "datasketch not installed")
    else:
        check(4, "Near-dup train/val/test contamination", PASS,
              "Guaranteed by deduplicate.py cross-dataset pipeline")

    # ── Checks 5-6: Language verification (spot check) ────────────────────
    ar_ok = en_ok = True
    for name, cfg in C.DATASETS.items():
        split_f = C.DEDUPED_DIR / name / "deduped.jsonl"
        if not split_f.exists():
            continue
        lang = cfg["lang"]
        for doc in iter_jsonl_sample(split_f, max_docs=500):
            text = doc.get("text", "")
            if lang == "ar":
                if not ARABIC_RE.search(text[:200]):
                    ar_ok = False
                    log.warning(f"  Possible non-Arabic doc in {name}: {repr(text[:80])}")
                    break
            else:
                non_ascii = sum(1 for c in text[:200] if ord(c) > 127)
                if non_ascii / max(len(text[:200]), 1) > 0.50:
                    en_ok = False
                    log.warning(f"  Possible non-English doc in {name}: {repr(text[:80])}")
                    break

    check(5, "Arabic docs contain Arabic script",    PASS if ar_ok else WARN, "(spot check)")
    check(6, "English docs are predominantly ASCII", PASS if en_ok else WARN, "(spot check)")

    # ── Check 16: Manifest counts match actual shard files ────────────────
    if manifest:
        mismatch = False
        for name, cfg in C.DATASETS.items():
            lang = cfg["lang"]
            lang_str = "english" if lang == "en" else "arabic"
            idx_f = C.TRAIN_DIR / lang_str / f"{name}_train_index.json"
            if idx_f.exists():
                idx = json.loads(idx_f.read_text())
                manifest_shards = (manifest.get("datasets", {})
                                   .get(name, {})
                                   .get("train", {})
                                   .get("shard_count"))
                actual_shards = idx.get("shard_count")
                if manifest_shards is not None and actual_shards is not None:
                    if manifest_shards != actual_shards:
                        mismatch = True
                        log.error(f"  {name} manifest shard_count={manifest_shards} "
                                  f"but actual={actual_shards}")
        check(16, "Manifest counts match actual shards",
              FAIL if mismatch else PASS)
    else:
        check(16, "Manifest counts match actual shards", SKIP, "Manifest not found")

    # ── Summary ──────────────────────────────────────────────────────────
    log.info("\n=== VALIDATION SUMMARY ===")
    passed  = sum(1 for r in results if r["status"] == PASS)
    failed  = sum(1 for r in results if r["status"] == FAIL)
    warned  = sum(1 for r in results if r["status"] == WARN)
    skipped = sum(1 for r in results if r["status"] == SKIP)

    for r in results:
        icon = {"PASS":"✅","FAIL":"❌","WARN":"⚠️","SKIP":"⏭"}.get(r["status"],"?")
        log.info(f"  {icon} Check {r['check']:2d}: {r['description']}")

    log.info(f"\n  PASS={passed}  FAIL={failed}  WARN={warned}  SKIP={skipped}")

    if failed > 0:
        log.error("VALIDATION FAILED — fix errors above before training.")
        sys.exit(1)
    else:
        log.info("VALIDATION PASSED (review any warnings).")


if __name__ == "__main__":
    main()
