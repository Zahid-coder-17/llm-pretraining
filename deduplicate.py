#!/usr/bin/env python3
"""
deduplicate.py — Stage 3: Three-level deduplication.

A. Exact dedup:       SHA-256 hash per document (within each dataset)
B. Near-dup:          MinHash + LSH (Jaccard >= 0.90 = duplicate)
C. Cross-dataset:     Same as above but across all datasets jointly

Input:  data/processed/<dataset>/clean.jsonl
Output: data/deduped/<dataset>/deduped.jsonl
        data/deduped/cross_dedup_stats.json

IMPORTANT: Cross-dataset dedup means a document cannot appear in two datasets.
The pipeline processes all datasets together in cross-dedup.

Run:  python deduplicate.py
"""
import json, logging, sys, time
from pathlib import Path
from datasketch import MinHash, MinHashLSH

sys.path.insert(0, str(Path(__file__).parent))
import config as C

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("deduplicate.log")]
)
log = logging.getLogger(__name__)


def make_minhash(text: str, num_perm: int = C.MINHASH_NUM_PERM) -> MinHash:
    m = MinHash(num_perm=num_perm)
    # Use character 5-grams for language-agnostic shingles (works for Arabic + English)
    for i in range(max(1, len(text) - 4)):
        m.update(text[i:i+5].encode("utf-8"))
    return m


def iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def exact_dedup_dataset(name: str) -> dict:
    """Per-dataset exact dedup by SHA-256 hash."""
    inp     = C.PROCESSED_DIR / name / "clean.jsonl"
    out_dir = C.DEDUPED_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out     = out_dir / "exact_deduped.jsonl"

    if out.exists():
        log.info(f"[{name}] Exact dedup already done — skipping.")
        meta_f = out_dir / "exact_stats.json"
        return json.loads(meta_f.read_text()) if meta_f.exists() else {}

    seen_hashes = set()
    kept = dup = 0
    tmp = out_dir / "exact_deduped.jsonl.tmp"

    with open(tmp, "w", encoding="utf-8") as fout:
        for doc in iter_jsonl(inp):
            h = doc.get("hash", "")
            if h in seen_hashes:
                dup += 1
                continue
            seen_hashes.add(h)
            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
            kept += 1

    tmp.rename(out)
    stats = {"kept": kept, "exact_duplicates_removed": dup}
    (out_dir / "exact_stats.json").write_text(json.dumps(stats, indent=2))
    log.info(f"[{name}] Exact dedup: kept {kept:,}, removed {dup:,} duplicates")
    return stats


def near_dedup_dataset(name: str) -> dict:
    """Per-dataset near-dup via MinHash LSH."""
    inp     = C.DEDUPED_DIR / name / "exact_deduped.jsonl"
    out_dir = C.DEDUPED_DIR / name
    out     = out_dir / "near_deduped.jsonl"

    if out.exists():
        log.info(f"[{name}] Near dedup already done — skipping.")
        meta_f = out_dir / "near_stats.json"
        return json.loads(meta_f.read_text()) if meta_f.exists() else {}

    lsh  = MinHashLSH(threshold=C.NEAR_DUP_THRESHOLD, num_perm=C.MINHASH_NUM_PERM)
    kept = dup = 0
    tmp  = out_dir / "near_deduped.jsonl.tmp"

    with open(tmp, "w", encoding="utf-8") as fout:
        for doc in iter_jsonl(inp):
            key  = doc["doc_id"]
            text = doc["text"]
            m    = make_minhash(text)
            try:
                result = lsh.query(m)
                if result:
                    dup += 1
                    continue
                lsh.insert(key, m)
            except Exception:
                # If insert fails (key exists), skip
                dup += 1
                continue
            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
            kept += 1

            if kept % 50_000 == 0:
                log.info(f"[{name}] Near dedup: {kept:,} kept, {dup:,} removed so far")

    tmp.rename(out)
    stats = {"kept": kept, "near_duplicates_removed": dup}
    (out_dir / "near_stats.json").write_text(json.dumps(stats, indent=2))
    log.info(f"[{name}] Near dedup done: kept {kept:,}, removed {dup:,}")
    return stats


def cross_dataset_dedup():
    """
    Cross-dataset dedup: remove documents that appear in more than one dataset.
    Builds a single shared MinHash LSH index across all datasets.
    Documents from datasets processed earlier take priority (dict order = config order).
    """
    final_stats = {}
    done_marker = C.DEDUPED_DIR / "cross_dedup_done.json"
    if done_marker.exists():
        log.info("Cross-dataset dedup already done — skipping.")
        return

    shared_lsh   = MinHashLSH(threshold=C.NEAR_DUP_THRESHOLD, num_perm=C.MINHASH_NUM_PERM)
    shared_hashes = set()

    for name in C.DATASETS:
        inp     = C.DEDUPED_DIR / name / "near_deduped.jsonl"
        out_dir = C.DEDUPED_DIR / name
        out     = out_dir / "deduped.jsonl"

        if not inp.exists():
            log.error(f"[{name}] near_deduped.jsonl missing. Run per-dataset steps first.")
            sys.exit(1)

        kept = cross_dup = 0
        tmp = out_dir / "deduped.jsonl.tmp"

        with open(tmp, "w", encoding="utf-8") as fout:
            for doc in iter_jsonl(inp):
                h    = doc.get("hash", "")
                key  = doc["doc_id"]
                text = doc["text"]

                # Exact cross-dataset check
                if h in shared_hashes:
                    cross_dup += 1
                    continue

                # Near-dup cross-dataset check
                m = make_minhash(text)
                try:
                    result = shared_lsh.query(m)
                    if result:
                        cross_dup += 1
                        continue
                    shared_lsh.insert(key, m)
                    shared_hashes.add(h)
                except Exception:
                    cross_dup += 1
                    continue

                fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
                kept += 1

            if kept % 50_000 == 0:
                log.info(f"[{name}] Cross dedup: {kept:,} kept")

        tmp.rename(out)
        final_stats[name] = {
            "kept": kept,
            "cross_dataset_duplicates_removed": cross_dup
        }
        log.info(f"[{name}] Cross dedup: kept {kept:,}, removed {cross_dup:,}")

    done_marker.write_text(json.dumps(final_stats, indent=2), encoding="utf-8")
    (C.DEDUPED_DIR / "cross_dedup_stats.json").write_text(
        json.dumps(final_stats, indent=2), encoding="utf-8")
    log.info("Cross-dataset deduplication complete.")


def main():
    C.ensure_dirs()
    C.DEDUPED_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=== Stage A: Exact deduplication (per dataset) ===")
    for name in C.DATASETS:
        exact_dedup_dataset(name)

    log.info("=== Stage B: Near-duplicate deduplication (per dataset) ===")
    for name in C.DATASETS:
        near_dedup_dataset(name)

    log.info("=== Stage C: Cross-dataset deduplication ===")
    cross_dataset_dedup()

    log.info("All deduplication done.")


if __name__ == "__main__":
    main()
