#!/usr/bin/env python3
"""
download_data.py ? Stage 1: Download raw data from HuggingFace.

Uses direct parquet file download via huggingface_hub (avoids datasets streaming
pickle incompatibility with Python 3.14).

Saves as JSONL to E:/pretrain_data/raw/<dataset_name>/raw.jsonl
"""
import argparse, hashlib, json, logging, os, sys, time
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download, list_repo_tree

sys.path.insert(0, str(Path(__file__).parent))
import config as C

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("download_data.log")]
)
log = logging.getLogger(__name__)
api = HfApi()


def list_parquet_files(repo_id: str, config_name: str = None) -> list:
    """List parquet files in a HuggingFace dataset repository."""
    try:
        # Try to find parquet files in the data/ directory
        prefix = f"data/{config_name}/" if config_name else "data/"
        files = []
        for item in api.list_repo_tree(repo_id, repo_type="dataset", recursive=True):
            path = item.path if hasattr(item, 'path') else str(item)
            if path.endswith(".parquet") and ("train" in path or "data" in path):
                if config_name and config_name not in path:
                    continue
                files.append(path)
        if not files:
            # Try root level
            for item in api.list_repo_tree(repo_id, repo_type="dataset"):
                path = item.path if hasattr(item, 'path') else str(item)
                if path.endswith(".parquet"):
                    files.append(path)
        return sorted(files)[:50]  # cap at 50 files to avoid over-downloading
    except Exception as e:
        log.warning(f"list_repo_tree failed: {e}")
        return []


def download_via_datasets_api(name: str, cfg: dict, out_dir: Path,
                               target_chars: float) -> tuple[int, int]:
    """
    Download using HF datasets API directly (streaming, Python 3.14 compatible).
    Falls back to pyarrow parquet approach if datasets fails.
    """
    # Try datasets library first (updated version may work)
    try:
        import datasets as hf_datasets
        hf_datasets.disable_progress_bar()

        kwargs = dict(trust_remote_code=True)
        if cfg["hf_name"]:
            ds = hf_datasets.load_dataset(
                cfg["hf_path"], cfg["hf_name"],
                split=cfg["hf_split"], streaming=True, **kwargs)
        else:
            ds = hf_datasets.load_dataset(
                cfg["hf_path"], split=cfg["hf_split"],
                streaming=True, **kwargs)

        log.info(f"[{name}] Using datasets streaming API")
        return _write_from_iterable(name, cfg, ds, out_dir, target_chars)

    except Exception as e:
        log.warning(f"[{name}] datasets streaming failed ({e}). Trying parquet download...")
        return download_via_parquet(name, cfg, out_dir, target_chars)


def download_via_parquet(name: str, cfg: dict, out_dir: Path,
                          target_chars: float) -> tuple[int, int]:
    """Download parquet files directly and stream rows from them."""
    import tempfile, requests

    log.info(f"[{name}] Listing parquet files on {cfg['hf_path']}...")
    parquet_files = list_parquet_files(cfg["hf_path"], cfg.get("hf_name"))

    if not parquet_files:
        log.error(f"[{name}] No parquet files found on {cfg['hf_path']}. "
                  f"Dataset may require authentication.")
        log.error(f"[{name}] Run: huggingface-cli login  then retry.")
        return 0, 0

    log.info(f"[{name}] Found {len(parquet_files)} parquet files")
    text_field = cfg["text_field"]
    tmp_cache = out_dir / "parquet_cache"
    tmp_cache.mkdir(exist_ok=True)

    doc_count = 0
    raw_chars = 0
    tmp_out = out_dir / "raw.jsonl.tmp"

    with open(tmp_out, "w", encoding="utf-8") as fout:
        for pf in parquet_files:
            if raw_chars >= target_chars:
                break
            try:
                local = hf_hub_download(
                    repo_id=cfg["hf_path"],
                    filename=pf,
                    repo_type="dataset",
                    local_dir=str(tmp_cache),
                )
                table = pq.read_table(local, columns=[text_field])
                for i, row in enumerate(table.to_pydict()[text_field]):
                    if not isinstance(row, str) or not row.strip():
                        continue
                    record = {
                        "text": row, "source": name,
                        "lang": cfg["lang"],
                        "doc_id": f"{name}_{doc_count:010d}",
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    doc_count += 1
                    raw_chars += len(row)
                    if raw_chars >= target_chars:
                        break

                log.info(f"[{name}] {pf}: {doc_count:,} docs, {raw_chars/1e9:.3f}B chars")
                # Delete local parquet after reading to save disk
                Path(local).unlink(missing_ok=True)

            except Exception as e:
                log.warning(f"[{name}] Skipping {pf}: {e}")
                continue

    # Cleanup cache dir
    import shutil
    shutil.rmtree(tmp_cache, ignore_errors=True)
    return doc_count, raw_chars


def _write_from_iterable(name, cfg, iterable, out_dir, target_chars):
    text_field = cfg["text_field"]
    doc_count = raw_chars = 0
    tmp = out_dir / "raw.jsonl.tmp"
    t0 = time.time()
    with open(tmp, "w", encoding="utf-8") as f:
        for i, row in enumerate(iterable):
            text = row.get(text_field, "") if isinstance(row, dict) else ""
            if not isinstance(text, str) or not text.strip():
                continue
            record = {
                "text": text, "source": name,
                "lang": cfg["lang"],
                "doc_id": f"{name}_{i:010d}",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            doc_count += 1
            raw_chars += len(text)
            if doc_count % 100_000 == 0:
                log.info(f"[{name}] {doc_count:,} docs | {raw_chars/1e9:.3f}B chars "
                         f"| {time.time()-t0:.0f}s")
            if raw_chars >= target_chars:
                break
    return doc_count, raw_chars


def download_dataset(name: str, cfg: dict):
    out_dir = C.RAW_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file  = out_dir / "raw.jsonl"
    meta_file = out_dir / "meta.json"

    if out_file.exists() and meta_file.exists():
        log.info(f"[{name}] Already downloaded ? skipping.")
        return

    log.info(f"[{name}] Downloading from {cfg['hf_path']} ...")
    t0 = time.time()

    # Target char count (2.5x buffer for dedup losses)
    target_chars = (cfg["train_tokens"] + cfg["val_tokens"] + cfg["test_tokens"]) \
                   * C.CHARS_PER_TOKEN[cfg["lang"]] * 2.5

    doc_count, raw_chars = download_via_datasets_api(name, cfg, out_dir, target_chars)

    if doc_count == 0:
        log.error(f"[{name}] No documents downloaded. Check authentication.")
        return

    # Rename tmp to final
    tmp = out_dir / "raw.jsonl.tmp"
    if tmp.exists():
        tmp.rename(out_file)

    elapsed = time.time() - t0
    meta = {
        "dataset":       name, "hf_path": cfg["hf_path"],
        "hf_name":       cfg["hf_name"], "hf_split": cfg["hf_split"],
        "download_date": datetime.now(timezone.utc).isoformat(),
        "raw_documents": doc_count, "raw_chars": raw_chars,
        "elapsed_seconds": round(elapsed, 1),
    }
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info(f"[{name}] Done: {doc_count:,} docs, {raw_chars/1e9:.3f}B chars in {elapsed:.0f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args()
    C.ensure_dirs()

    targets = {args.dataset: C.DATASETS[args.dataset]} if args.dataset else C.DATASETS
    for name, cfg in targets.items():
        download_dataset(name, cfg)

    log.info("All downloads complete.")


if __name__ == "__main__":
    main()

