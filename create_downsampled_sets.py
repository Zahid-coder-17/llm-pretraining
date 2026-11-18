#!/usr/bin/env python3
"""
create_downsampled_sets.py — Stage 10: Create deterministic training subsets.

Fractions: 100%, 50%, 25%, 10%, 5%
Maintains 50/50 Arabic-English ratio.
Does NOT change validation/test sets.
Outputs manifests only (points to existing shards via index, does not copy data).

Run:  python create_downsampled_sets.py
"""
import hashlib, json, logging, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config as C

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("create_downsampled_sets.log")]
)
log = logging.getLogger(__name__)


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def get_shard_index(name: str, lang: str) -> dict:
    lang_str = "english" if lang == "en" else "arabic"
    idx_f    = C.TRAIN_DIR / lang_str / f"{name}_train_index.json"
    return json.loads(idx_f.read_text()) if idx_f.exists() else {}


def load_manifest() -> dict:
    mf = C.MANIFESTS_DIR / "dataset_manifest.json"
    return json.loads(mf.read_text()) if mf.exists() else {}


def create_subset_manifest(fraction: float, source_manifest: dict):
    pct  = int(fraction * 100)
    name = f"subset_{pct:03d}pct"
    out_f = C.MANIFESTS_DIR / f"subset_manifest_{pct:03d}pct.json"

    if out_f.exists():
        log.info(f"[{name}] Subset manifest already exists — skipping.")
        return

    rng = random.Random(C.SEED + int(pct))   # deterministic per fraction

    source_hash = sha256_str(json.dumps(source_manifest, sort_keys=True))
    target_total = 500_000_000 * fraction
    target_en    = target_total / 2
    target_ar    = target_total / 2

    subset = {
        "fraction":           fraction,
        "target_train_tokens": int(target_total),
        "target_en_tokens":   int(target_en),
        "target_ar_tokens":   int(target_ar),
        "sampling_seed":      C.SEED + int(pct),
        "source_manifest_hash": source_hash,
        "note": "Shard files are shared with 100% corpus. Use shard_indices below for subset.",
        "datasets": {},
    }

    for name_ds, cfg in C.DATASETS.items():
        lang   = cfg["lang"]
        idx    = get_shard_index(name_ds, lang)
        shards = idx.get("shards", [])

        if not shards:
            subset["datasets"][name_ds] = {"status": "no_shards"}
            continue

        total_seqs   = idx.get("total_sequences", 0)
        subset_seqs  = max(1, int(total_seqs * fraction))
        subset_toks  = subset_seqs * C.SEQ_LEN

        # Pick shards deterministically to cover subset_seqs
        all_shard_names = [s["file"] for s in shards]
        rng.shuffle(all_shard_names)
        selected_shards = []
        covered = 0
        for sf in all_shard_names:
            selected_shards.append(sf)
            # find seq count for this shard
            for sh in shards:
                if sh["file"] == sf:
                    covered += sh.get("sequences", 0)
                    break
            if covered >= subset_seqs:
                break

        subset["datasets"][name_ds] = {
            "language":        lang,
            "total_sequences": covered,
            "total_tokens":    covered * C.SEQ_LEN,
            "selected_shards": selected_shards,
        }

    out_f.write_text(json.dumps(subset, indent=2), encoding="utf-8")
    log.info(f"Subset {pct}%: target {int(target_total)/1e6:.0f}M tokens "
             f"— manifest saved to {out_f.name}")


def main():
    C.ensure_dirs()
    C.MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

    source = load_manifest()
    if not source:
        log.warning("No dataset_manifest.json found. Subset manifests will have limited info.")
        log.warning("Run build_manifest.py after pack_sequences.py.")

    for frac in C.DOWNSAMPLING_FRACTIONS:
        create_subset_manifest(frac, source)

    log.info("All downsampling manifests created.")
    log.info("NOTE: These manifests point to existing shards — no data is copied.")
    log.info("Your training script should read shard list from the appropriate subset manifest.")


if __name__ == "__main__":
    main()
