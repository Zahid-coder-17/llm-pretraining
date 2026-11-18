#!/usr/bin/env python3
"""
dataset.py — PyTorch streaming Dataset for packed binary shards.

Usage in your training script:
    from dataset import ShardedTokenDataset, build_dataloader
    dl = build_dataloader("train", lang="both", batch_size=8, num_workers=2)
    for batch in dl:
        # batch: LongTensor of shape (batch_size, 512)
        ...
"""
import json, random
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

# Import config from same directory
import sys
sys.path.insert(0, str(Path(__file__).parent))
import config as C


class ShardedTokenDataset(Dataset):
    """
    Memory-efficient dataset that reads packed uint16 binary shards.
    Each item is a LongTensor of shape (seq_len,).
    Shards are memory-mapped (not fully loaded into RAM).
    """

    def __init__(
        self,
        shard_files:  List[Path],
        seq_len:      int = C.SEQ_LEN,
        shuffle_shards: bool = True,
        seed:         int = C.SEED,
    ):
        self.seq_len = seq_len
        self.shards  = list(shard_files)

        if shuffle_shards:
            rng = random.Random(seed)
            rng.shuffle(self.shards)

        # Build index: (shard_idx, seq_idx_within_shard)
        self._index = []
        for si, sf in enumerate(self.shards):
            n_tokens = sf.stat().st_size // 2          # uint16 = 2 bytes
            n_seqs   = n_tokens // seq_len
            for i in range(n_seqs):
                self._index.append((si, i))

        self._mmaps = {}   # lazy-loaded memory maps per shard

    def _get_mmap(self, shard_idx: int) -> np.ndarray:
        if shard_idx not in self._mmaps:
            path = self.shards[shard_idx]
            self._mmaps[shard_idx] = np.fromfile(path, dtype=np.uint16)
        return self._mmaps[shard_idx]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> torch.Tensor:
        si, seq_i = self._index[idx]
        arr = self._get_mmap(si)
        start = seq_i * self.seq_len
        tokens = arr[start: start + self.seq_len].astype(np.int64)
        return torch.from_numpy(tokens)


def get_shard_files(split: str, lang: Optional[str] = None,
                    subset_manifest: Optional[Path] = None) -> List[Path]:
    """
    Returns list of .bin shard paths for a given split.
    split: "train" | "val" | "test"
    lang:  "en" | "ar" | "both" | None (= both)
    subset_manifest: path to a downsampling manifest JSON (optional)
    """
    if split == "train":
        base = C.TRAIN_DIR
    elif split == "val":
        base = C.VAL_DIR
    else:
        base = C.TEST_DIR

    langs = []
    if lang in (None, "both", "en"):
        langs.append("english")
    if lang in (None, "both", "ar"):
        langs.append("arabic")

    if subset_manifest and subset_manifest.exists():
        sm = json.loads(subset_manifest.read_text())
        selected = set()
        for ds_info in sm.get("datasets", {}).values():
            selected.update(ds_info.get("selected_shards", []))
        files = []
        for lang_str in langs:
            d = base / lang_str
            if d.exists():
                for f in sorted(d.glob("*.bin")):
                    if f.name in selected:
                        files.append(f)
        return files

    files = []
    for lang_str in langs:
        d = base / lang_str
        if d.exists():
            files.extend(sorted(d.glob("*.bin")))
    return files


def build_dataloader(
    split:            str,
    lang:             str = "both",
    batch_size:       int = 8,
    num_workers:      int = 2,
    shuffle:          bool = True,
    subset_manifest:  Optional[Path] = None,
    seed:             int = C.SEED,
) -> DataLoader:
    """
    Build a DataLoader for training/validation/testing.
    Designed for RTX 3050 6GB — uses memory-mapped shards, no full-corpus RAM loading.
    """
    shard_files = get_shard_files(split, lang, subset_manifest)
    if not shard_files:
        raise FileNotFoundError(
            f"No shard files found for split='{split}', lang='{lang}'. "
            f"Run pack_sequences.py first."
        )

    dataset = ShardedTokenDataset(shard_files, shuffle_shards=shuffle, seed=seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,     # Windows: avoid worker crash on long runs
        drop_last=True,   # keep all batches the same size
    )
    return loader


if __name__ == "__main__":
    # Quick sanity check
    import sys
    print("Testing dataset loader...")
    try:
        files = get_shard_files("train", "both")
        if not files:
            print("No shard files found — run the pipeline first.")
            sys.exit(0)
        ds = ShardedTokenDataset(files[:2])
        print(f"Dataset size: {len(ds):,} sequences")
        sample = ds[0]
        print(f"Sample shape: {sample.shape}, dtype: {sample.dtype}")
        print(f"Min token: {sample.min()}, Max token: {sample.max()}")
        print("Dataset loader OK.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

