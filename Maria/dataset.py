#!/usr/bin/env python3
"""
dataset.py — Domain-Weighted Parquet Streamer for Maria 1B (50B Budget).

Composition (50B Target):
  - General:      38.0% (19.0B tokens)
  - Education:    22.0% (11.0B tokens)
  - Code:         20.0% (10.0B tokens)
  - STEM/Math:    12.0% ( 6.0B tokens)
  - Multilingual:  8.0% ( 4.0B tokens)
"""
import glob
import os
import random
from pathlib import Path
from typing import Iterator, List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
import pyarrow.parquet as pq


DOMAIN_SPECS = {
    "general":      {"prefix": "general_",      "weight": 0.38},  # 19B
    "edu":          {"prefix": "edu_",          "weight": 0.22},  # 11B
    "code":         {"prefix": "code_",         "weight": 0.20},  # 10B
    "science_math": {"prefix": "science_math_", "weight": 0.12},  # 6B
    "multilingual": {"prefix": "multilingual_", "weight": 0.08},  # 4B
}


class DomainWeightedDataset(IterableDataset):
    def __init__(
        self,
        data_dir: Path,
        seq_len: int = 8192,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0

        self.domain_files: Dict[str, List[str]] = {}
        for dom, spec in DOMAIN_SPECS.items():
            pattern = str(self.data_dir / f"{spec['prefix']}*.parquet")
            files = sorted(glob.glob(pattern))
            if files:
                self.domain_files[dom] = files

        self.domains = list(self.domain_files.keys())
        self.weights = np.array([DOMAIN_SPECS[d]["weight"] for d in self.domains], dtype=np.float64)
        self.weights = self.weights / self.weights.sum()

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        worker_seed = self.seed + (self.rank * 100) + worker_id + (self.epoch * 10000)
        rng = random.Random(worker_seed)
        np_rng = np.random.default_rng(worker_seed)

        worker_file_iters = {}
        for dom, files in self.domain_files.items():
            total_shards = self.world_size * num_workers
            shard_id = (self.rank * num_workers) + worker_id
            assigned = files[shard_id :: total_shards]
            if not assigned:
                assigned = files
            shuffled = list(assigned)
            rng.shuffle(shuffled)
            worker_file_iters[dom] = iter(shuffled)

        buffers: Dict[str, np.ndarray] = {d: np.empty(0, dtype=np.int64) for d in self.domains}

        while True:
            dom = np_rng.choice(self.domains, p=self.weights)
            buf = buffers[dom]

            while len(buf) < self.seq_len:
                try:
                    fpath = next(worker_file_iters[dom])
                except StopIteration:
                    shuffled = list(self.domain_files[dom])
                    rng.shuffle(shuffled)
                    worker_file_iters[dom] = iter(shuffled)
                    fpath = next(worker_file_iters[dom])

                try:
                    # Stream in 250,000 token slices (~2MB) for sub-millisecond network reads
                    pf = pq.ParquetFile(fpath)
                    for batch in pf.iter_batches(batch_size=250000, columns=["input_ids"]):
                        arr = batch.column(0).to_numpy()
                        buf = np.concatenate([buf, arr]) if len(buf) > 0 else arr
                        if len(buf) >= self.seq_len * 4:
                            break
                except Exception:
                    continue

            seq_tokens = buf[: self.seq_len]
            buffers[dom] = buf[self.seq_len :]

            tensor = torch.from_numpy(seq_tokens.copy())
            yield tensor, tensor.clone()


def build_dataloader(
    data_dir: Path,
    batch_size: int,
    seq_len: int = 8192,
    rank: int = 0,
    world_size: int = 1,
    num_workers: int = 4,
    seed: int = 42,
) -> DataLoader:
    dataset = DomainWeightedDataset(
        data_dir=data_dir,
        seq_len=seq_len,
        rank=rank,
        world_size=world_size,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=8,
        persistent_workers=True,
        drop_last=True,
    )
