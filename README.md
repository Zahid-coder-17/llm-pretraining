# LLM Pretraining: GQA vs. GQE vs. HyperGQE (Maria-1B)

<p align="center">
  <b>A Controlled Bilingual Pre-Training Pipeline & Architecture Benchmark Suite</b>
  <br>
  <i>From-Scratch Pretraining • 500M-Token Canonical Corpus • Dynamic Router Hypernetworks • Cluster Training</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Under%20Active%20Development-orange?style=flat-square&logo=git" alt="Status">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=flat-square&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-red.svg?style=flat-square&logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Corpus-500M%20Tokens-purple.svg?style=flat-square" alt="Tokens">
  <img src="https://img.shields.io/badge/Seed-42-lightgrey.svg?style=flat-square" alt="Seed">
  <img src="https://img.shields.io/badge/License-Apache%202.0-green.svg?style=flat-square" alt="License">
</p>

---

> [!WARNING]
> ### 🚧 Project Status: Under Active Development
> This repository is under active development. Dataset ingestion scripts, distributed cluster training harnesses, tokenization packing pipelines, and benchmark evaluation suites are continuously being updated.

---

## 📖 Overview

**LLM Pretraining** is a research and engineering codebase dedicated to training Large Language Models from scratch. It hosts both:
1. **A Controlled Architecture Study**: Head-to-head empirical comparison of **Grouped Query Attention (GQA)**, **Grouped Query Experts (GQE)**, and **Hypernetwork-based Grouped Query Experts (HyperGQE)** trained under identical conditions on a canonical 500M-token bilingual corpus.
2. **Maria-1B Pretraining Suite**: Industrial distributed cluster pretraining, dataset streaming, sequence packing, and SFT patching tools developed for the **Maria-1B** foundation model.

### Key Principles
* **Identical Conditioning**: All architectures are evaluated on the exact same token sequence splits, random seeds (Seed 42), and loss configurations.
* **Pure Pretraining**: Clean scratch pretraining without fine-tuning artifacts, LoRA abstractions, or unmeasured token counts.
* **Streaming & Scalable**: Designed for memory efficiency — streams datasets from source repositories and packs uint16 binary token arrays.

---

## 🚦 Features & Roadmap

| Component | Description | Status |
| :--- | :--- | :---: |
| **Data Ingestion** | Streaming downloads for FineWeb-Edu, WikiText, OpenWebText, Arabic Wikipedia, OSCAR, CulturaX | ✅ Completed |
| **Data Cleaning** | Unicode normalization, script filtering, MinHash LSH near-deduplication | ✅ Completed |
| **Tokenizer** | Shared SentencePiece BPE (32,000 vocab, joint English + Arabic) | ✅ Completed |
| **Sequence Packing** | Sharded uint16 binary sequence packager (512-token rows, incomplete blocks dropped) | ✅ Completed |
| **Architectures** | Reference implementations of GQA, GQE, and HyperGQE models | ✅ Completed |
| **Maria-1B Cluster** | Distributed multi-GPU & H100 cluster training harness (`run_cluster_train.sh`) | ✅ Completed |
| **Downsampling Sets** | Automated subset manifest generation (5%, 10%, 25%, 50%) for scale experiments | ✅ Completed |
| **Conversational SFT** | Domain reasoning, RunPod analyst pipelines, and authentic chat patchers | 🔄 In Progress |
| **Online Checkpointing** | Automated state recovery, gradient health monitoring, and live wandb tracking | 🔄 In Progress |

---

## 📚 Dataset Sources (500M Canonical Corpus)

| Dataset | Language | Target Train Tokens | Validation Tokens | Test Tokens |
| :--- | :---: | :---: | :---: | :---: |
| **FineWeb-Edu** | EN | 200M | 5M | 5M |
| **WikiText-103** | EN | 25M | 2M | 2M |
| **OpenWebText** | EN | 25M | 2M | 2M |
| **Arabic Wikipedia** | AR | 100M | 3M | 3M |
| **OSCAR Arabic** | AR | 100M | 3M | 3M |
| **CulturaX Arabic** | AR | 50M | 2M | 2M |
| **Total** | **EN + AR** | **500M Tokens** | **17M Tokens** | **17M Tokens** |

---

## 🔄 End-to-End Pipeline Execution

Run the data curation and preparation stages in sequential order:

```bash
# 1. Download raw data (streaming — does NOT load full corpus into RAM)
python download_data.py

# 2. Text normalization + language/script filtering + quality scoring
python clean_data.py

# 3. Exact dedup (SHA-256) + near-dup (MinHash/LSH, threshold=0.90)
python deduplicate.py

# 4. Document-level train/validation/test split (Seed=42)
python split_data.py

# 5. Train shared SentencePiece BPE tokenizer (32k vocab)
python train_tokenizer.py

# 6. Tokenize all splits (measures and logs ACTUAL token counts)
python tokenize_data.py

# 7. Pack into fixed 512-token shards (uint16 binary arrays)
python pack_sequences.py

# 8. Build manifests (JSON & CSV indices)
python build_manifest.py

# 9. Verify dataset integrity against 16 structural checks
python validate_dataset.py

# 10. Generate downsampled subsets (50%, 25%, 10%, 5%)
python create_downsampled_sets.py
```

---

## 💻 Training with PyTorch Dataloaders

```python
from dataset import build_dataloader
from pathlib import Path

# Build dataloader for the full bilingual training corpus
train_dl = build_dataloader("train", lang="both", batch_size=8, shuffle=True)
val_dl   = build_dataloader("val",   lang="both", batch_size=8, shuffle=False)

# Or load from an exact downsampled experiment manifest (e.g. 50% scale)
subset_manifest = Path("data/manifests/subset_manifest_050pct.json")
train_dl_50 = build_dataloader("train", lang="both", batch_size=8, subset_manifest=subset_manifest)

for batch in train_dl:
    input_ids = batch["input_ids"]  # torch.LongTensor (batch_size, 512)
    # Forward pass
```

---

## 📁 Repository Structure

```
llm-pretraining/
├── Maria/                               # Maria-1B Foundation Pretraining Suite
│   ├── models/                          # HyperGQE model definitions
│   ├── evaluation_graphs/               # MMLU & multi-domain benchmark curves
│   ├── runpod_sft_patch/                # Post-pretraining conversational patches
│   ├── config.py                        # Hyperparameters & cluster configs
│   ├── train.py                         # Single-node training script
│   ├── remote_train.py                  # Distributed multi-GPU cluster engine
│   └── run_cluster_train.sh             # H100 / A100 cluster launcher
├── models/                              # Comparative Architecture Implementations
│   ├── gqa/                             # Grouped Query Attention
│   ├── gqe/                             # Grouped Query Experts
│   └── hyper_gqe/                       # HyperGQE implementation
├── download_data.py                     # Hugging Face streaming downloader
├── clean_data.py                        # Quality filter & normalization
├── deduplicate.py                       # Exact & MinHash LSH deduplication
├── split_data.py                        # Split manager (Seed 42)
├── train_tokenizer.py                   # SentencePiece 32k trainer
├── tokenize_data.py                     # Split tokenizer & token auditor
├── pack_sequences.py                    # Binary uint16 array packer
├── build_manifest.py                    # Shard manifest builder
├── validate_dataset.py                  # 16-point validation harness
├── create_downsampled_sets.py           # Multi-scale experiment manifests
├── requirements.txt                     # Python dependencies
└── README.md                            # Project documentation
```

---

## 🛡️ Reproducibility Guarantee

* **Strict Determinism**: Master random seed `42` pinned across all dataset splitters, initializations, and shufflers.
* **Document Boundary Integrity**: Train/val/test splits occur strictly at document boundaries to eliminate cross-split data leakage.
* **No Artificial Padding**: Sequence packing drops residual tokens at end-of-shard rather than injecting artificial `<pad>` tokens into pretraining attention windows.

---

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).
