"""
Shared configuration for the GQA/GQE/HyperGQE pre-training data pipeline.
All pipeline scripts import from here. Edit this file to change any setting.
"""
from pathlib import Path

# Paths
ROOT          = Path(__file__).parent
DATA_DIR      = Path("E:/pretrain_data")
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DEDUPED_DIR   = DATA_DIR / "deduped"
SPLITS_DIR    = DATA_DIR / "splits"
TOKENIZER_DIR = DATA_DIR / "tokenizer"
TOKENIZED_DIR = DATA_DIR / "tokenized"
MANIFESTS_DIR = DATA_DIR / "manifests"
TRAIN_DIR     = DATA_DIR / "train"
VAL_DIR       = DATA_DIR / "validation"
TEST_DIR      = DATA_DIR / "test"

SEED       = 42          # canonical data pipeline seed (fixed, never changes)
TRAIN_SEEDS = [42, 123, 456]  # 3 training seeds for mean +- std reporting
VOCAB_SIZE = 32_000
SEQ_LEN    = 512

DATASETS = {
    "fineweb_edu": {
        "lang": "en", "hf_path": "HuggingFaceFW/fineweb-edu",
        "hf_name": "sample-10BT", "hf_split": "train", "text_field": "text",
        "train_tokens": 4_500_000_000, "val_tokens": 5_000_000, "test_tokens": 5_000_000,
    },
    "wikitext103": {
        "lang": "en", "hf_path": "wikitext",
        "hf_name": "wikitext-103-raw-v1", "hf_split": "train", "text_field": "text",
        "train_tokens": 100_000_000, "val_tokens": 2_000_000, "test_tokens": 2_000_000,
    },
    "openwebtext": {
        "lang": "en", "hf_path": "openwebtext",
        "hf_name": None, "hf_split": "train", "text_field": "text",
        "train_tokens": 4_000_000_000, "val_tokens": 2_000_000, "test_tokens": 2_000_000,
    },
    
    
    "culturax_arabic": {
        "lang": "ar", "hf_path": "uonlp/CulturaX",
        "hf_name": "ar", "hf_split": "train", "text_field": "text",
        "train_tokens": 1_400_000_000, "val_tokens": 2_000_000, "test_tokens": 2_000_000,
    },
}

CHARS_PER_TOKEN     = {"en": 4.5, "ar": 3.0}
SHARD_SIZE_TOKENS   = 50_000_000   # 100 MB at uint16
MIN_DOC_CHARS       = 100
MAX_DOC_CHARS       = 500_000
MIN_ALPHA_RATIO     = 0.60
NEAR_DUP_THRESHOLD  = 0.90
MINHASH_NUM_PERM    = 64
LANG_CONF_THRESHOLD = 0.80
TOKENIZER_SAMPLE_CHARS = 500_000_000
DOWNSAMPLING_FRACTIONS = [1.0, 0.50, 0.25, 0.10, 0.05]

ALL_DIRS = [
    RAW_DIR, PROCESSED_DIR, DEDUPED_DIR, SPLITS_DIR,
    TOKENIZER_DIR, TOKENIZED_DIR, MANIFESTS_DIR,
    TRAIN_DIR / "english", TRAIN_DIR / "arabic",
    VAL_DIR   / "english", VAL_DIR   / "arabic",
    TEST_DIR  / "english", TEST_DIR  / "arabic",
]

def ensure_dirs():
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)






