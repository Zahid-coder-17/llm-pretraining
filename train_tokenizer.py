#!/usr/bin/env python3
"""
train_tokenizer.py — Stage 5: Train shared BPE tokenizer (Arabic + English).

- Samples deterministically from TRAINING splits only (not val/test)
- Trains SentencePiece BPE with vocab_size=32000
- Saves tokenizer files to data/tokenizer/

Run:  python train_tokenizer.py
"""
import json, logging, random, sys, tempfile, time
from pathlib import Path

import sentencepiece as spm

sys.path.insert(0, str(Path(__file__).parent))
import config as C

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("train_tokenizer.log")]
)
log = logging.getLogger(__name__)


def iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def sample_training_text(target_chars: int) -> Path:
    """
    Samples text from training splits (seed-deterministic) and writes
    to a temporary file for SentencePiece training.
    Returns path to the sample file.
    """
    sample_file = C.TOKENIZER_DIR / "tokenizer_training_sample.txt"
    if sample_file.exists():
        log.info("Tokenizer training sample already exists — reusing.")
        return sample_file

    rng = random.Random(C.SEED)
    total_chars = 0
    tmp = C.TOKENIZER_DIR / "tokenizer_training_sample.txt.tmp"

    # Proportion of chars from each language
    en_target = target_chars // 2
    ar_target = target_chars // 2

    en_datasets = [n for n, d in C.DATASETS.items() if d["lang"] == "en"]
    ar_datasets = [n for n, d in C.DATASETS.items() if d["lang"] == "ar"]

    with open(tmp, "w", encoding="utf-8") as fout:
        for lang, ds_list, lang_target in [("en", en_datasets, en_target),
                                            ("ar", ar_datasets, ar_target)]:
            lang_chars = 0
            per_ds_target = lang_target // max(len(ds_list), 1)

            for name in ds_list:
                train_f = C.SPLITS_DIR / name / "train.jsonl"
                if not train_f.exists():
                    log.warning(f"[{name}] train.jsonl not found — skipping in tokenizer sample")
                    continue

                ds_chars = 0
                docs = list(iter_jsonl(train_f))
                rng.shuffle(docs)

                for doc in docs:
                    if ds_chars >= per_ds_target:
                        break
                    text = doc.get("text", "")
                    fout.write(text + "\n")
                    ds_chars  += len(text)
                    lang_chars += len(text)

                log.info(f"[{name}] Sampled {ds_chars/1e6:.1f}M chars for tokenizer")
                if lang_chars >= lang_target:
                    break

        total_chars += lang_chars

    tmp.rename(sample_file)
    log.info(f"Tokenizer training sample: {total_chars/1e6:.0f}M chars")
    return sample_file


def train_tokenizer(sample_file: Path):
    model_prefix = str(C.TOKENIZER_DIR / "tokenizer")
    model_file   = C.TOKENIZER_DIR / "tokenizer.model"

    if model_file.exists():
        log.info("Tokenizer already trained — skipping.")
        return

    log.info(f"Training SentencePiece BPE tokenizer (vocab={C.VOCAB_SIZE})...")
    t0 = time.time()

    spm.SentencePieceTrainer.train(
        input=str(sample_file),
        model_prefix=model_prefix,
        vocab_size=C.VOCAB_SIZE,
        model_type="bpe",
        character_coverage=0.9999,   # high coverage for Arabic + English
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="[PAD]",
        unk_piece="[UNK]",
        bos_piece="[BOS]",
        eos_piece="[EOS]",
        normalization_rule_name="nfc",
        add_dummy_prefix=False,
        # Allow digits, punctuation, whitespace
        user_defined_symbols=[],
        shuffle_input_sentence=True,
        input_sentence_size=10_000_000,
        seed_sentencepiece_random=C.SEED,
        num_threads=4,
    )

    elapsed = time.time() - t0
    log.info(f"Tokenizer trained in {elapsed:.0f}s")

    # Verify vocab size
    sp = spm.SentencePieceProcessor(model_file=str(model_file))
    actual_vocab = sp.get_piece_size()
    if actual_vocab != C.VOCAB_SIZE:
        log.warning(f"Vocab size mismatch: expected {C.VOCAB_SIZE}, got {actual_vocab}")
    else:
        log.info(f"Vocabulary size verified: {actual_vocab}")

    # Save config
    config = {
        "model_type": "bpe",
        "vocab_size": actual_vocab,
        "model_file": "tokenizer.model",
        "pad_id": 0, "unk_id": 1, "bos_id": 2, "eos_id": 3,
        "training_chars": "see tokenizer_training_sample.txt",
        "seed": C.SEED,
        "normalization": "nfc",
    }
    (C.TOKENIZER_DIR / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8")

    # Sample token stats
    log.info("=== Tokenizer sample stats ===")
    en_sample = "The quick brown fox jumps over the lazy dog."
    ar_sample  = "تعلم الآلة هو مجال من مجالات الذكاء الاصطناعي."
    log.info(f"EN tokens: {sp.encode(en_sample)}")
    log.info(f"AR tokens: {sp.encode(ar_sample)}")


def main():
    C.ensure_dirs()
    C.TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

    sample = sample_training_text(C.TOKENIZER_SAMPLE_CHARS)
    train_tokenizer(sample)
    log.info("Tokenizer training complete.")


if __name__ == "__main__":
    main()
