#!/usr/bin/env python3
"""
clean_data.py — Stage 2: Unicode normalization, language filtering, quality filtering.

Input:  data/raw/<dataset>/raw.jsonl
Output: data/processed/<dataset>/clean.jsonl
        data/processed/<dataset>/stats.json

Run:  python clean_data.py [--dataset <name>]
"""
import argparse, hashlib, json, logging, re, sys, time, unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config as C

try:
    from langdetect import detect, detect_langs, LangDetectException
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False
    logging.warning("langdetect not installed — language filtering disabled. pip install langdetect")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("clean_data.log")]
)
log = logging.getLogger(__name__)

# Arabic Unicode ranges (for script presence check)
ARABIC_RANGE = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
SPACE_RE     = re.compile(r'[ \t]+')
NEWLINE_RE   = re.compile(r'\r\n|\r')


def normalize(text: str) -> str:
    """NFC normalize, fix line endings, collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = NEWLINE_RE.sub("\n", text)
    lines = []
    for line in text.split("\n"):
        line = SPACE_RE.sub(" ", line).strip()
        lines.append(line)
    text = "\n".join(lines).strip()
    return text


def alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    alpha = sum(1 for c in text if c.isalpha())
    return alpha / len(text)


def detect_lang(text: str) -> tuple[str, float]:
    """Returns (lang_code, confidence). Returns ('unk', 0) on failure."""
    if not HAS_LANGDETECT:
        return ("unk", 1.0)
    try:
        sample = text[:2000]
        langs = detect_langs(sample)
        if langs:
            best = langs[0]
            return (best.lang, best.prob)
    except LangDetectException:
        pass
    return ("unk", 0.0)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def passes_quality(text: str, expected_lang: str) -> tuple[bool, str]:
    n = len(text)
    if n < C.MIN_DOC_CHARS:
        return False, "too_short"
    if n > C.MAX_DOC_CHARS:
        return False, "too_long"
    ar = alpha_ratio(text)
    if ar < C.MIN_ALPHA_RATIO:
        return False, "low_alpha"
    if expected_lang == "ar":
        ar_chars = len(ARABIC_RANGE.findall(text))
        if ar_chars / max(n, 1) < 0.10:
            return False, "no_arabic_script"
    return True, "ok"


def clean_dataset(name: str, cfg: dict):
    inp  = C.RAW_DIR  / name / "raw.jsonl"
    out_dir = C.PROCESSED_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out  = out_dir / "clean.jsonl"
    stats_f = out_dir / "stats.json"

    if out.exists() and stats_f.exists():
        log.info(f"[{name}] Already cleaned — skipping.")
        return

    if not inp.exists():
        log.error(f"[{name}] raw.jsonl not found. Run download_data.py first.")
        sys.exit(1)

    log.info(f"[{name}] Cleaning...")
    t0 = time.time()
    expected_lang = cfg["lang"]

    counts = dict(raw=0, too_short=0, too_long=0, low_alpha=0,
                  no_arabic_script=0, wrong_lang=0, kept=0)
    tmp = out_dir / "clean.jsonl.tmp"

    with open(inp, "r", encoding="utf-8") as fin, \
         open(tmp, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue

            counts["raw"] += 1
            text = doc.get("text", "")

            # 1. Normalize
            text = normalize(text)
            if not text:
                counts["too_short"] += 1
                continue

            # 2. Quality filter
            ok, reason = passes_quality(text, expected_lang)
            if not ok:
                counts[reason] = counts.get(reason, 0) + 1
                continue

            # 3. Language filter
            if HAS_LANGDETECT and C.LANG_CONF_THRESHOLD > 0:
                detected, conf = detect_lang(text)
                # map langdetect codes to ours
                lang_map = {"ar": "ar", "en": "en"}
                detected_mapped = lang_map.get(detected, detected)
                if detected_mapped != expected_lang or conf < C.LANG_CONF_THRESHOLD:
                    counts["wrong_lang"] += 1
                    continue

            # 4. Compute hash of normalized text
            h = sha256(text)
            doc["text"]  = text
            doc["hash"]  = h
            doc["nchars"]= len(text)
            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
            counts["kept"] += 1

            if counts["kept"] % 50_000 == 0:
                log.info(f"[{name}] kept {counts['kept']:,} / {counts['raw']:,} raw docs")

    tmp.rename(out)
    elapsed = time.time() - t0
    counts["elapsed_seconds"] = round(elapsed, 1)
    stats_f.write_text(json.dumps(counts, indent=2), encoding="utf-8")
    log.info(f"[{name}] Done: kept {counts['kept']:,} / {counts['raw']:,} docs in {elapsed:.0f}s")
    log.info(f"[{name}] Rejection breakdown: {counts}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args()
    C.ensure_dirs()

    targets = {args.dataset: C.DATASETS[args.dataset]} if args.dataset else C.DATASETS
    for name, cfg in targets.items():
        clean_dataset(name, cfg)


if __name__ == "__main__":
    main()
