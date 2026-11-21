"""Train a Qwen3-style byte-level BPE from scratch with HF `tokenizers` (Phase 1 quick win, Level 2 tokenizers).

Same pipeline as Qwen3 (NFC -> Split(regex, Isolated) -> ByteLevel(use_regex=False)), only the split
regex differs:
  --pattern original    Qwen3's letters-only pattern
  --pattern mark_aware  same with `\\p{L}+` -> `[\\p{L}\\p{M}]+`

Training files are mixed by character budget so both variants see identical text.

Usage:
  python -m tokenizer.train_bpe --pattern mark_aware --vocab-size 32000 \
      --train data/processed/train/kan_Knda.txt data/processed/train/kan_Latn.txt data/processed/train/eng_Latn.txt \
      --out artifacts/quickwin/mark_aware_32k
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

from tokenizer.extend import mark_aware_pattern

QWEN3_PATTERN = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
                 r"|\s*[\r\n]+|\s+(?!\S)|\s+")
PATTERNS = {"original": QWEN3_PATTERN, "mark_aware": mark_aware_pattern(QWEN3_PATTERN)}


def build(pattern: str) -> Tokenizer:
    tk = Tokenizer(models.BPE())
    tk.normalizer = normalizers.NFC()
    tk.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(pattern), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tk.decoder = decoders.ByteLevel()
    return tk


def iter_budget(files: list[str], max_chars_per_file: int | None):
    for f in files:
        n = 0
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                yield line
                n += len(line)
                if max_chars_per_file and n >= max_chars_per_file:
                    break


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", choices=list(PATTERNS), required=True)
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--max-chars-per-file", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tk = build(PATTERNS[a.pattern])
    trainer = trainers.BpeTrainer(vocab_size=a.vocab_size, min_frequency=2, show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=[])
    tk.train_from_iterator(iter_budget(a.train, a.max_chars_per_file), trainer=trainer)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tk.save(str(out / "tokenizer.json"))
    (out / "train_config.json").write_text(json.dumps(dict(vars(a), regex=PATTERNS[a.pattern],
                                                           real_vocab_size=tk.get_vocab_size()), indent=1))
    print("wrote", out, "vocab", tk.get_vocab_size())


if __name__ == "__main__":
    main()
