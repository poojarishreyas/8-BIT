"""Context fit (tokenizer-only): how many FLORES devtest sentences fit in a 4K / 8K / 32K token window.

Sentences are taken in order and joined with newlines; a sentence counts only if it fits completely.
Writes results/context_fit.csv (one row per tokenizer x form x window).

Usage: python -m eval.level3.context_fit [--tokenizers qwen3 artifacts/ext/std_16k/tokenizer.json artifacts/final/bigo_final/tokenizer.json]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WINDOWS = (4096, 8192, 32768)


def fit(tok, sents: list[str], window: int) -> tuple[int, int]:
    """(sentences that fit, tokens used). FLORES devtest has 1,012 sentences; the text is repeated if needed."""
    n = used = 0
    i = 0
    while True:
        s = sents[i % len(sents)]
        t = len(tok.encode(("\n" if n else "") + s))
        if used + t > window:
            return n, used
        used += t
        n += 1
        i += 1


def main() -> None:
    from tokenizer.registry import load

    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizers", nargs="+", default=["qwen3", "artifacts/ext/std_16k/tokenizer.json",
                                                        "artifacts/final/bigo_final/tokenizer.json"])
    ap.add_argument("--out", default=str(ROOT / "results/context_fit.csv"))
    a = ap.parse_args()
    forms = {"eng_Latn": ROOT / "data/raw/test/flores_devtest_eng_Latn.txt",
             "kan_Knda": ROOT / "data/raw/test/flores_devtest_kan_Knda.txt",
             "kan_Latn_xlit": ROOT / "data/processed/test/flores_devtest_kan_Latn_xlit.txt"}
    rows = []
    for spec in a.tokenizers:
        tok = load(spec if not spec.endswith(".json") else str(ROOT / spec))
        for form, p in forms.items():
            if not p.exists():
                continue
            sents = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
            for w in WINDOWS:
                n, used = fit(tok, sents, w)
                rows.append(dict(tokenizer=tok.name, form=form, window=w, sentences_fit=n, tokens_used=used,
                                 chars_fit=sum(len(sents[i % len(sents)]) for i in range(n)), split="flores_devtest",
                                 date=dt.date.today().isoformat()))
                print(tok.name, form, w, n)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)


if __name__ == "__main__":
    main()
