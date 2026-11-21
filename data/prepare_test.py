"""Build parallel, one-item-per-line test files from downloaded test sets.

  belebele -> <out>/belebele_passages_{lang}.txt  (unique FLORES passages, parallel across langs)

Belebele passages ARE FLORES text: they are test-only and also go into the FLORES exclusion
list used by data/clean.py.

Usage: python -m data.prepare_test --raw data/raw/test --out data/processed/test
"""
from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

LANGS = ("eng_Latn", "kan_Knda")


def belebele_passages(raw: Path, lang: str) -> list[tuple[str, str]]:
    seen, out = set(), []
    for line in (raw / f"belebele_{lang}.jsonl").read_text(encoding="utf-8").splitlines():
        ex = json.loads(line)
        key = (ex["link"], ex["question_number"])  # row identity is shared across languages
        out.append((key, unicodedata.normalize("NFC", " ".join(ex["flores_passage"].split()))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/test")
    ap.add_argument("--out", default="data/processed/test")
    a = ap.parse_args()
    raw, out = Path(a.raw), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = {l: belebele_passages(raw, l) for l in LANGS}
    keys = [k for k, _ in rows["eng_Latn"]]
    assert all([k for k, _ in rows[l]] == keys for l in LANGS), "belebele rows not aligned"
    keep, seen = [], set()
    for i, (_, p) in enumerate(rows["eng_Latn"]):  # one row per unique passage
        if p not in seen:
            seen.add(p)
            keep.append(i)
    for l in LANGS:
        (out / f"belebele_passages_{l}.txt").write_text("\n".join(rows[l][i][1] for i in keep) + "\n", encoding="utf-8")
    print(f"belebele: {len(keys)} questions -> {len(keep)} unique passages")


if __name__ == "__main__":
    main()
