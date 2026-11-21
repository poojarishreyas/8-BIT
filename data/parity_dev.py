"""Parallel dev set for parity-aware BPE (Layer 4, from-scratch tokenizers). Not FLORES.

Samples N en-kn pairs from Samanantar (ai4bharat/samanantar, kn, shard 1), 5-40 words, drops any pair
sharing a word 8-gram with the test sets (same rule as data/clean.py). The Kannada side is romanized on
Kaggle with IndicXlit (notebooks/kaggle_romanize_xlit.ipynb) -> parity_dev_kan_Latn_xlit.txt, giving a
3-way parallel dev set: eng_Latn / kan_Knda / kan_Latn.

Usage: python -m data.parity_dev --n 2000 --out data/processed/dev
"""
from __future__ import annotations

import argparse
import random
import unicodedata
from pathlib import Path

from data.clean import grams, test_texts


def norm(s: str) -> str:
    return unicodedata.normalize("NFC", " ".join(s.replace("​", " ").split()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", default="data/processed/dev")
    ap.add_argument("--test", nargs="+", default=["data/raw/test", "data/processed/test"])
    ap.add_argument("--row-groups", type=int, default=40)
    a = ap.parse_args()

    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    bad = set()
    for t in test_texts([Path(d) for d in a.test if Path(d).exists()]):
        bad.update(grams(t))
    rows = []
    with HfFileSystem().open("datasets/ai4bharat/samanantar/kn/train-00001-of-00002.parquet", "rb") as f:
        pf = pq.ParquetFile(f)
        for g in range(min(a.row_groups, pf.metadata.num_row_groups)):
            for r in pf.read_row_group(g, columns=["src", "tgt"]).to_pylist():
                en, kn = norm(r["src"]), norm(r["tgt"])
                if 5 <= len(en.split()) <= 40 and 3 <= len(kn.split()) <= 40 and \
                        not any(x in bad for x in grams(en)) and not any(x in bad for x in grams(kn)):
                    rows.append((en, kn))
    pick = random.Random(0).sample(rows, min(a.n, len(rows)))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "parity_dev_eng_Latn.txt").write_text("\n".join(e for e, _ in pick) + "\n", encoding="utf-8")
    (out / "parity_dev_kan_Knda.txt").write_text("\n".join(k for _, k in pick) + "\n", encoding="utf-8")
    print(f"{len(rows)} candidate pairs -> {len(pick)} in {out}")


if __name__ == "__main__":
    main()
