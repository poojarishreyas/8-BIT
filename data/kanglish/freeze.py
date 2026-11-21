"""Freeze the hand-written Kanglish test set: validate, then record its SHA-256 in results/.

Usage: python -m data.kanglish.freeze --tsv data/kanglish/kanglish_test_v1.tsv
After freezing, any edit changes the hash; tests/test_data_hygiene.py checks it.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default="data/kanglish/kanglish_test_v1.tsv")
    ap.add_argument("--min", type=int, default=150)
    a = ap.parse_args()
    p = Path(a.tsv)
    rows = list(csv.DictReader(p.open(encoding="utf-8"), delimiter="\t"))
    filled = [r for r in rows if r["kanglish"].strip()]
    if len(filled) < a.min:
        raise SystemExit(f"only {len(filled)} rows have kanglish filled; need >= {a.min}")
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    rec = dict(file=str(p), sha256=sha, rows=len(rows), kanglish_filled=len(filled),
               romanized_filled=sum(1 for r in rows if r["kan_Latn_romanized"].strip()),
               authors=sorted({r["author"] for r in rows if r["author"]}), frozen=dt.datetime.now().isoformat(timespec="seconds"))
    Path("results").mkdir(exist_ok=True)
    Path("results/kanglish_test_frozen.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(rec)


if __name__ == "__main__":
    main()
