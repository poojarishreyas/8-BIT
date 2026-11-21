"""Create the Kanglish test-set template from FLORES-200 devtest (seed 0, 200 sentences).

Output: data/kanglish/kanglish_test_template.tsv with columns
  flores_id  eng_Latn  kan_Knda  kan_Latn_romanized  kanglish  author
The team fills `kanglish` (how you'd actually type it: Kannada in English letters, English words mixed in
freely) and optionally `kan_Latn_romanized` (pure romanized Kannada). Do NOT machine-generate `kanglish`.

When done, save as data/kanglish/kanglish_test_v1.tsv and freeze it (python -m data.kanglish.freeze).
The set is test-only: never put it in training data.

Usage: python -m data.kanglish.make_template --flores data/raw/test
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flores", default="data/raw/test")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="data/kanglish/kanglish_test_template.tsv")
    a = ap.parse_args()
    d = Path(a.flores)
    eng = (d / "flores_devtest_eng_Latn.txt").read_text(encoding="utf-8").splitlines()
    kan = (d / "flores_devtest_kan_Knda.txt").read_text(encoding="utf-8").splitlines()
    assert len(eng) == len(kan)
    # prefer shorter, everyday sentences: people type those in Kanglish
    ids = sorted(i for i in range(len(eng)) if len(eng[i].split()) <= 25)
    pick = sorted(random.Random(0).sample(ids, a.n))
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; not overwriting the team's work")
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["flores_id", "eng_Latn", "kan_Knda", "kan_Latn_romanized", "kanglish", "author"])
        for i in pick:
            w.writerow([i, eng[i], kan[i], "", "", ""])
    print("wrote", out, len(pick))


if __name__ == "__main__":
    main()
