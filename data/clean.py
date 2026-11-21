"""Remove test-set text (FLORES-200 dev/devtest, Belebele passages, IN22-Gen) from training data, and dedupe.

A training document is dropped if it shares any word 8-gram with any test text (in any language;
grams are taken from all test files). Exact duplicate documents are also dropped.
Writes cleaned files plus a JSON report to results/.

Usage: python -m data.clean --train data/raw/train_raw --test data/raw/test data/processed/test \
                            --out data/processed/train
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

N = 8


def grams(text: str, n: int = N):
    w = text.split()
    for i in range(len(w) - n + 1):
        yield hash(" ".join(w[i:i + n]))


def test_texts(dirs: list[Path]):
    for d in dirs:
        for p in sorted(d.glob("*")):
            if p.suffix == ".txt":
                yield from p.read_text(encoding="utf-8").splitlines()
            elif p.suffix == ".jsonl":
                for l in p.read_text(encoding="utf-8").splitlines():
                    ex = json.loads(l)
                    yield from (v for v in ex.values() if isinstance(v, str) and len(v) > 40)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/raw/train_raw")
    ap.add_argument("--test", nargs="+", default=["data/raw/test", "data/processed/test"])
    ap.add_argument("--out", default="data/processed/train")
    ap.add_argument("--report", default="results/data_clean_report.json")
    a = ap.parse_args()

    test_dirs = [Path(d) for d in a.test if Path(d).exists()]
    bad = set()
    n_test = 0
    for t in test_texts(test_dirs):
        n_test += 1
        bad.update(grams(t))
    print(f"{n_test} test texts, {len(bad):,} test 8-grams from {[str(d) for d in test_dirs]}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = dict(date=dt.date.today().isoformat(), ngram=N, test_dirs=[str(d) for d in test_dirs],
                  test_files=sorted(p.name for d in test_dirs for p in d.glob("*") if p.suffix in (".txt", ".jsonl")),
                  files={})
    for p in sorted(Path(a.train).glob("*.txt")):
        seen = set()
        kept = dropped_test = dropped_dup = chars = 0
        with p.open(encoding="utf-8") as fi, (out / p.name).open("w", encoding="utf-8") as fo:
            for line in fi:
                h = hashlib.md5(line.encode()).digest()
                if h in seen:
                    dropped_dup += 1
                    continue
                seen.add(h)
                if any(g in bad for g in grams(line)):
                    dropped_test += 1
                    continue
                fo.write(line)
                kept += 1
                chars += len(line) - 1
        report["files"][p.name] = dict(kept=kept, dropped_test_overlap=dropped_test, dropped_dup=dropped_dup, chars=chars)
        print(p.name, report["files"][p.name])
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(report, indent=1), encoding="utf-8")
    if not any("flores" in f for f in report["test_files"]):
        print("WARNING: no FLORES files in test dirs yet - rerun after `python -m data.download --only flores in22`")


if __name__ == "__main__":
    main()
