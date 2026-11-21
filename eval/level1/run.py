"""Level 1 runner: tokenizers x parallel text forms -> CSV in results/.

Usage:
  python -m eval.level1.run --split flores_devtest \
      --form eng_Latn=data/raw/test/flores_devtest_eng_Latn.txt \
      --form kan_Knda=data/raw/test/flores_devtest_kan_Knda.txt \
      --ref eng_Latn --tokenizers cl100k_base o200k_base qwen3 \
      --out results/level1_baselines_flores_devtest.csv

Each --form file has one sentence per line, line i parallel across forms. Tokenizers are names from
tokenizer.registry.BASELINES or paths to a tokenizer.json.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path

from eval.level1.metrics import aggregate, sentence_stats
from tokenizer.registry import BASELINES, load


def read_lines(p: str) -> list[str]:
    return [l.rstrip("\n") for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--form", action="append", required=True, help="name=path (repeatable)")
    ap.add_argument("--ref", default="eng_Latn")
    ap.add_argument("--tokenizers", nargs="+", default=list(BASELINES))
    ap.add_argument("--split", required=True, help="data split label written to results")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    forms = dict(f.split("=", 1) for f in a.form)
    texts = {k: read_lines(v)[: a.limit] for k, v in forms.items()}
    n = {len(v) for v in texts.values()}
    if len(n) != 1:
        raise SystemExit(f"forms are not parallel: { {k: len(v) for k, v in texts.items()} }")

    rows, patterns = [], {}
    today = dt.date.today().isoformat()
    for name in a.tokenizers:
        tok = load(name)
        patterns[tok.name] = tok.pattern
        stats = {f: [sentence_stats(tok, s) for s in ss] for f, ss in texts.items()}
        ref = stats.get(a.ref)
        for f, st in stats.items():
            agg = aggregate(st, ref if f != a.ref else None)
            rows.append(dict(tokenizer=tok.name, vocab_size=tok.vocab_size, form=f, split=a.split,
                             seed="", date=today, source=forms[f], **agg))
            print(f"{tok.name:>16} {f:>10}  tok/sent={agg['tokens_per_sent']:7.2f}  "
                  f"parity={agg.get('parity', 1.0):5.2f}  fert={agg['fertility']:5.2f}  "
                  f"akshara_split={agg['akshara_split_rate'] if agg['akshara_split_rate'] is not None else float('nan'):.3f}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    out.with_suffix(".patterns.json").write_text(json.dumps(patterns, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
