"""List Kannada aksharas that a tokenizer splits (failure analysis input).

Usage: python -m eval.level1.akshara_failures --tokenizer artifacts/ext/l12z_16k/tokenizer.json \
         --files data/raw/test/flores_devtest_kan_Knda.txt data/raw/test/in22gen_kan_Knda.txt \
         --out results/akshara_failures_l12z_16k.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import Counter
from pathlib import Path

from tokenizer.akshara import kannada_akshara_byte_spans
from tokenizer.registry import load


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = load(a.tokenizer)
    rep = dict(tokenizer=tok.name, vocab_size=tok.vocab_size, date=dt.date.today().isoformat(), files={})
    for f in a.files:
        bad, n, examples = Counter(), 0, {}
        for s in Path(f).read_text(encoding="utf-8").splitlines():
            s = tok.normalize(s)
            pieces = tok.token_bytes(s)
            b, p = set(), 0
            for x in pieces[:-1]:
                p += len(x)
                b.add(p)
            raw = s.encode()
            for st, en in kannada_akshara_byte_spans(s):
                n += 1
                if any(st < x < en for x in b):
                    ak = raw[st:en].decode()
                    bad[ak] += 1
                    examples.setdefault(ak, s[:80])
        show = lambda k: k.replace("‌", "<ZWNJ>").replace("‍", "<ZWJ>")
        rep["files"][Path(f).name] = dict(
            multi_codepoint_aksharas=n, split=sum(bad.values()), split_rate=sum(bad.values()) / max(1, n),
            with_zw=sum(c for k, c in bad.items() if "‌" in k or "‍" in k),
            top=[dict(akshara=show(k), count=c, context=examples[k]) for k, c in bad.most_common(40)])
        print(Path(f).name, rep["files"][Path(f).name]["split"], "/", n)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
