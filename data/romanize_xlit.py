"""Romanize Kannada test sentences with AI4Bharat IndicXlit (Kannada script -> Latin), line by line.

Runs on Kaggle (fairseq won't build on Windows without MSVC). For each input file *_kan_Knda.txt,
writes *_kan_Latn_xlit.txt next to it in --out (same line count, so it stays parallel).
Also writes results/romanize_xlit_<date>.json with counts and timings.

Usage:
  python -m data.romanize_xlit --inputs /kaggle/input/bigo-test/flores_devtest_kan_Knda.txt ... \
      --out /kaggle/working/romanized --beam 4
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
import unicodedata
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--results", default="results")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    from ai4bharat.transliteration import XlitEngine

    eng = XlitEngine(src_script_type="indic", beam_width=a.beam, rescore=False)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = dict(tool="ai4bharat-transliteration (IndicXlit)", direction="kn->en", beam=a.beam,
               date=dt.date.today().isoformat(), files={})
    for inp in a.inputs:
        p = Path(inp)
        lines = p.read_text(encoding="utf-8").splitlines()[: a.limit]
        t0 = time.perf_counter()
        res, failed = [], 0
        for i, line in enumerate(lines):
            line = unicodedata.normalize("NFC", line)
            try:
                r = eng.translit_sentence(line, lang_code="kn")
                r = r if isinstance(r, str) else r.get("kn", line)
            except Exception:
                failed += 1
                r = line  # keep line count; counted as failure
            res.append(" ".join(r.split()))
            if i % 200 == 0:
                print(p.name, i, "|", line[:50], "->", res[-1][:60], flush=True)
        dst = out / p.name.replace("kan_Knda", "kan_Latn_xlit")
        dst.write_text("\n".join(res) + "\n", encoding="utf-8")
        rep["files"][p.name] = dict(lines=len(lines), failed=failed, seconds=round(time.perf_counter() - t0, 1),
                                   output=dst.name)
        print("wrote", dst, rep["files"][p.name])
    r = Path(a.results)
    r.mkdir(parents=True, exist_ok=True)
    (r / f"romanize_xlit_{rep['date']}.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
