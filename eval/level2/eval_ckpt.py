"""Re-score saved Level 2 checkpoints on (new) eval forms, e.g. romanized / Kanglish sets that arrive later.

Tokenizes the given sentence files with the run's tokenizer and appends BPC rows to <results>/level2_bpc.csv.

Usage:
  python -m eval.level2.eval_ckpt --run runs/l123_32k_s0 --data data/level2/l123_32k --seed 0 \
      --eval kan_Latn.flores_devtest_xlit=data/processed/test/flores_devtest_kan_Latn_xlit.txt
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import unicodedata
from pathlib import Path

import torch

from eval.level2.bpc import bpc
from eval.level2.model import GPT, GPTConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir containing ckpt.pt")
    ap.add_argument("--data", required=True, help="prepared data dir (meta.json gives the tokenizer)")
    ap.add_argument("--tokenizer", default=None, help="override tokenizer.json path (default: meta.json)")
    ap.add_argument("--eval", action="append", required=True, help="form.split=file")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    from eval.level2.prepare import load_tok

    meta = json.loads((Path(a.data) / "meta.json").read_text())
    tk = load_tok(a.tokenizer or meta["tokenizer"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s = torch.load(Path(a.run) / "ckpt.pt", map_location=dev)
    model = GPT(GPTConfig(**s["cfg"])).to(dev)
    model.load_state_dict(s["model"])
    rows = []
    for e in a.eval:
        form, p = e.split("=", 1)
        sents = [unicodedata.normalize("NFC", x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]
        enc = tk.encode_batch(sents, add_special_tokens=False)
        r = bpc(model, dict(ids=[x.ids for x in enc], chars=[len(x) for x in sents]), meta["eos_id"], s["cfg"]["ctx"], dev)
        rows.append(dict(tokenizer=meta["name"], vocab_size=meta["vocab_size"], seed=a.seed, form=form.split(".")[0],
                         split=form.split(".", 1)[1] if "." in form else "", **r, params=model.n_params(),
                         params_non_embedding=model.n_params(True), train_chars=meta["train_chars"],
                         train_tokens=meta["train_tokens"], steps=s["step"], step_tokens="", train_seconds=s["train_seconds"],
                         d_model=s["cfg"]["d_model"], n_layer=s["cfg"]["n_layer"], ctx=s["cfg"]["ctx"], gpus="",
                         date=dt.date.today().isoformat()))
        print(f"{meta['name']} s{a.seed} {form}: BPC {r['bpc']:.4f}")
    f = Path(a.results) / "level2_bpc.csv"
    f.parent.mkdir(parents=True, exist_ok=True)
    new = not f.exists()
    with f.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
