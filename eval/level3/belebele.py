"""Belebele reading comprehension (Level 3), log-likelihood scoring of the answer letter.

Prompt (lm-evaluation-harness style):  P: {passage}\\nQ: {question}\\nA: ..\\nB: ..\\nC: ..\\nD: ..\\nAnswer:
Score = argmax over the logits of the tokens " A"/" B"/" C"/" D" at the last position (one forward pass
per question). Same prompt for every model, so differences come from the model + tokenizer.
Also records prompt tokens per question (context cost) and wall time.

Usage:
  python -m eval.level3.belebele --model /kaggle/working/ours_C/hf --data /kaggle/input/bigo-test \
      --langs kan_Knda eng_Latn --out /kaggle/working/results
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import time
import unicodedata
from pathlib import Path

import torch

from model.hf_load import load, model_label


def prompt(ex: dict) -> str:
    n = lambda s: unicodedata.normalize("NFC", " ".join(str(s).split()))
    return (f"P: {n(ex['flores_passage'])}\nQ: {n(ex['question'])}\n"
            + "".join(f"{L}: {n(ex[f'mc_answer{i}'])}\n" for i, L in enumerate("ABCD", 1)) + "Answer:")


@torch.no_grad()
def run(model, tok, rows: list[dict], dev, batch: int) -> tuple[list[int], list[int]]:
    letter_ids = []
    for L in "ABCD":
        ids = tok(" " + L, add_special_tokens=False)["input_ids"]
        assert len(ids) == 1, f"' {L}' is not one token"
        letter_ids.append(ids[0])
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    preds, lens = [], []
    for k in range(0, len(rows), batch):
        ps = [prompt(r) for r in rows[k:k + batch]]
        enc = tok(ps, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            logits = model(**enc).logits[:, -1, :]
        preds += logits[:, letter_ids].argmax(-1).tolist()
        lens += enc["attention_mask"].sum(-1).tolist()
    return preds, lens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model folder or hub id")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--data", required=True, help="folder with belebele_<lang>.jsonl")
    ap.add_argument("--langs", nargs="+", default=["kan_Knda", "eng_Latn"])
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    model, tok = load(a.model, a.device, a.tokenizer)
    dev = torch.device(a.device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_out = []
    for lang in a.langs:
        f = next(Path(a.data).rglob(f"belebele_{lang}.jsonl"))
        rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines()][: a.limit]
        t0 = time.perf_counter()
        preds, lens = run(model, tok, rows, dev, a.batch)
        secs = time.perf_counter() - t0
        gold = [int(r["correct_answer_num"]) - 1 for r in rows]
        acc = sum(p == g for p, g in zip(preds, gold)) / len(rows)
        r = dict(model=model_label(a.model), tokenizer_len=len(tok), lang=lang, split="belebele", n=len(rows),
                 accuracy=round(acc, 4), mean_prompt_tokens=round(sum(lens) / len(lens), 1), seconds=round(secs, 1),
                 date=dt.date.today().isoformat())
        rows_out.append(r)
        print(r, flush=True)
        (out / f"belebele_preds_{model_label(a.model).replace('/', '_')}_{lang}.json").write_text(
            json.dumps(dict(preds=preds, gold=gold, prompt_tokens=lens)), encoding="utf-8")
    f = out / "level3_belebele.csv"
    new = not f.exists()
    with f.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
        if new:
            w.writeheader()
        w.writerows(rows_out)


if __name__ == "__main__":
    main()
