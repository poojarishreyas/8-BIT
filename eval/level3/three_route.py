"""Three-route test on the same Belebele questions (default 200): which input route is best per token and second?

  native     Kannada passage + Kannada question/options -> answer (letter log-likelihood)
  translate  model translates the Kannada passage, question and options to English (greedy), then answers the
             English prompt. For multiple choice, "translate back" is not needed (the answer is a letter).
  kanglish   romanized (IndicXlit) passage + English question/options: the code-mixed way people type
Per route: accuracy, total tokens (all prompts + generated), latency (s/question).

Usage: python -m eval.level3.three_route --model <path> --data <bigo-test dir> --out results/models/<tag> [--limit 200]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from eval.level3.belebele import load_rows, prompt
from eval.level3.flores_chrf import chat_prompt
from model.hf_load import load, model_label


@torch.no_grad()
def letter(model, tok, text: str, letter_ids: list[int]) -> tuple[int, int]:
    dev = next(model.parameters()).device
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(dev)
    lg = model(**enc).logits[0, -1, letter_ids]
    return int(lg.argmax()), enc["input_ids"].shape[1]


@torch.no_grad()
def translate(model, tok, text: str, max_new: int = 512) -> tuple[str, int]:
    dev = next(model.parameters()).device
    p = chat_prompt(tok, "Kannada", "English", text)
    enc = tok(p, return_tensors="pt", add_special_tokens=False).to(dev)
    g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    new = g[0, enc["input_ids"].shape[1]:]
    out = tok.decode(new, skip_special_tokens=True).split("</think>")[-1].strip()
    return out, enc["input_ids"].shape[1] + len(new)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    model, tok = load(a.model, a.device, a.tokenizer)
    letter_ids = [tok(" " + L, add_special_tokens=False)["input_ids"][0] for L in "ABCD"]
    kn = load_rows(a.data, "kan_Knda")[: a.limit]
    en = {(r["link"], r["question_number"]): r for r in load_rows(a.data, "eng_Latn")}
    lt = {(r["link"], r["question_number"]): r for r in load_rows(a.data, "kan_Latn")}
    res = {k: dict(correct=0, tokens=0, seconds=0.0, n=0) for k in ("native", "translate", "kanglish")}
    for r in kn:
        key, gold = (r["link"], r["question_number"]), int(r["correct_answer_num"]) - 1
        t0 = time.perf_counter()
        p, n = letter(model, tok, prompt(r), letter_ids)
        res["native"].update(correct=res["native"]["correct"] + (p == gold), tokens=res["native"]["tokens"] + n,
                             seconds=res["native"]["seconds"] + time.perf_counter() - t0, n=res["native"]["n"] + 1)
        # translate route: translate passage, question and options in one call, keep the A-D layout
        t0 = time.perf_counter()
        body = prompt(r).rsplit("\nAnswer:", 1)[0]
        tr, nt = translate(model, tok, body)
        p, n = letter(model, tok, tr + "\nAnswer:", letter_ids)
        res["translate"].update(correct=res["translate"]["correct"] + (p == gold), tokens=res["translate"]["tokens"] + nt + n,
                                seconds=res["translate"]["seconds"] + time.perf_counter() - t0, n=res["translate"]["n"] + 1)
        if key in lt and key in en:
            mixed = {**en[key], "flores_passage": lt[key]["flores_passage"]}
            t0 = time.perf_counter()
            p, n = letter(model, tok, prompt(mixed), letter_ids)
            res["kanglish"].update(correct=res["kanglish"]["correct"] + (p == gold), tokens=res["kanglish"]["tokens"] + n,
                                   seconds=res["kanglish"]["seconds"] + time.perf_counter() - t0, n=res["kanglish"]["n"] + 1)
    rows = []
    for route, v in res.items():
        if v["n"]:
            rows.append(dict(model=model_label(a.model), route=route, n=v["n"], accuracy=v["correct"] / v["n"],
                             tokens_total=v["tokens"], tokens_per_question=v["tokens"] / v["n"],
                             latency_s_per_question=v["seconds"] / v["n"]))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "three_route.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    for x in rows:
        print(x)


if __name__ == "__main__":
    main()
