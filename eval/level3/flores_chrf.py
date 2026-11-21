"""FLORES-200 devtest translation, chrF++ (sacrebleu), both directions (Level 3).

Greedy decoding with the model's chat template (Qwen3: enable_thinking=False). Besides chrF++, records
the cost side: output tokens, generation seconds, output tokens/s and output *characters*/s (what a
reader sees), prompt tokens. Hypotheses are saved for the failure analysis.

Usage:
  python -m eval.level3.flores_chrf --model /kaggle/working/ours_C/hf --data /kaggle/input/bigo-test \
      --directions eng-kan kan-eng --limit 200 --out /kaggle/working/results
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

NAMES = {"eng": ("English", "eng_Latn"), "kan": ("Kannada", "kan_Knda"),
         "kanglish": ("Kanglish (Kannada written in English letters, mixed with English)", "kanglish")}


def read(data: str, code: str, pair: str = "") -> list[str]:
    """FLORES devtest files; for the Kanglish direction both sides come from the hand-written set."""
    if "kanglish" in pair:
        name = "kanglish_handwritten.txt" if code == "kanglish" else "kanglish_handwritten_eng.txt"
    else:
        name = f"flores_devtest_{code}.txt"
    f = next(Path(data).rglob(name), None)
    if f is None:
        return []
    return [unicodedata.normalize("NFC", l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]


def chat_prompt(tok, src_lang: str, tgt_lang: str, text: str) -> str:
    msgs = [{"role": "user", "content": f"Translate the following {src_lang} text into {tgt_lang}. "
                                        f"Reply with only the translation.\n\n{text}"}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def translate(model, tok, srcs: list[str], src_lang: str, tgt_lang: str, dev, batch: int, max_new: int):
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    hyps, out_toks, in_toks, secs = [], 0, 0, 0.0
    for k in range(0, len(srcs), batch):
        ps = [chat_prompt(tok, src_lang, tgt_lang, s) for s in srcs[k:k + batch]]
        enc = tok(ps, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        secs += time.perf_counter() - t0
        new = g[:, enc["input_ids"].shape[1]:]
        for row in new:
            ids = [t for t in row.tolist() if t != tok.pad_token_id]
            if tok.eos_token_id in ids:
                ids = ids[: ids.index(tok.eos_token_id)]
            out_toks += len(ids)
            text = tok.decode(ids, skip_special_tokens=True)
            text = text.split("</think>")[-1].strip()
            hyps.append(" ".join(text.split()))
        in_toks += int(enc["attention_mask"].sum())
    return hyps, out_toks, in_toks, secs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--data", required=True, help="folder with flores_devtest_{eng_Latn,kan_Knda}.txt")
    ap.add_argument("--directions", nargs="+", default=["eng-kan", "kan-eng", "kanglish-eng"])
    ap.add_argument("--limit", type=int, default=200, help="200 for quick runs; 1012 = full devtest")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=600)
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default=None, help="default: cuda if available (incl. ROCm), else cpu")
    a = ap.parse_args()
    import sacrebleu

    model, tok = load(a.model, a.device, a.tokenizer)
    dev = next(model.parameters()).device
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    label = model_label(a.model)
    for d in a.directions:
        s, t = d.split("-")
        srcs, refs = read(a.data, NAMES[s][1], d)[: a.limit], read(a.data, NAMES[t][1], d)[: a.limit]
        if not srcs or len(srcs) != len(refs):
            print(f"[skip] {d}: data missing or not parallel")
            continue
        hyps, ot, it, secs = translate(model, tok, srcs, NAMES[s][0], NAMES[t][0], dev, a.batch, a.max_new_tokens)
        chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score
        r = dict(model=label, tokenizer_len=len(tok), direction=d, split="flores_devtest", n=len(srcs),
                 chrfpp=round(chrf, 2), output_tokens=ot, output_chars=sum(len(h) for h in hyps), prompt_tokens=it,
                 gen_seconds=round(secs, 1), out_tokens_per_s=round(ot / secs, 1), out_chars_per_s=round(sum(len(h) for h in hyps) / secs, 1),
                 tokens_per_answer=round(ot / len(srcs), 1), batch=a.batch, date=dt.date.today().isoformat())
        rows.append(r)
        print(r, flush=True)
        (out / f"flores_hyps_{label.replace('/', '_')}_{d}.jsonl").write_text(
            "\n".join(json.dumps(dict(src=x, ref=y, hyp=h), ensure_ascii=False) for x, y, h in zip(srcs, refs, hyps)),
            encoding="utf-8")
    f = out / "level3_chrf.csv"
    new = not f.exists()
    with f.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
