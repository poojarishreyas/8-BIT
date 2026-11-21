"""Speed and cost on a model: prefill tokens/s, decode tokens/s, latency per answer, ₹ per 1,000 answers.

Prompts: the first N FLORES devtest Kannada sentences in a chat prompt ("Explain in Kannada: ..."). For each:
  prefill = one forward pass over the prompt (time-to-first-token proxy)
  decode  = greedy generation of exactly --new-tokens tokens (min = max), decode tok/s = tokens / (gen - prefill)
Cost uses an explicit, stated price assumption (--inr-per-m-input / --inr-per-m-output, default ₹25 / ₹100 per
million tokens) applied to the measured tokens per answer from FLORES eng->kan translation when available
(results of eval.level3.flores_chrf), else to this run's prompt/output tokens.

Usage: python -m eval.level3.speed_cost --model <path> --data <bigo-test dir> --out results/models/<tag>
"""
from __future__ import annotations

import argparse
import json
import time
import unicodedata
from pathlib import Path

import torch

from eval.level3.flores_chrf import chat_prompt
from model.hf_load import load, model_label


@torch.no_grad()
def measure(model, tok, sents: list[str], new_tokens: int) -> dict:
    dev = next(model.parameters()).device
    sync = (lambda: torch.cuda.synchronize(dev)) if dev.type == "cuda" else (lambda: None)
    pre_t, dec_t, p_tok, lat = 0.0, 0.0, 0, []
    for i, s in enumerate(sents):
        p = chat_prompt(tok, "Kannada", "Kannada", s).replace("Translate the following Kannada text into Kannada. Reply with only the translation.",
                                                                "Explain the following Kannada sentence in Kannada, in two sentences.")
        enc = tok(p, return_tensors="pt", add_special_tokens=False).to(dev)
        if i == 0:  # warm-up
            model.generate(**enc, max_new_tokens=4, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        sync()
        t0 = time.perf_counter()
        model(**enc)
        sync()
        t1 = time.perf_counter()
        model.generate(**enc, max_new_tokens=new_tokens, min_new_tokens=new_tokens, do_sample=False,
                       pad_token_id=tok.pad_token_id or tok.eos_token_id)
        sync()
        t2 = time.perf_counter()
        pre_t += t1 - t0
        dec_t += max(1e-6, (t2 - t1) - (t1 - t0))
        p_tok += enc["input_ids"].shape[1]
        lat.append(t2 - t1)
    n = len(sents)
    return dict(n=n, prompt_tokens_mean=p_tok / n, prefill_tokens_per_s=p_tok / pre_t, decode_tokens_per_s=n * new_tokens / dec_t,
                ttft_s_mean=pre_t / n, latency_s_mean=sum(lat) / n, new_tokens=new_tokens)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--new-tokens", type=int, default=128)
    ap.add_argument("--inr-per-m-input", type=float, default=25.0)
    ap.add_argument("--inr-per-m-output", type=float, default=100.0)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    model, tok = load(a.model, a.device, a.tokenizer)
    f = next(Path(a.data).rglob("flores_devtest_kan_Knda.txt"))
    sents = [unicodedata.normalize("NFC", l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()][: a.n]
    r = measure(model, tok, sents, a.new_tokens)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # cost per 1,000 answers from the eng->kan translation run if present (real answers), else this run
    tr = out / "level3_chrf.csv"
    basis = "speed_cost prompts (fixed 128 output tokens)"
    pin, pout = r["prompt_tokens_mean"], r["new_tokens"]
    if tr.exists():
        import csv

        for row in csv.DictReader(tr.open(encoding="utf-8")):
            if row["direction"] == "eng-kan":
                pin, pout = float(row["prompt_tokens"]) / float(row["n"]), float(row["tokens_per_answer"])
                basis = "FLORES eng->kan translation answers (eval.level3.flores_chrf)"
    cost = 1000 * (pin * a.inr_per_m_input + pout * a.inr_per_m_output) / 1e6
    r.update(model=model_label(a.model), tokens_in_per_answer=pin, tokens_out_per_answer=pout, inr_per_1000_answers=cost,
             price_assumption=f"₹{a.inr_per_m_input}/M input, ₹{a.inr_per_m_output}/M output tokens (same price for every model)",
             cost_basis=basis, dtype=str(next(model.parameters()).dtype), device=str(next(model.parameters()).device))
    (out / "speed_cost.json").write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(r, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
