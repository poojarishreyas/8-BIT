"""Bits per character on held-out sentences (Level 2 metric).

Each sentence is scored on its own: context = [EOS] + tokens, sum of -log p(token) over the sentence's
tokens, divided by ln 2 and by the sentence's character count (NFC code points). Sentences longer than the
model context use a sliding window (stride ctx/2), so every token is scored once with the longest context.
Comparable across tokenizers because it is normalised per character, not per token.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def sentence_nll(model, seqs: list[list[int]], eos: int, ctx: int, device, batch: int = 32) -> list[float]:
    """Total NLL in nats for each sentence."""
    model.eval()
    out = [0.0] * len(seqs)
    short = [i for i, s in enumerate(seqs) if len(s) + 1 <= ctx]
    long_ = [i for i, s in enumerate(seqs) if len(s) + 1 > ctx]
    short.sort(key=lambda i: len(seqs[i]))
    for k in range(0, len(short), batch):
        idx = short[k:k + batch]
        L = max(len(seqs[i]) for i in idx) + 1
        x = torch.full((len(idx), L), eos, dtype=torch.long)
        m = torch.zeros((len(idx), L - 1), dtype=torch.bool)
        for r, i in enumerate(idx):
            s = [eos] + seqs[i]
            x[r, :len(s)] = torch.tensor(s)
            m[r, :len(s) - 1] = True
        x, m = x.to(device), m.to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x[:, :-1])
        nll = F.cross_entropy(logits.float().transpose(1, 2), x[:, 1:], reduction="none") * m
        for r, i in enumerate(idx):
            out[i] = float(nll[r].sum())
    stride = ctx // 2
    for i in long_:
        s = torch.tensor([eos] + seqs[i], dtype=torch.long, device=device)
        total, scored, n_scored = 0.0, 1, 0  # position 0 (EOS) is never a target
        begin = 0
        while scored < len(s):
            end = min(begin + ctx, len(s))
            w = s[begin:end].unsqueeze(0)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(w[:, :-1])
            nll = F.cross_entropy(logits.float()[0], w[0, 1:], reduction="none")
            first_new = scored - begin - 1  # index into nll of first not-yet-scored target
            total += float(nll[first_new:].sum())
            n_scored += len(nll) - first_new
            scored = end
            begin = end - ctx + stride  # next window ends `stride` tokens later, keeping ctx-stride context
        assert n_scored == len(s) - 1, (n_scored, len(s))
        out[i] = total
    model.train()
    return out


def bpc(model, ev: dict, eos: int, ctx: int, device) -> dict:
    """ev = {"ids": [[...]], "chars": [...]} -> dict(bpc, nll_nats, chars, tokens)."""
    nlls = sentence_nll(model, ev["ids"], eos, ctx, device)
    chars = sum(ev["chars"])
    tot = sum(nlls)
    return dict(bpc=tot / math.log(2) / chars, nll_nats=tot, chars=chars,
                tokens=sum(len(s) for s in ev["ids"]), sentences=len(ev["ids"]),
                bits_per_token=tot / math.log(2) / sum(len(s) for s in ev["ids"]))
