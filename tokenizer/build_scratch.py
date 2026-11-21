"""From-scratch byte-level BPE tokenizers with our layers, for Level 2 tiny models (equal vocab size).

Same layers and the same parity-aware-bpe glue as tokenizer/build_extension.py, but the base is the 256
byte symbols (no merges) and every language gets merges:
  --pattern original|mark_aware|mark_zw   layer 1
  --units old|akshara                     layer 2 (Kannada words only; other words start from bytes)
  --suffix none|seed                      layer 3 (Kannada words only)
  --parity off|on                         layer 4: off = pooled standard BPE over all languages;
                                          on  = parity-aware selection over languages, using a parallel dev set
Vocab = 256 bytes + merges. --vocab-size counts the bytes, so all tokenizers compare at equal size.

Usage:
  python -m tokenizer.build_scratch --pattern mark_zw --units akshara --suffix seed --vocab-size 32000 \
     --lang eng_Latn=data/processed/train/eng_Latn.txt --lang kan_Knda=data/processed/train/kan_Knda_mix.txt \
     --lang kan_Latn=data/processed/train/kan_Latn.txt --out artifacts/scratch/l123_32k
  # layer 4: add --parity on --dev eng_Latn=... --dev kan_Knda=... --dev kan_Latn=...
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers

from tokenizer import suffix as suffix_mod
from tokenizer.akshara import aksharas, is_kannada
from tokenizer.bpe_glue import apply, learn
from tokenizer.build_extension import BE, from_bl, groups
from tokenizer.extend import mark_aware_pattern, mark_zw_pattern
from tokenizer.train_bpe import QWEN3_PATTERN

PATTERNS = {"original": QWEN3_PATTERN, "mark_aware": mark_aware_pattern(QWEN3_PATTERN),
            "mark_zw": mark_zw_pattern(QWEN3_PATTERN)}


def byte_base(pattern: str) -> Tokenizer:
    vocab = {BE[b]: b for b in range(256)}
    tk = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tk.normalizer = normalizers.NFC()
    tk.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(pattern), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tk.decoder = decoders.ByteLevel()
    return tk


def count_all(tk: Tokenizer, path: str, max_lines: int | None) -> Counter:
    words = Counter()
    pre = tk.pre_tokenizer
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            for w, _ in pre.pre_tokenize_str(line):
                words[w] += 1
    return words


def kannada_body(text: str) -> int | None:
    """Prefix length (0/1) if text is [one non-Kannada prefix char] + Kannada/ZW body, else None."""
    pre = 0 if is_kannada(text[0]) else 1
    body = text[pre:]
    if body and is_kannada(body[0]) and all(is_kannada(c) or c in "‌‍" for c in body):
        return pre
    return None


def run(a) -> dict:
    t0 = time.time()
    tk = byte_base(PATTERNS[a.pattern])
    langs = dict(x.split("=", 1) for x in a.lang)
    devs = dict(x.split("=", 1) for x in (a.dev or []))
    if a.parity == "on" and set(devs) != set(langs):
        raise SystemExit(f"--parity on needs a --dev file for each language {sorted(langs)}")
    words = {l: count_all(tk, p, a.max_lines) for l, p in langs.items()}
    dev_words = {l: count_all(tk, p, None) for l, p in devs.items()}
    rep = dict(config=vars(a).copy(), date=dt.date.today().isoformat(),
               word_types={l: len(w) for l, w in words.items()})
    print("word types", rep["word_types"], f"{time.time()-t0:.0f}s")

    @lru_cache(maxsize=None)
    def dec(w: str) -> str:
        return from_bl(w).decode("utf-8", "replace")

    stage: dict[str, list] = {}
    full: dict[tuple, str | None] = {}
    if a.units == "akshara":
        a0_units, a1_words = Counter(), Counter()
        for l, ws in words.items():
            for w, c in ws.items():
                t = dec(w)
                pre = kannada_body(t)
                if pre is None:
                    continue
                gs, spans = groups(w, list(w))
                for g in gs:
                    if len(g) > 1 and any(is_kannada(ch) for ch in from_bl("".join(g)).decode("utf-8", "replace")):
                        a0_units[tuple(g)] += c
                for us in akshara_units(t, pre, gs, spans)[1]:
                    a1_words[tuple(us)] += c
        m_a0, seen = [], set()
        for pieces in sorted(a0_units, key=lambda g: -a0_units[g]):
            acc = pieces[0]
            for nxt in pieces[1:]:
                if (acc, nxt) not in seen:
                    seen.add((acc, nxt))
                    m_a0.append((acc, nxt))
                acc += nxt
        stage["a0"] = m_a0
        r0 = {p: i for i, p in enumerate(m_a0)}
        # a1 runs over a0-built character units
        a1_char_words = Counter()
        for us, c in a1_words.items():
            a1_char_words[tuple(x for u in us for x in apply(tuple(u), r0))] += c
        stage["a1"] = learn([a1_char_words], 10 ** 6, min_frequency=a.min_akshara_freq)
        r1 = {p: i for i, p in enumerate(stage["a1"])}
        for us in a1_char_words:
            o = apply(us, r1)
            full[us] = o[0] if len(o) == 1 else None
        occ = sum(a1_char_words.values())
        rep["akshara_types"] = len(full)
        rep["akshara_occurrence_coverage"] = sum(c for u, c in a1_char_words.items() if full[u]) / occ
        print(f"a0 {len(m_a0)}  a1 {len(stage['a1'])}  akshara coverage {rep['akshara_occurrence_coverage']:.4%}")
    else:
        r0 = {}

    def b_segments(w: str) -> list[tuple[str, ...]]:
        """Units for stage b: aksharas for Kannada words (layer 2), bytes otherwise; suffix splits (layer 3)."""
        t = dec(w)
        pre = kannada_body(t) if t else None
        if pre is None:
            return [tuple(w)]
        aks_text = aksharas(t[pre:])
        if a.units != "akshara":
            if a.suffix == "seed" and (cut := suffix_mod.split(aks_text)) is not None:
                cb = len((t[:pre] + "".join(aks_text[:cut])).encode("utf-8"))
                return [tuple(w[:cb]), tuple(w[cb:])]
            return [tuple(w)]
        gs, spans = groups(w, list(w))
        prefix_units, per_ak = akshara_units(t, pre, gs, spans)
        cut = suffix_mod.split(aks_text) if a.suffix == "seed" else None
        segs, seq = [], list(prefix_units)
        for k, us in enumerate(per_ak):
            key = tuple(x for u in us for x in apply(tuple(u), r0))
            u = full.get(key)
            if u is None or k == cut:
                if seq:
                    segs.append(tuple(seq))
                seq = []
                if u is None:  # rare akshara: keep its char units as their own segment (still learnable)
                    segs.append(key)
                    continue
            seq.append(u)
        if seq:
            segs.append(tuple(seq))
        return segs

    def to_train(ws: Counter) -> Counter:
        out = Counter()
        for w, c in ws.items():
            for s in b_segments(w):
                out[s] += c
        return out

    n_pre = sum(len(v) for v in stage.values())
    budget = a.vocab_size - 256 - n_pre
    if a.parity == "on":
        order = sorted(langs)
        stage["b"] = learn([to_train(words[l]) for l in order], budget,
                           dev_per_lang=[to_train(dev_words[l]) for l in order], min_frequency=2)
        rep["parity_lang_order"] = order
    else:
        pooled = Counter()
        for ws in words.values():
            pooled.update(to_train(ws))
        stage["b"] = learn([pooled], budget, min_frequency=2)
    print(f"b {len(stage['b'])} (budget {budget})")

    vocab = {BE[b]: b for b in range(256)}
    merges, dropped = [], Counter()
    for st in ("a0", "a1", "b"):
        for l, r in stage.get(st, []):
            if l not in vocab or r not in vocab:
                dropped["missing_component"] += 1
                continue
            if l + r in vocab:
                dropped["duplicate_result_kept_merge"] += 1  # another pair already makes this string
            else:
                vocab[l + r] = len(vocab)
            merges.append((l, r))
    tk.model = models.BPE(vocab=vocab, merges=merges)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tk.save(str(out / "tokenizer.json"))
    rep.update(stage_merge_counts={k: len(v) for k, v in stage.items()}, vocab_size=len(vocab),
               dropped=dict(dropped), seconds=round(time.time() - t0, 1))
    (out / "build_report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "config"}, ensure_ascii=False))
    return rep


def akshara_units(text: str, pre: int, gs: list[list[str]], spans: list[tuple[int, int]]):
    """Split char groups into (prefix units, units per akshara)."""
    aks = aksharas(text[pre:])
    ends, p = [], pre
    for ak in aks:
        p += len(ak)
        ends.append(p)
    per_ak: list[list[str]] = [[] for _ in aks]
    prefix: list[str] = []
    for g, (s, e) in zip(gs, spans):
        u = "".join(g)
        if e <= pre:
            prefix.extend(g)  # non-Kannada prefix char stays as byte units
            continue
        per_ak[next(i for i, x in enumerate(ends) if e <= x)].append(u)
    return prefix, per_ak


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", choices=list(PATTERNS), default="mark_zw")
    ap.add_argument("--units", choices=["old", "akshara"], default="akshara")
    ap.add_argument("--suffix", choices=["none", "seed"], default="seed")
    ap.add_argument("--parity", choices=["off", "on"], default="off")
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--min-akshara-freq", type=int, default=5)
    ap.add_argument("--lang", action="append", required=True, help="name=train_file (repeatable)")
    ap.add_argument("--dev", action="append", help="name=parallel_dev_file (layer 4)")
    ap.add_argument("--max-lines", type=int, default=None)
    ap.add_argument("--out", required=True)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
