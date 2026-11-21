"""Extend Qwen3's tokenizer.json with new Kannada merges (append-only; old tokens, merges, IDs untouched).

Layers (each is a flag, so ablations come from one script):
  --pattern original|mark_aware   layer 1: pre-tokenizer split rule
  --units old|akshara             layer 2: 'old'  = BPE continues from Qwen3's own tokens (standard extension)
                                           'akshara' = a0 build each Kannada char, a1 build each akshara
                                           (BPE on aksharas as isolated words), b merge whole aksharas only
  --suffix none|seed              layer 3: hard stem|suffix splits in training words (tokenizer/suffix.py)
  --parity off|on                 layer 4: parity-aware merge selection (needs >1 language; see below)

Merge learning always goes through swiss-ai/parity-aware-bpe via tokenizer/bpe_glue.py.
Only Kannada-script words are used, and every kept merge's result must contain Kannada bytes
(UTF-8 lead E0 B2 / E0 B3), so English/JSON/code tokenization cannot change (tested).

Usage:
  python -m tokenizer.build_extension --pattern mark_aware --units akshara --suffix seed \
      --new-tokens 16000 --train data/processed/train/kan_Knda.txt --out artifacts/ext/l123_16k
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

from tokenizers import Tokenizer

from tokenizer import suffix as suffix_mod
from tokenizer.akshara import aksharas, is_kannada
from tokenizer.bpe_glue import apply, learn
from tokenizer.extend import make_mark_aware
from tokenizer.registry import _byte_decoder

BD = _byte_decoder()
BE = {v: k for k, v in BD.items()}


def to_bl(s: str) -> str:
    return "".join(BE[b] for b in s.encode("utf-8"))


def from_bl(t: str) -> bytes:
    return bytes(BD[c] for c in t)


def has_kannada_bytes(t: str) -> bool:
    b = from_bl(t)
    return b"\xe0\xb2" in b or b"\xe0\xb3" in b


def load_base(base: str) -> dict:
    p = Path(base)
    if not p.exists():
        from huggingface_hub import hf_hub_download

        p = Path(hf_hub_download(base, "tokenizer.json"))
    return json.loads(p.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ word collection
def count_words(tk: Tokenizer, path: str, max_lines: int | None) -> Counter:
    """Pre-tokens (byte-level strings) whose text is: optional 1 non-letter prefix char + Kannada body."""
    words = Counter()
    pre = tk.pre_tokenizer
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            for w, _ in pre.pre_tokenize_str(line):
                words[w] += 1
    keep = Counter()
    for w, c in words.items():
        s = from_bl(w).decode("utf-8", "replace")
        body = s[1:] if s and not (is_kannada(s[0])) else s
        if body and all(is_kannada(ch) or ch in "‌‍" for ch in body) and any(is_kannada(ch) for ch in body):
            keep[w] += c
    return keep


# ------------------------------------------------------------------ unit decomposition
def groups(word_bl: str, old_tokens: list[str]) -> tuple[list[list[str]], list[tuple[int, int]]]:
    """Group old tokens so no group boundary falls inside a character (fuses e.g. 'Ġà²'+'®' = ' ಮ').
    Returns (pieces per group, char span (start,end) per group)."""
    text = from_bl(word_bl).decode("utf-8")
    char_end_bytes, p = [], 0
    for ch in text:
        p += len(ch.encode("utf-8"))
        char_end_bytes.append(p)
    ends = set(char_end_bytes)
    out, spans, cur, b, start_char = [], [], [], 0, 0
    nchar = 0
    for t in old_tokens:
        cur.append(t)
        b += len(from_bl(t))
        if b in ends:
            end_char = char_end_bytes.index(b) + 1
            out.append(cur)
            spans.append((start_char, end_char))
            cur, start_char = [], end_char
    assert not cur, "old tokens must end on a character boundary"
    # attach ZWJ/ZWNJ to the preceding Kannada character, so building them needs only merges that
    # contain Kannada bytes (ZWJ alone is two old pieces; a ZW-only merge would be dropped by the filter)
    fo, fs = [], []
    for g, (s, e) in zip(out, spans):
        if fo and all(ch in "‌‍" for ch in text[s:e]) and is_kannada(text[s - 1]):
            fo[-1] = fo[-1] + g
            fs[-1] = (fs[-1][0], e)
        else:
            fo.append(g)
            fs.append((s, e))
    return fo, fs


def run(a) -> dict:
    t0 = time.time()
    base = load_base(a.base)
    tj = make_mark_aware(base, zw=a.pattern == "mark_zw") if a.pattern != "original" else json.loads(json.dumps(base))
    tk = Tokenizer.from_str(json.dumps(tj))
    old_vocab: dict[str, int] = tj["model"]["vocab"]
    old_merges = {tuple(m) if isinstance(m, list) else tuple(m.split(" ")) for m in tj["model"]["merges"]}
    next_id = max(max(old_vocab.values()), max(t["id"] for t in tj["added_tokens"])) + 1
    first_new_id = next_id

    words = count_words(tk, a.train, a.max_lines)
    rep = dict(config=vars(a).copy(), date=dt.date.today().isoformat(), first_new_id=first_new_id,
               kannada_word_types=len(words), kannada_word_tokens=sum(words.values()))
    print(f"{len(words):,} Kannada word types ({sum(words.values()):,} tokens) in {time.time()-t0:.0f}s")

    @lru_cache(maxsize=None)
    def old_tok(w: str) -> tuple[str, ...]:
        return tuple(t.value for t in tk.model.tokenize(w))

    stage_merges: dict[str, list[tuple[str, str]]] = {}

    if a.units == "old":
        train = Counter()
        for w, c in words.items():
            if a.suffix == "seed":
                for seg in suffix_segments_old(w, old_tok):
                    train[seg] += c
            else:
                train[old_tok(w)] += c
        stage_merges["b"] = learn([train], a.new_tokens, min_frequency=2)
    else:
        # a0: every character (or ' '+char fused group) -> one token
        a0_units, a1_words, b_words_raw = Counter(), Counter(), []
        for w, c in words.items():
            gs, spans = groups(w, list(old_tok(w)))
            text = from_bl(w).decode("utf-8")
            for g in gs:
                if len(g) > 1:
                    a0_units[tuple(g)] += c
            # akshara grouping over char-groups
            pre_len = 0 if is_kannada(text[0]) else 1
            aks = aksharas(text[pre_len:])
            ak_ends, p = [], pre_len
            for ak in aks:
                p += len(ak)
                ak_ends.append(p)
            units_per_ak: list[list[str]] = [[] for _ in aks]
            prefix_units: list[str] = []
            for g, (s, e) in zip(gs, spans):
                u = "".join(g)
                if e <= pre_len:
                    prefix_units.append(u)
                    continue
                k = next(i for i, x in enumerate(ak_ends) if e <= x)
                units_per_ak[k].append(u)
            for k, us in enumerate(units_per_ak):
                a1_words[tuple(us)] += c
            b_words_raw.append((prefix_units, units_per_ak, aks, c))
        m_a0 = []
        for pieces in sorted(a0_units, key=lambda g: -a0_units[g]):
            acc = pieces[0]
            for nxt in pieces[1:]:
                if (acc, nxt) not in m_a0_set(m_a0):
                    m_a0.append((acc, nxt))
                acc += nxt
        stage_merges["a0"] = m_a0
        print(f"a0: {len(m_a0)} char-building merges")
        # a1: aksharas as isolated words, BPE to completion for aksharas with freq >= min-akshara-freq
        stage_merges["a1"] = learn([a1_words], 10 ** 6, min_frequency=a.min_akshara_freq)
        r1 = {p: i for i, p in enumerate(stage_merges["a1"])}
        full = {}
        for us in a1_words:
            out = apply(us, r1)
            full[us] = out[0] if len(out) == 1 else None
        n_ak_full = sum(1 for v in full.values() if v)
        rep["akshara_types"] = len(full)
        rep["akshara_types_single_token"] = n_ak_full
        occ = sum(a1_words.values())
        rep["akshara_occurrence_coverage"] = sum(c for u, c in a1_words.items() if full[u]) / occ
        print(f"a1: {len(stage_merges['a1'])} merges; {n_ak_full}/{len(full)} akshara types single-token, "
              f"{rep['akshara_occurrence_coverage']:.4%} of occurrences")
        # b: words as sequences of whole-akshara units; barriers at rare aksharas and suffix boundaries
        train = Counter()
        n_suffix_splits = 0
        for prefix_units, units_per_ak, aks, c in b_words_raw:
            cut = suffix_mod.split(aks) if a.suffix == "seed" else None
            if cut is not None:
                n_suffix_splits += c
            seq: list[str] = list(prefix_units)
            for k, us in enumerate(units_per_ak):
                u = full[tuple(us)]
                if u is None or k == cut:
                    if len(seq) > 1:
                        train[tuple(seq)] += c
                    seq = []
                    if u is None:
                        continue
                seq.append(u)
            if len(seq) > 1:
                train[tuple(seq)] += c
        rep["suffix_split_word_tokens"] = n_suffix_splits
        budget = a.new_tokens - len(stage_merges["a0"]) - len(stage_merges["a1"])
        if budget <= 0:
            raise SystemExit(f"--new-tokens {a.new_tokens} too small for a0+a1 ({a.new_tokens - budget})")
        stage_merges["b"] = learn([train], budget, min_frequency=2)
        print(f"b: {len(stage_merges['b'])} merges (budget {budget})")

    # ------------------------------------------------------------------ append (filtered)
    available = set(old_vocab)
    new_vocab, new_merges, dropped = {}, [], Counter()
    for stage in ("a0", "a1", "b"):
        for l, r in stage_merges.get(stage, []):
            res = l + r
            if (l, r) in old_merges:
                dropped["already_old_merge"] += 1
                continue
            if l not in available or r not in available:
                dropped["missing_component"] += 1
                continue
            if not has_kannada_bytes(res):
                dropped["no_kannada_bytes"] += 1
                continue
            new_merges.append([l, r])
            if res not in old_vocab and res not in new_vocab:
                new_vocab[res] = next_id
                next_id += 1
            available.add(res)
    # HF `tokenizers` re-numbers added tokens after the model vocab when the model vocab has a gap
    # (old ids end at 151642, ours start at 151669). Put the added special tokens into model.vocab at
    # their original ids so the vocab is contiguous; no merge produces them, so tokenization is unchanged.
    for t in tj["added_tokens"]:
        tj["model"]["vocab"].setdefault(t["content"], t["id"])
    tj["model"]["vocab"].update(new_vocab)
    tj["model"]["merges"] = [list(m) if isinstance(m, (list, tuple)) else m.split(" ") for m in tj["model"]["merges"]] + new_merges
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tokenizer.json").write_text(json.dumps(tj, ensure_ascii=False), encoding="utf-8")
    Tokenizer.from_file(str(out / "tokenizer.json"))
    rep.update(stage_merge_counts={k: len(v) for k, v in stage_merges.items()}, appended_merges=len(new_merges),
               appended_tokens=len(new_vocab), dropped=dict(dropped), last_new_id=next_id - 1,
               seconds=round(time.time() - t0, 1))
    (out / "build_report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "config"}, indent=1, ensure_ascii=False))
    return rep


_m_cache: dict[int, set] = {}


def m_a0_set(m: list) -> set:
    s = _m_cache.setdefault(id(m), set())
    if len(s) != len(m):
        s.update(m)
    return s


def suffix_segments_old(w: str, old_tok) -> list[tuple[str, ...]]:
    """Suffix split for --units old: split the pre-token text at the seed boundary, tokenize parts with old merges."""
    text = from_bl(w).decode("utf-8")
    pre_len = 0 if is_kannada(text[0]) else 1
    aks = aksharas(text[pre_len:])
    cut = suffix_mod.split(aks)
    if cut is None:
        return [old_tok(w)]
    left = text[:pre_len] + "".join(aks[:cut])
    right = "".join(aks[cut:])
    return [old_tok(to_bl(left)), old_tok(to_bl(right))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--pattern", choices=["original", "mark_aware", "mark_zw"], default="mark_zw")
    ap.add_argument("--units", choices=["old", "akshara"], default="akshara")
    ap.add_argument("--suffix", choices=["none", "seed"], default="seed")
    ap.add_argument("--parity", choices=["off", "on"], default="off")
    ap.add_argument("--new-tokens", type=int, default=16000, help="total merges to append (a0+a1+b)")
    ap.add_argument("--min-akshara-freq", type=int, default=5)
    ap.add_argument("--train", default="data/processed/train/kan_Knda.txt")
    ap.add_argument("--max-lines", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.parity == "on":
        raise SystemExit("--parity on: not wired yet for the extension (Kannada-only merges => single language); "
                         "see docs note in build report")
    run(a)


if __name__ == "__main__":
    main()
