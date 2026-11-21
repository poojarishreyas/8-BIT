"""Level 1 matrix: every tokenizer x test set x form -> results/level1_matrix.csv (one long table).

Metrics per (tokenizer, split, form), over sentences parallel across forms:
  tokens_total, tokens_per_sent, fertility (tokens/word), tokens_per_char, parity (tokens / English tokens),
  parity_median, bytes_premium (UTF-8 bytes / English bytes; tokenizer-independent), renyi_eff (alpha 2.5,
  tokenization-scorer, observed vocab), single_token_word_rate (% words covered by exactly one token),
  akshara_violations (count) + akshara_split_rate, byte_fallback_rate, vocab_size, new_vocab (over Qwen3's 151,669).

Incremental: each (tokenizer, split) is cached in results/level1/parts/ keyed by the input files' size+mtime,
so new forms (romanized, Kanglish) or new tokenizers only compute what changed.

Usage: python -m eval.level1.matrix [--workers 4] [--only NAME ...] [--force]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import statistics
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW, PROC = ROOT / "data/raw/test", ROOT / "data/processed/test"
QWEN3_BASE_LEN = 151669
E = "artifacts/ext"

# name -> (family, spec). spec = registry baseline name or tokenizer.json path (relative to repo root)
TOKENIZERS: dict[str, tuple[str, str]] = {
    "cl100k_base": ("baseline", "cl100k_base"), "o200k_base": ("baseline", "o200k_base"),
    "llama3": ("baseline", "llama3"), "gemma3": ("baseline", "gemma3"), "sarvam1": ("baseline", "sarvam1"),
    "indictok256k": ("baseline", "indictok256k"), "qwen3": ("baseline", "qwen3"),
    "std_8k": ("std_ext", f"{E}/std_8k/tokenizer.json"), "std_16k": ("std_ext", f"{E}/std_16k/tokenizer.json"),
    "std_32k": ("std_ext", f"{E}/std_32k/tokenizer.json"), "std_64k": ("std_ext", f"{E}/std_64k/tokenizer.json"),
    "l1_16k": ("ablation", f"{E}/l1_16k/tokenizer.json"), "l1z_16k": ("ablation", f"{E}/l1z_16k/tokenizer.json"),
    "l12_16k": ("ablation", f"{E}/l12_16k/tokenizer.json"), "l12z_16k": ("ablation", f"{E}/l12z_16k/tokenizer.json"),
    "l123_16k": ("ablation", f"{E}/l123_16k/tokenizer.json"), "l123z_16k": ("ablation", f"{E}/l123z_16k/tokenizer.json"),
    "l1234_16k": ("ablation", f"{E}/l1234_16k/tokenizer.json"),
    "l123z_8k": ("sweep", f"{E}/l123z_8k/tokenizer.json"), "l123z_32k": ("sweep", f"{E}/l123z_32k/tokenizer.json"),
    "l123z_64k": ("sweep", f"{E}/l123z_64k/tokenizer.json"), "l1234_32k": ("sweep", f"{E}/l1234_32k/tokenizer.json"),
    "l1234latn_16k": ("variant", f"{E}/l1234latn_16k/tokenizer.json"),
    "l1234latn_32k": ("variant", f"{E}/l1234latn_32k/tokenizer.json"),
    "bigo_final": ("final", "artifacts/final/bigo_final/tokenizer.json"),
}


def splits() -> dict[str, dict[str, Path]]:
    s = {}
    for sp, d in (("flores_devtest", RAW), ("in22gen", RAW), ("belebele_passages", PROC)):
        s[sp] = {"eng_Latn": d / f"{sp}_eng_Latn.txt", "kan_Knda": d / f"{sp}_kan_Knda.txt",
                 "kan_Latn_xlit": PROC / f"{sp}_kan_Latn_xlit.txt"}
    s["kanglish_handwritten"] = {"eng_Latn": PROC / "kanglish_handwritten_eng.txt",
                                 "kanglish": PROC / "kanglish_handwritten.txt"}
    return {k: {f: p for f, p in v.items() if p.exists()} for k, v in s.items()}


def lines(p: Path) -> list[str]:
    return [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def in_vocab(tok, akshara: str, after_space: bool = False) -> bool:
    """True if the akshara, in the form it occurs (with its leading space when it starts a word after a space:
    Qwen3's old merge 'Ġà²' always fuses that space into the first akshara), is a single token."""
    return len(tok.encode((" " if after_space else "") + akshara)) == 1


def sent_stats(tok, text: str):
    from tokenizer.akshara import kannada_akshara_byte_spans

    text = tok.normalize(text)
    ids = tok.encode(text)
    pieces = tok.token_bytes(text)
    spans, pos = [], 0
    for p in pieces:
        spans.append((pos, pos + len(p)))
        pos += len(p)
    inner = {e for _, e in spans[:-1]}
    raw = text.encode("utf-8")
    ak_spans = kannada_akshara_byte_spans(text)
    bad = [(raw[s:e].decode("utf-8"), s > 0 and raw[s - 1:s] == b" ") for s, e in ak_spans if any(s < b < e for b in inner)]
    viol = len(bad)
    viol_in_vocab = sum(1 for a, sp in bad if in_vocab(tok, a, sp))  # must be 0: in-inventory aksharas never split
    n_ak = len(ak_spans)
    # word byte spans (whitespace-separated)
    words, i = [], 0
    for w in text.split():
        j = text.index(w, i)
        bs = len(text[:j].encode("utf-8"))
        words.append((bs, bs + len(w.encode("utf-8"))))
        i = j + len(w)
    single, k = 0, 0
    for ws, we in words:
        while k < len(spans) and spans[k][1] <= ws:
            k += 1
        n, m = 0, k
        while m < len(spans) and spans[m][0] < we:
            n += 1
            m += 1
        single += n == 1
    partial = 0
    for p in pieces:
        try:
            p.decode("utf-8")
        except UnicodeDecodeError:
            partial += 1
    return dict(tokens=len(pieces), words=len(words), chars=len(text), bytes=len(raw), aksharas=n_ak,
                violations=viol, viol_in_vocab=viol_in_vocab, single=single, byte_tok=sum(1 for p in pieces if len(p) == 1 and p[0] >= 0x80),
                partial=partial, ids=ids)


def job(args):
    name, family, spec, split, files, out = args
    sys.path.insert(0, str(ROOT))
    from tokenization_scorer.metrics import get_metric

    from tokenizer.registry import load

    tok = load(spec if not spec.endswith(".json") else str(ROOT / spec))
    try:
        from tokenizers import Tokenizer

        full_len = Tokenizer.from_file(str(ROOT / spec)).get_vocab_size(True) if spec.endswith(".json") else None
    except Exception:
        full_len = None
    if name == "qwen3":
        full_len = QWEN3_BASE_LEN
    new_vocab = (full_len - QWEN3_BASE_LEN) if (full_len and (family in ("std_ext", "ablation", "sweep", "variant", "final")
                                                               or name == "qwen3")) else ""
    texts = {f: lines(Path(p)) for f, p in files.items()}
    n = min(len(v) for v in texts.values())
    if len({len(v) for v in texts.values()}) != 1:
        print(f"[warn] {split}: forms not parallel {({f: len(v) for f, v in texts.items()})}; using first {n}")
    st = {f: [sent_stats(tok, s) for s in v[:n]] for f, v in texts.items()}
    ref = st.get("eng_Latn")
    rows = []
    for f, ss in st.items():
        T = sum(s["tokens"] for s in ss)
        renyi = get_metric("renyi")(([str(i) for i in s["ids"]] for s in ss), power=2.5)
        ak = sum(s["aksharas"] for s in ss)
        r = dict(tokenizer=name, family=family, vocab_size=full_len or tok.vocab_size, new_vocab=new_vocab, split=split,
                 form=f, n_sent=len(ss), tokens_total=T, tokens_per_sent=T / len(ss),
                 fertility=T / max(1, sum(s["words"] for s in ss)), tokens_per_char=T / sum(s["chars"] for s in ss),
                 parity=T / sum(x["tokens"] for x in ref) if ref else "",
                 parity_median=statistics.median(s["tokens"] / max(1, x["tokens"]) for s, x in zip(ss, ref)) if ref else "",
                 bytes_premium=sum(s["bytes"] for s in ss) / sum(x["bytes"] for x in ref) if ref else "",
                 renyi_eff=float(renyi), single_token_word_rate=sum(s["single"] for s in ss) / max(1, sum(s["words"] for s in ss)),
                 akshara_violations=sum(s["violations"] for s in ss),
                 akshara_violations_in_vocab=sum(s["viol_in_vocab"] for s in ss), aksharas_multi_cp=ak,
                 akshara_split_rate=(sum(s["violations"] for s in ss) / ak) if ak else "",
                 byte_fallback_rate=sum(s["byte_tok"] for s in ss) / T, partial_char_rate=sum(s["partial"] for s in ss) / T,
                 source=Path(files[f]).name, pattern=tok.pattern[:200], date=dt.date.today().isoformat())
        rows.append(r)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return name, split, len(rows)


def key(spec: str, files: dict) -> str:
    parts = [spec] + [f"{f}:{p.stat().st_size}:{int(p.stat().st_mtime)}" for f, p in sorted(files.items())]
    if spec.endswith(".json") and (ROOT / spec).exists():
        p = ROOT / spec
        parts.append(f"{p.stat().st_size}:{int(p.stat().st_mtime)}")
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:12]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results/level1_matrix.csv"))
    a = ap.parse_args()
    parts = ROOT / "results/level1/parts"
    sp = splits()
    jobs, wanted = [], []
    for name, (fam, spec) in TOKENIZERS.items():
        if a.only and name not in a.only:
            continue
        if spec.endswith(".json") and not (ROOT / spec).exists():
            print(f"[skip] {name}: {spec} not built")
            continue
        for split, files in sp.items():
            if "eng_Latn" not in files or len(files) < 2:
                continue
            out = parts / f"{name}__{split}__{key(spec, files)}.csv"
            wanted.append(out)
            if out.exists() and not a.force:
                continue
            jobs.append((name, fam, spec, split, {f: str(p) for f, p in files.items()}, str(out)))
    print(f"{len(jobs)} jobs to run, {len(wanted) - len(jobs)} cached")
    if jobs:
        with Pool(a.workers) as pool:
            for name, split, n in pool.imap_unordered(job, jobs):
                print("done", name, split, n, flush=True)
    # merge every current part (one file per tokenizer x split; newest key wins)
    rows = []
    for out in wanted:
        with open(out, encoding="utf-8") as fh:
            rows += list(csv.DictReader(fh))
    cols = list(dict.fromkeys(k for r in rows for k in r))
    try:
        with open(a.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
    except PermissionError:
        raise SystemExit(f"cannot write {a.out}: it is open in another program (Excel?). Close it and rerun "
                         "(all parts are cached, the merge takes seconds).")
    print("wrote", a.out, len(rows), "rows")


if __name__ == "__main__":
    main()
