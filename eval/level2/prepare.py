"""Pre-tokenize Level 2 data on the laptop (uploaded to Kaggle as a dataset; Kaggle only needs torch).

For each tokenizer, writes <out>/<name>/:
  train.bin, val.bin   uint16 token ids (uint32 if vocab+1 > 65535); docs separated by EOS = vocab_size
  eval.json            {form: [[ids...], ...]} plus char counts, for BPC (one sentence per item)
  meta.json            vocab_size, eos_id, dtype, per-language chars/tokens, tokenizer path, date

All tokenizers get the *same documents in the same order* (shuffled with seed 0 across languages) and the
same character budget per language, so models are matched on training characters.

Usage:
  python -m eval.level2.prepare --out data/level2 --chars-per-lang 25000000 \
     --tok plain_32k=artifacts/scratch/plain_32k/tokenizer.json --tok bytes=bytes ... \
     --lang eng_Latn=data/processed/train/eng_Latn.txt --lang kan_Knda=... --lang kan_Latn=... \
     --eval kan_Knda.flores_devtest=data/raw/test/flores_devtest_kan_Knda.txt ...
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import unicodedata
from pathlib import Path

import numpy as np

VAL_FRACTION = 0.01


def read_docs(path: str, budget: int) -> list[str]:
    docs, n = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            docs.append(line)
            n += len(line)
            if n >= budget:
                break
    return docs


def load_tok(spec: str):
    from tokenizers import Tokenizer

    if spec == "bytes":
        from tokenizer.build_scratch import byte_base

        return byte_base(r"\s+|\S+")  # bytes only; split rule is irrelevant without merges
    return Tokenizer.from_file(spec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/level2")
    ap.add_argument("--tok", action="append", required=True, help="name=tokenizer.json or name=bytes")
    ap.add_argument("--lang", action="append", required=True, help="name=train_file")
    ap.add_argument("--eval", action="append", default=[], help="form.split=file (one sentence per line)")
    ap.add_argument("--chars-per-lang", type=int, default=25_000_000)
    ap.add_argument("--lang-chars", action="append", default=[], help="name=chars override per language")
    ap.add_argument("--eos-token", default=None, help='use an existing token as doc separator (Qwen: "<|endoftext|>")')
    ap.add_argument("--no-eval", action="store_true")
    a = ap.parse_args()

    langs = dict(x.split("=", 1) for x in a.lang)
    # one shared doc list: (lang, text), shuffled once; last VAL_FRACTION of docs per language -> val
    train, val = [], []
    for l, p in langs.items():
        d = read_docs(p, int(dict(x.split("=", 1) for x in a.lang_chars).get(l, a.chars_per_lang)))
        k = max(1, int(len(d) * VAL_FRACTION))
        train += [(l, t) for t in d[:-k]]
        val += [(l, t) for t in d[-k:]]
    random.Random(0).shuffle(train)
    evals = {}
    for e in a.eval:
        name, p = e.split("=", 1)
        if Path(p).exists():
            evals[name] = [unicodedata.normalize("NFC", s) for s in Path(p).read_text(encoding="utf-8").splitlines() if s.strip()]
        else:
            print(f"[skip eval] {name}: {p} missing")

    for spec in a.tok:
        name, path = spec.split("=", 1)
        tk = load_tok(path)
        V = tk.get_vocab_size()
        if a.eos_token:
            eos = tk.token_to_id(a.eos_token)
            assert eos is not None, f"{a.eos_token} not in {path}"
            n_emb = V
        else:
            eos, n_emb = V, V + 1
        dtype = np.uint16 if n_emb <= 65535 else np.uint32
        d = Path(a.out) / name
        d.mkdir(parents=True, exist_ok=True)
        meta = dict(name=name, tokenizer=path, vocab_size=V, eos_id=eos, n_vocab_with_eos=n_emb,
                    dtype=np.dtype(dtype).name, date=dt.date.today().isoformat(), langs={})
        for split, docs in (("train", train), ("val", val)):
            stats = {}
            with (d / f"{split}.bin").open("wb") as f:
                for k in range(0, len(docs), 1000):  # chunked: bounded memory
                    chunk = docs[k:k + 1000]
                    encs = tk.encode_batch([t for _, t in chunk], add_special_tokens=False)
                    for (l, t), e in zip(chunk, encs):
                        np.asarray(e.ids + [eos], dtype=dtype).tofile(f)
                        s = stats.setdefault(l, [0, 0, 0])
                        s[0] += len(t)
                        s[1] += len(e.ids)
                        s[2] += 1
            meta["langs"][split] = {l: dict(chars=c, tokens=n, docs=k, chars_per_token=c / n) for l, (c, n, k) in stats.items()}
            meta[f"{split}_tokens"] = sum(s[1] + s[2] for s in stats.values())  # + EOS per doc
            meta[f"{split}_chars"] = sum(s[0] for s in stats.values())
        ev = {}
        for form, sents in ({} if a.no_eval else evals).items():
            encs = tk.encode_batch(sents, add_special_tokens=False)
            ev[form] = dict(ids=[e.ids for e in encs], chars=[len(s) for s in sents])
        (d / "eval.json").write_text(json.dumps(ev), encoding="utf-8")
        (d / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print(name, "V", V, "train tokens", f"{meta['train_tokens']:,}", "chars", f"{meta['train_chars']:,}",
              {l: round(v["chars_per_token"], 2) for l, v in meta["langs"]["train"].items()})


if __name__ == "__main__":
    main()
