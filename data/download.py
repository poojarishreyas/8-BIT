"""Download Phase-1 data: training samples (streamed, capped by characters) and test sets.

Training text (written to <out>/train_raw/):
  kan_Knda.txt   Sangraha verified/kan       (Kannada script)
  kan_Latn.txt   Sangraha synthetic/kan_Latn (romanized Kannada)
  eng_Latn.txt   FineWeb-Edu sample-10BT     (English)
Test text (written to <out>/test/):
  belebele_{eng_Latn,kan_Knda}.jsonl   (ungated)
  flores_devtest_{eng_Latn,kan_Knda}.txt  (gated: openlanguagedata/flores_plus, needs HF login)
  in22gen_{eng_Latn,kan_Knda}.txt         (gated: ai4bharat/IN22-Gen, needs HF login)

One document per line, newlines inside a document replaced by a space, NFC-normalized.

Usage: python -m data.download --out data/raw --max-chars 30000000
"""
from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

# Parquet files are read row-group by row-group over HfFileSystem (datasets streaming stalls on
# these large shards). Files are listed in order and read until --max-chars is reached.
TRAIN_SOURCES = {
    "kan_Knda": dict(repo="ai4bharat/sangraha", files=[f"verified/kan/data-{i}.parquet" for i in range(21)], field="text"),
    "kan_Latn": dict(repo="ai4bharat/sangraha",
                     files=[f"synthetic/kan_Latn/wiki_kan_Latn_{i:04d}_of_0063.parquet" for i in range(63)], field="text"),
    "eng_Latn": dict(repo="HuggingFaceFW/fineweb-edu", files=["sample/10BT/000_00000.parquet"], field="text"),
    # Sangraha verified/kan has ZWNJ stripped (0.00 per 10k chars vs 33 in FLORES devtest, 80 in IN22-Gen);
    # Kannada Wikipedia keeps it (13 per 10k), so it is mixed into Kannada training text.
    "kan_Knda_wiki": dict(repo="wikimedia/wikipedia", files=["20231101.kn/train-00000-of-00001.parquet"], field="text"),
}


def norm(s: str) -> str:
    return unicodedata.normalize("NFC", " ".join(s.split()))


def stream_train(lang: str, out: Path, max_chars: int) -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    src = TRAIN_SOURCES[lang]
    fs = HfFileSystem()
    out.parent.mkdir(parents=True, exist_ok=True)
    n_chars = n_docs = 0
    with out.open("w", encoding="utf-8") as f:
        for fname in src["files"]:
            with fs.open(f"datasets/{src['repo']}/{fname}", "rb", block_size=8 << 20) as fh:
                pf = pq.ParquetFile(fh)
                for batch in pf.iter_batches(batch_size=2048, columns=[src["field"]]):
                    for t in batch.column(0).to_pylist():
                        t = norm(t or "")
                        if len(t) < 50:
                            continue
                        f.write(t + "\n")
                        n_chars += len(t)
                        n_docs += 1
                        if n_chars >= max_chars:
                            break
                    if n_chars >= max_chars:
                        break
            if n_chars >= max_chars:
                break
    print(f"{lang}: {n_docs} docs, {n_chars:,} chars -> {out}")


def get_belebele(out_dir: Path) -> None:
    from huggingface_hub import hf_hub_download

    for lang in ("eng_Latn", "kan_Knda"):
        p = hf_hub_download("facebook/belebele", f"data/{lang}.jsonl", repo_type="dataset")
        dst = out_dir / f"belebele_{lang}.jsonl"
        dst.write_bytes(Path(p).read_bytes())
        print("belebele", lang, "->", dst)


def get_flores(out_dir: Path) -> None:
    from datasets import load_dataset

    for lang in ("eng_Latn", "kan_Knda"):
        for split in ("dev", "devtest"):
            ds = load_dataset("openlanguagedata/flores_plus", lang, split=split)
            dst = out_dir / f"flores_{split}_{lang}.txt"
            dst.write_text("\n".join(norm(t) for t in ds["text"]) + "\n", encoding="utf-8")
            print("flores", split, lang, len(ds), "->", dst)


def get_in22(out_dir: Path) -> None:
    from datasets import load_dataset

    ds = load_dataset("ai4bharat/IN22-Gen", "default", split="test")  # 1,024 rows, one column per language
    for lang in ("eng_Latn", "kan_Knda"):
        dst = out_dir / f"in22gen_{lang}.txt"
        dst.write_text("\n".join(norm(t) for t in ds[lang]) + "\n", encoding="utf-8")
        print("in22", lang, len(ds), "->", dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--max-chars", type=int, default=30_000_000, help="per training language")
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of: kan_Knda kan_Latn eng_Latn belebele flores in22")
    a = ap.parse_args()
    out = Path(a.out)
    test = out / "test"
    test.mkdir(parents=True, exist_ok=True)
    steps = a.only or ["belebele", "flores", "in22", *TRAIN_SOURCES]
    for s in steps:
        try:
            if s in TRAIN_SOURCES:
                stream_train(s, out / "train_raw" / f"{s}.txt", a.max_chars)
            elif s == "belebele":
                get_belebele(test)
            elif s == "flores":
                get_flores(test)
            elif s == "in22":
                get_in22(test)
        except Exception as e:  # gated sets fail without HF login; keep going
            print(f"[skip] {s}: {type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    main()
