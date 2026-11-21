"""Build zips to upload as Kaggle datasets (Kaggle -> Datasets -> New dataset -> upload zip; keep them Private).

  dist/bigo-code.zip    repo code + exported HF tokenizers (tokenizers_hf/), no data/venvs/artifacts
  dist/bigo-test.zip    test sets: FLORES devtest + IN22-Gen (eng/kan), Belebele jsonl, Belebele passages,
                        parity dev (Kannada side). Test-only: never train on these.
  dist/bigo-train.zip   Qwen Stage A/B pre-tokenized bins (data/qwen_train/{A,B}/<tok>/) + Stage C chat jsonl
  dist/bigo-level2.zip  Level 2 bins: from-scratch 32K set + bytes (data/level2/<tok>/) and, as requested,
                        Qwen3 vs ours copied from data/qwen_train/B (qwen3, l123z_16k)

Usage: python notebooks/make_bundles.py [--only code test train level2]
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".venv", ".venv-xlit", "data/raw", "data/processed", "data/level2", "data/qwen_train", "artifacts",
        "third_party", "dist", ".git", ".pytest_cache"}
BIN_FILES = ("train.bin", "val.bin", "eval.json", "meta.json")
LEVEL2_FROM_QWEN_B = ("qwen3", "l123z_16k")


def skipped(rel: Path) -> bool:
    s = rel.as_posix()
    return any(s == k or s.startswith(k + "/") for k in SKIP) or "__pycache__" in rel.parts


def bins(z: zipfile.ZipFile, src: Path, arc_prefix: str, names=None) -> int:
    n = 0
    for d in sorted(p for p in src.iterdir() if p.is_dir() and (p / "meta.json").exists()):
        if names and d.name not in names:
            continue
        for f in BIN_FILES:
            if (d / f).exists():
                z.write(d / f, f"{arc_prefix}{d.name}/{f}")
                n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["code", "test", "train", "level2"])
    a = ap.parse_args()
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    if "code" in a.only:
        with zipfile.ZipFile(dist / "bigo-code.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for p in ROOT.rglob("*"):
                rel = p.relative_to(ROOT)
                if p.is_file() and not skipped(rel):
                    z.write(p, rel.as_posix())
    if "test" in a.only:
        files = sorted((ROOT / "data/raw/test").glob("*.txt")) + sorted((ROOT / "data/raw/test").glob("belebele_*.jsonl")) + \
            sorted((ROOT / "data/processed/test").glob("*.txt")) + sorted((ROOT / "data/processed/dev").glob("*_kan_Knda.txt"))
        files = [p for p in files if not p.name.startswith("flores_dev_")]  # FLORES dev not needed on Kaggle
        with zipfile.ZipFile(dist / "bigo-test.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for p in files:
                z.write(p, p.name)
        print("test files:", [p.name for p in files])
    if "train" in a.only and (ROOT / "data/qwen_train").exists():
        with zipfile.ZipFile(dist / "bigo-train.zip", "w", zipfile.ZIP_STORED) as z:
            for stage in ("A", "B"):
                if (ROOT / "data/qwen_train" / stage).exists():
                    bins(z, ROOT / "data/qwen_train" / stage, f"{stage}/")
            for p in sorted((ROOT / "data/processed/sft").glob("*.json*")):
                z.write(p, f"C/{p.name}")
    if "level2" in a.only:
        with zipfile.ZipFile(dist / "bigo-level2.zip", "w", zipfile.ZIP_STORED) as z:
            n = bins(z, ROOT / "data/level2", "") if (ROOT / "data/level2").exists() else 0
            if (ROOT / "data/qwen_train/B").exists():
                n += bins(z, ROOT / "data/qwen_train/B", "", LEVEL2_FROM_QWEN_B)
        print("level2 files:", n)
    for f in ("bigo-code.zip", "bigo-test.zip", "bigo-train.zip", "bigo-level2.zip"):
        if (dist / f).exists():
            print(dist / f, f"{(dist / f).stat().st_size / 2**20:,.1f} MB")


if __name__ == "__main__":
    main()
