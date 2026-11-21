"""Build zips to upload as Kaggle datasets (Kaggle -> Datasets -> New dataset -> upload zip).

  dist/bigo-code.zip   repo code (no venvs, raw data, artifacts, third_party)
  dist/bigo-test.zip   test text needed on Kaggle (Kannada-script test files)

Usage: python notebooks/make_bundles.py
"""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".venv", ".venv-xlit", "data/raw", "data/processed", "artifacts", "third_party", "dist", ".git", ".pytest_cache"}


def skipped(rel: Path) -> bool:
    s = rel.as_posix()
    return any(s == k or s.startswith(k + "/") for k in SKIP) or "__pycache__" in rel.parts


def main() -> None:
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    with zipfile.ZipFile(dist / "bigo-code.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in ROOT.rglob("*"):
            rel = p.relative_to(ROOT)
            if p.is_file() and not skipped(rel):
                z.write(p, rel.as_posix())
    tests = sorted((ROOT / "data/raw/test").glob("*_kan_Knda.txt")) + \
        sorted((ROOT / "data/processed/test").glob("*_kan_Knda.txt")) + \
        sorted((ROOT / "data/processed/dev").glob("*_kan_Knda.txt"))
    with zipfile.ZipFile(dist / "bigo-test.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in tests:
            z.write(p, p.name)
    for f in ("bigo-code.zip", "bigo-test.zip"):
        print(dist / f, (dist / f).stat().st_size // 1024, "KB")
    print("test files:", [p.name for p in tests])


if __name__ == "__main__":
    main()
