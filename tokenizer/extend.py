"""Edit a Qwen3-style tokenizer.json in place of using add_tokens().

Phase 1 provides `make_mark_aware`: swap `\\p{L}+` -> `[\\p{L}\\p{M}]+` in the Split regex
(the single occurrence), keeping vocab, merges and IDs unchanged. Appending merges comes in Phase 2.

Usage: python -m tokenizer.extend --base Qwen/Qwen3-1.7B --out artifacts/qwen3_markaware
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LETTERS = r"\p{L}+"
MARK_AWARE = r"[\p{L}\p{M}]+"
# ZWNJ/ZWJ are category Cf, so `[\p{L}\p{M}]+` cuts ಕಾರ್‌ನಲ್ಲಿ before the ZWNJ (91% of remaining
# akshara splits on FLORES). Let ZW joiners continue a word only right after a Kannada character,
# so emoji ZWJ sequences and Latin text tokenize exactly as before.
MARK_ZW = r"[\p{L}\p{M}]+(?:(?<=[\x{0C80}-\x{0CFF}])[\x{200C}\x{200D}]+[\p{L}\p{M}]*)*"


def mark_zw_pattern(pattern: str) -> str:
    return mark_aware_pattern(pattern).replace(MARK_AWARE, MARK_ZW, 1)


def mark_aware_pattern(pattern: str) -> str:
    # Qwen3/Llama-3 pattern has exactly one `\p{L}+` (the other `\p{L}`s sit in negated classes,
    # which the greedy `[\p{L}\p{M}]+` alternative pre-empts for marks that follow a letter).
    n = pattern.count(LETTERS)
    if n != 1:
        raise ValueError(f"expected 1 occurrence of {LETTERS!r} in pre-tokenizer pattern, found {n}: {pattern}")
    return pattern.replace(LETTERS, MARK_AWARE)


def _split_nodes(pre: dict):
    if pre.get("type") == "Split":
        yield pre
    for p in pre.get("pretokenizers", []) or []:
        yield from _split_nodes(p)


def make_mark_aware(tj: dict, zw: bool = False) -> dict:
    tj = json.loads(json.dumps(tj))
    splits = list(_split_nodes(tj["pre_tokenizer"]))
    if len(splits) != 1:
        raise ValueError(f"expected exactly one Split pre-tokenizer, found {len(splits)}")
    s = splits[0]
    s["pattern"]["Regex"] = (mark_zw_pattern if zw else mark_aware_pattern)(s["pattern"]["Regex"])
    return tj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B", help="HF repo id or path to tokenizer.json")
    ap.add_argument("--out", default="artifacts/qwen3_markaware")
    ap.add_argument("--zw", action="store_true", help="ZWJ/ZWNJ-aware variant (mark_zw)")
    a = ap.parse_args()
    p = Path(a.base)
    if not p.exists():
        from huggingface_hub import hf_hub_download

        p = Path(hf_hub_download(a.base, "tokenizer.json"))
    tj = make_mark_aware(json.loads(p.read_text(encoding="utf-8")), zw=a.zw)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tokenizer.json").write_text(json.dumps(tj, ensure_ascii=False), encoding="utf-8")
    from tokenizers import Tokenizer

    Tokenizer.from_file(str(out / "tokenizer.json"))  # must load
    print("wrote", out / "tokenizer.json")


if __name__ == "__main__":
    main()
