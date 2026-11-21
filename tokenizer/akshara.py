"""Akshara spans for Kannada text, via `akshara-tokenizer` (virama-aware; NOT regex \\X).

Fix-up on top of akshara-tokenizer: ZWJ/ZWNJ are returned as standalone pieces there; we attach
them to the preceding Kannada akshara (e.g. ಕಾರ್‌ keeps its ZWNJ), so a token boundary just
before a ZWNJ counts as splitting the akshara.
"""
from __future__ import annotations

from akshara_tokenizer import segment_aksharas

ZW = {"‌", "‍"}


def is_kannada(ch: str) -> bool:
    return "ಀ" <= ch <= "೿"


def aksharas(text: str) -> list[str]:
    segs = segment_aksharas(text)
    out: list[str] = []
    for s in segs:
        if s in ZW and out and is_kannada(out[-1][-1]):
            out[-1] += s
        else:
            out.append(s)
    assert "".join(out) == text, "akshara segmentation must cover the text exactly"
    return out


def kannada_akshara_byte_spans(text: str, multi_only: bool = True) -> list[tuple[int, int]]:
    """(start, end) UTF-8 byte offsets of Kannada aksharas. multi_only: only aksharas with >1 code point
    (single-code-point aksharas cannot be split at a character boundary)."""
    spans = []
    pos = 0
    for a in aksharas(text):
        n = len(a.encode("utf-8"))
        if any(is_kannada(c) for c in a) and (len(a) > 1 or not multi_only):
            spans.append((pos, pos + n))
        pos += n
    return spans
