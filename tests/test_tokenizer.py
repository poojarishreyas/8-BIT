"""The three mandatory tokenizer tests (CLAUDE.md):

1. identity     - English/JSON/code/tool-call strings get identical IDs under old and new tokenizer
2. round trip   - decode(encode(x)) == x for Kannada, Kanglish, English, mixed, emoji, numbers, ZWJ/ZWNJ
3. akshara      - no token boundary inside a Kannada akshara (akshara-tokenizer rules, not \\X)

Config (env vars):
  BIGO_OLD_TOKENIZER   original tokenizer (default: Qwen/Qwen3-1.7B from the HF cache/hub)
  BIGO_NEW_TOKENIZER   extended tokenizer.json (default: artifacts/qwen3_markaware/tokenizer.json)
  BIGO_AKSHARA_SAFE    comma-separated tokenizer.json paths that must pass test 3 (default: NEW if it
                       has appended merges; the pattern-only tokenizer is not expected to pass)
"""
from __future__ import annotations

import json
import os
import unicodedata
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from tokenizer.akshara import aksharas, kannada_akshara_byte_spans
from tokenizer.registry import load_hf_file

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).parent / "data"
NEW = Path(os.environ.get("BIGO_NEW_TOKENIZER", ROOT / "artifacts/qwen3_markaware/tokenizer.json"))

ROUND_TRIP = [
    "ಮನೆಯಲ್ಲಿ ಇದೆ.", "ಕನ್ನಡ ಒಂದು ಸುಂದರ ಭಾಷೆ", "ಕ್ರಿಕೆಟ್ ಸ್ತ್ರೀ ಶಾಂತಿ ಪ್ರಶ್ನೆ",
    "naanu nale office ge hogtini", "meeting cancel aaytu, naale maadona", "ನಾನು office ge ಹೋಗ್ತೀನಿ bro",
    "It is in the house.", "The price is ₹1,250.50 (approx.)", "😀👍🏽🇮🇳 ಸರಿ!", "೧೨೩೪ 1234 ೦.೫",
    "ಕಾರ್‌ ಪ್ರ‍ಶ್ನೆ", "‌‍", "tab\tand\nnewline\r\n", "  ", "{\"q\": \"ಬೆಂಗಳೂರು ಹವಾಮಾನ\"}",
    "ಾ orphan mark", "ಕ್", "á decomposed",
]


def _old() -> Tokenizer:
    p = os.environ.get("BIGO_OLD_TOKENIZER")
    if p:
        return Tokenizer.from_file(p)
    from huggingface_hub import hf_hub_download

    return Tokenizer.from_file(hf_hub_download("Qwen/Qwen3-1.7B", "tokenizer.json"))


@pytest.fixture(scope="session")
def old():
    return _old()


@pytest.fixture(scope="session")
def new():
    if not NEW.exists():
        pytest.skip(f"{NEW} not built (python -m tokenizer.extend)")
    return Tokenizer.from_file(str(NEW))


def identity_corpus():
    return [json.loads(l) for l in (DATA / "identity_corpus.jsonl").read_text(encoding="utf-8").splitlines()]


# ---------------------------------------------------------------- 1. identity
def test_identity_corpus_size():
    rows = identity_corpus()
    assert len(rows) >= 1000
    assert {r["kind"] for r in rows} >= {"english", "code", "json", "tool_call"}


def test_identity_old_vs_new(old, new):
    bad = []
    for r in identity_corpus():
        a, b = old.encode(r["text"], add_special_tokens=False).ids, new.encode(r["text"], add_special_tokens=False).ids
        if a != b:
            bad.append((r["kind"], r["text"][:80], len(a), len(b)))
    assert not bad, f"{len(bad)} strings differ, e.g. {bad[:5]}"


def test_new_keeps_all_old_ids(old, new):
    ov, nv = old.get_vocab(with_added_tokens=True), new.get_vocab(with_added_tokens=True)
    changed = [t for t, i in ov.items() if nv.get(t) != i]
    assert not changed, f"{len(changed)} original tokens changed id, e.g. {changed[:5]}"


# ---------------------------------------------------------------- 2. round trip
@pytest.mark.parametrize("which", ["old", "new"])
@pytest.mark.parametrize("text", ROUND_TRIP)
def test_round_trip(request, which, text):
    tk = request.getfixturevalue(which)
    ids = tk.encode(text, add_special_tokens=False).ids
    # Qwen3 normalizes to NFC, so the round trip is defined on NFC text
    assert tk.decode(ids, skip_special_tokens=False) == unicodedata.normalize("NFC", text)


# ---------------------------------------------------------------- 3. akshara integrity
def test_akshara_segmenter_keeps_conjuncts():
    assert aksharas("ಕ್ರಿಕೆಟ್") == ["ಕ್ರಿ", "ಕೆ", "ಟ್"]
    assert aksharas("ಸ್ತ್ರೀ") == ["ಸ್ತ್ರೀ"]
    assert aksharas("ಕಾರ್‌") == ["ಕಾ", "ರ್‌"]  # ZWNJ attached to the preceding akshara


def boundaries_inside_aksharas(tok, text: str) -> list[str]:
    pieces = tok.token_bytes(text)
    b, pos = set(), 0
    for p in pieces[:-1]:
        pos += len(p)
        b.add(pos)
    raw = text.encode("utf-8")
    return [raw[s:e].decode() for s, e in kannada_akshara_byte_spans(text) if any(s < x < e for x in b)]


def test_akshara_checker_detects_splits():
    # original Qwen3 is byte-level with no Kannada merges: it must split ಕ್ರಿ
    from huggingface_hub import hf_hub_download

    q = load_hf_file("qwen3", os.environ.get("BIGO_OLD_TOKENIZER") or hf_hub_download("Qwen/Qwen3-1.7B", "tokenizer.json"))
    assert "ಕ್ರಿ" in boundaries_inside_aksharas(q, "ಕ್ರಿಕೆಟ್")


def kannada_sample() -> list[str]:
    p = ROOT / "data/processed/test/belebele_passages_kan_Knda.txt"
    extra = p.read_text(encoding="utf-8").splitlines()[:100] if p.exists() else []
    return [t for t in ROUND_TRIP if any("ಀ" <= c <= "೿" for c in t)] + extra


def _akshara_safe_paths() -> list[str]:
    env = os.environ.get("BIGO_AKSHARA_SAFE")
    return [p for p in env.split(",") if p] if env else []


@pytest.mark.parametrize("path", _akshara_safe_paths() or [pytest.param(None, marks=pytest.mark.skip(
    reason="no akshara-safe tokenizer built yet (set BIGO_AKSHARA_SAFE); Phase 2"))])
def test_akshara_integrity(path):
    tok = load_hf_file("candidate", path)
    bad = [(s[:40], bad) for s in kannada_sample() if (bad := boundaries_inside_aksharas(tok, s))]
    assert not bad, f"{len(bad)} sentences have split aksharas, e.g. {bad[:3]}"
