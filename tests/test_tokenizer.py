"""The three mandatory tokenizer tests (CLAUDE.md):

1. identity     - English/JSON/code/tool-call strings get identical IDs under old and new tokenizer
2. round trip   - decode(encode(x)) == x for Kannada, Kanglish, English, mixed, emoji, numbers, ZWJ/ZWNJ
3. akshara      - no token boundary inside a Kannada akshara (akshara-tokenizer rules, not \\X)

Config (env vars):
  BIGO_OLD_TOKENIZER   original tokenizer (default: Qwen/Qwen3-1.7B from the HF cache/hub)
  BIGO_NEW_TOKENIZER   extended tokenizer.json (default: artifacts/final/bigo_final/tokenizer.json)
  BIGO_AKSHARA_SAFE    comma-separated tokenizer.json paths for test 3 (default: NEW)

Test 3, precisely: an akshara that never occurred in training cannot be a single token, so *some* boundary
must fall inside it (and Qwen3's own old merge 'Ġà²' fuses a word's leading space into part of it). The
guarantee we test is: no akshara that IS in the tokenizer's inventory (single token in the form it occurs)
is ever split in running text - this count must be 0 on all test Kannada. The total number of split
(out-of-inventory) aksharas is reported to results/tests_akshara.json.
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
NEW = Path(os.environ.get("BIGO_NEW_TOKENIZER", ROOT / "artifacts/final/bigo_final/tokenizer.json"))
RAW, PROC = ROOT / "data/raw/test", ROOT / "data/processed/test"


def find_test_files(pattern: str) -> list[Path]:
    return sorted([*RAW.glob(pattern), *PROC.glob(pattern)])


def file_lines(p: Path) -> list[str]:
    return [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

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
    kinds = {k: sum(r["kind"] == k for r in rows) for k in ("english", "code", "json", "tool_call")}
    assert kinds["json"] + kinds["tool_call"] >= 50 and kinds["code"] > 0
    assert kinds["english"] + sum(len(file_lines(p)) for p in find_test_files("*_eng_Latn.txt")) >= 500


def test_identity_english_test_sets(old, new):
    """Every English line of every test set (FLORES devtest, IN22-Gen, Belebele passages, Kanglish-eng)."""
    bad = []
    for p in find_test_files("*_eng_Latn.txt") + find_test_files("kanglish_handwritten_eng.txt"):
        for line in file_lines(p):
            if old.encode(line, add_special_tokens=False).ids != new.encode(line, add_special_tokens=False).ids:
                bad.append((p.name, line[:60]))
    assert not bad, f"{len(bad)} English test lines changed, e.g. {bad[:3]}"


def test_identity_old_vs_new(old, new):
    bad = []
    for r in identity_corpus():
        a, b = old.encode(r["text"], add_special_tokens=False).ids, new.encode(r["text"], add_special_tokens=False).ids
        if a != b:
            bad.append((r["kind"], r["text"][:80], len(a), len(b)))
    assert not bad, f"{len(bad)} strings differ, e.g. {bad[:5]}"


ZW_NON_KANNADA = [
    "👨‍👩‍👧 family", "🏳️‍🌈 flag", "word‌ next", "a‍b", "می‌خواهم",  # Persian ZWNJ
    "क्‍ष", "x ‌ y", "‍", "emoji 👍🏽‍ ok",
]


@pytest.mark.parametrize("text", ZW_NON_KANNADA)
def test_identity_zw_outside_kannada(old, new, text):
    # the ZW-aware split rule may only change text where ZWJ/ZWNJ follows a Kannada character
    assert old.encode(text, add_special_tokens=False).ids == new.encode(text, add_special_tokens=False).ids


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
    out = [t for t in ROUND_TRIP if any("ಀ" <= c <= "೿" for c in t)]
    for p in find_test_files("*_kan_Knda.txt"):
        if not p.name.startswith("flores_dev_"):
            out += file_lines(p)
    return out


def _akshara_safe_paths() -> list[str]:
    env = os.environ.get("BIGO_AKSHARA_SAFE")
    if env:
        return [p for p in env.split(",") if p]
    return [str(NEW)] if NEW.exists() else []


def akshara_report(path: str) -> dict:
    tok = load_hf_file("candidate", path)
    total = in_inv = n = 0
    examples = {}
    for s in kannada_sample():
        s = tok.normalize(s)
        raw = s.encode("utf-8")
        for a_bytes in kannada_akshara_byte_spans(s):
            n += 1
        for st, en in [(st, en) for st, en in kannada_akshara_byte_spans(s)
                       if any(st < b < en for b in _inner(tok, s))]:
            total += 1
            a = raw[st:en].decode("utf-8")
            sp = st > 0 and raw[st - 1:st] == b" "
            if len(tok.encode((" " if sp else "") + a)) == 1:
                in_inv += 1
                examples.setdefault(a, s[:60])
    return dict(tokenizer=path, multi_codepoint_aksharas=n, split_total=total, split_in_inventory=in_inv,
                examples_in_inventory=list(examples.items())[:5])


def _inner(tok, s: str) -> set:
    b, pos = set(), 0
    pieces = tok.token_bytes(s)
    for x in pieces[:-1]:
        pos += len(x)
        b.add(pos)
    return b


@pytest.mark.parametrize("path", _akshara_safe_paths() or [pytest.param(None, marks=pytest.mark.skip(
    reason="bigo_final not built yet (or set BIGO_AKSHARA_SAFE)"))])
def test_akshara_integrity(path):
    r = akshara_report(path)
    out = ROOT / "results/tests_akshara.json"
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    prev[Path(path).parent.name] = r
    out.write_text(json.dumps(prev, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n{r['split_total']} of {r['multi_codepoint_aksharas']} aksharas split (all out-of-inventory); "
          f"in-inventory splits: {r['split_in_inventory']}")
    assert r["split_in_inventory"] == 0, r["examples_in_inventory"]


# ---------------------------------------------------------------- round trip on every test file, every form
@pytest.mark.parametrize("which", ["old", "new"])
def test_round_trip_all_test_sets(request, which):
    tk = request.getfixturevalue(which)
    bad = []
    files = [p for p in find_test_files("*.txt")]
    assert files, "no test files found"
    for p in files:
        for line in file_lines(p):
            ids = tk.encode(line, add_special_tokens=False).ids
            if tk.decode(ids, skip_special_tokens=False) != unicodedata.normalize("NFC", line):
                bad.append((p.name, line[:50]))
    assert not bad, f"{len(bad)} lines fail the round trip, e.g. {bad[:3]}"
