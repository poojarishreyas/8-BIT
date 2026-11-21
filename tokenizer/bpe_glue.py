"""Glue: learn BPE merges over arbitrary *units* with swiss-ai/parity-aware-bpe, unchanged.

parity-aware-bpe (a subword-nmt fork) treats each character of a dict-input word as one starting
symbol. We map every unit (an existing Qwen3 token, a Kannada character, an akshara...) to one
private-use code point, write `word count` dict files per language, run its `learn_bpe`, and map
the learned pairs back to unit strings.

  learn(words_per_lang, num_merges, dev_per_lang=None, min_frequency=2) -> list[(left, right)]

`words_per_lang`: list (one per language) of Counter{tuple(unit_str, ...): count}.
With one language this is standard BPE. With several, each step picks the language that is
currently worst compressed on `dev_per_lang` (parity-aware BPE, arXiv 2508.04796).
Returned pairs are tuples of unit strings concatenated (i.e. token strings in the units' alphabet).
"""
from __future__ import annotations

import io
import logging
import tempfile
from collections import Counter
from pathlib import Path

# Supplementary Private Use Area-A then -B: 131,068 code points, none are whitespace.
_PUA = [*range(0xF0000, 0xFFFFE), *range(0x100000, 0x10FFFE)]


class UnitMap:
    def __init__(self):
        self.to_pua: dict[str, str] = {}
        self.from_pua: dict[str, str] = {}

    def enc(self, unit: str) -> str:
        c = self.to_pua.get(unit)
        if c is None:
            if len(self.to_pua) >= len(_PUA):
                raise OverflowError("too many distinct units")
            c = chr(_PUA[len(self.to_pua)])
            self.to_pua[unit] = c
            self.from_pua[c] = unit
        return c

    def dec(self, s: str) -> str:
        return "".join(self.from_pua[c] for c in s)


def _write_dict(words: Counter, um: UnitMap, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for w, c in words.items():
            if c > 0 and w:
                f.write("".join(um.enc(u) for u in w) + " " + str(int(c)) + "\n")


def learn(words_per_lang: list[Counter], num_merges: int, dev_per_lang: list[Counter] | None = None,
          min_frequency: int = 2, verbose: bool = False) -> list[tuple[str, str]]:
    from parity_aware_bpe import parity_aware_learn_bpe as pa

    pa.logger.setLevel(logging.WARNING)
    um = UnitMap()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ins, devs = [], []
        for i, w in enumerate(words_per_lang):
            _write_dict(w, um, td / f"train{i}.dict")
            ins.append(td / f"train{i}.dict")
        if dev_per_lang:
            for i, w in enumerate(dev_per_lang):
                _write_dict(w, um, td / f"dev{i}.dict")
                devs.append(td / f"dev{i}.dict")
        out = io.StringIO()
        fin = [p.open(encoding="utf-8") for p in ins]
        fdev = [p.open(encoding="utf-8") for p in devs] or None
        try:
            pa.learn_bpe(fin, out, fdev, num_merges, min_frequency=min_frequency, verbose=verbose,
                         is_dict=True, ratio=None if fdev else [1.0] * len(fin))
        finally:
            for f in fin + (fdev or []):
                f.close()
    merges = []
    for line in out.getvalue().splitlines():
        if line.startswith("#version") or not line.strip():
            continue
        a, b = line.split(" ")
        merges.append((um.dec(a), um.dec(b)))
    return merges


def apply(word: tuple[str, ...], ranks: dict[tuple[str, str], int]) -> list[str]:
    """Plain BPE application (lowest rank first) of `ranks` to one unit sequence."""
    w = list(word)
    while len(w) > 1:
        best, bi = None, -1
        for i in range(len(w) - 1):
            r = ranks.get((w[i], w[i + 1]))
            if r is not None and (best is None or r < best):
                best, bi = r, i
        if best is None:
            break
        w[bi:bi + 2] = [w[bi] + w[bi + 1]]
    return w
