"""Layer 3: stem | suffix boundaries for Kannada words, akshara-aligned.

Seed list = common Kannada case endings, plural, clitics and finite-verb endings, written as
akshara strings in the form they take after sandhi (e.g. ಮನೆ+ಅಲ್ಲಿ -> ಮನೆ|ಯಲ್ಲಿ, ಮರ+ಅಲ್ಲಿ -> ಮರ|ದಲ್ಲಿ).
Hand-written (no external licence). Because Kannada sandhi fuses the stem's last vowel into the
suffix's first akshara (ಊರು+ಇನ -> ಊರಿನ), boundaries are approximate; we only split where the
suffix starts on an akshara boundary, and require a stem of >= MIN_STEM aksharas.

`split(aksharas)` returns the index where the suffix starts, or None.
Morfessor is the planned data-driven alternative (ablation); this list is the seed.
"""
from __future__ import annotations

from tokenizer.akshara import aksharas as seg

MIN_STEM = 2

SEED_SUFFIXES = [
    # plural + case stacks (longest first is enforced in code)
    "ಗಳಲ್ಲಿ", "ಗಳನ್ನು", "ಗಳಿಗೆ", "ಗಳಿಂದ", "ಗಳೊಂದಿಗೆ", "ಗಳಿಗಾಗಿ", "ಗಳೇ", "ಗಳೂ", "ಗಳು", "ಗಳ",
    # locative
    "ಯಲ್ಲಿ", "ದಲ್ಲಿ", "ನಲ್ಲಿ", "ವಲ್ಲಿ", "ರಲ್ಲಿ", "ಲ್ಲಿ",
    # ablative / instrumental
    "ಯಿಂದ", "ದಿಂದ", "ನಿಂದ", "ರಿಂದ", "ವಿಂದ",
    # dative
    "ಕ್ಕಾಗಿ", "ಗಾಗಿ", "ಕ್ಕೆ", "ಗೆ", "ರಿಗೆ", "ನಿಗೆ", "ಳಿಗೆ",
    # accusative
    "ಯನ್ನು", "ವನ್ನು", "ನನ್ನು", "ರನ್ನು", "ದನ್ನು", "ಳನ್ನು", "ನ್ನು",
    # genitive (multi-akshara forms only; bare ದ/ನ/ಯ are too ambiguous)
    "ಯಿನ", "ದಿನ",
    # sociative / comitative / postpositions written joined
    "ಯೊಂದಿಗೆ", "ದೊಂದಿಗೆ", "ನೊಂದಿಗೆ", "ರೊಂದಿಗೆ", "ಜೊತೆ", "ಯೊಡನೆ", "ದೊಡನೆ",
    # nominative markers
    "ವು", "ಯು",
    # clitics
    "ವೂ", "ಯೂ", "ವೇ", "ಯೇ", "ನ್ನೂ",
    # finite verb endings (present/future, past, negative, modal)
    "ತ್ತಾರೆ", "ತ್ತಾನೆ", "ತ್ತಾಳೆ", "ತ್ತದೆ", "ತ್ತೇನೆ", "ತ್ತೀರಿ", "ತ್ತೀಯ", "ತ್ತೇವೆ", "ತ್ತವೆ",
    "ದ್ದಾರೆ", "ದ್ದಾನೆ", "ದ್ದಾಳೆ", "ದ್ದೇನೆ", "ದ್ದೇವೆ",
    "ದರು", "ದನು", "ದಳು", "ದೆನು", "ದೆವು", "ಯಿತು", "ದಿತು",
    "ಲಿಲ್ಲ", "ವುದಿಲ್ಲ", "ಬೇಕು", "ಬಹುದು", "ಬೇಡ", "ಲಾಗಿದೆ", "ಲಾಗುತ್ತದೆ", "ಲಾಯಿತು",
    "ವುದು",
]
_SUFFIX_AKS = sorted({tuple(seg(s)) for s in SEED_SUFFIXES if s and not s.startswith("್")},
                     key=len, reverse=True)


def split(aks: list[str], min_stem: int = MIN_STEM) -> int | None:
    """Index in `aks` (akshara list of one word, no prefix) where the longest seed suffix starts."""
    n = len(aks)
    for suf in _SUFFIX_AKS:
        k = len(suf)
        if n - k >= min_stem and tuple(aks[n - k:]) == suf:
            return n - k
    return None


def demo(words: list[str]) -> list[str]:
    out = []
    for w in words:
        a = seg(w)
        i = split(a)
        out.append(w if i is None else "".join(a[:i]) + "|" + "".join(a[i:]))
    return out


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    print(demo(["ಮನೆಯಲ್ಲಿ", "ಮರದಲ್ಲಿ", "ಊರಿನಲ್ಲಿ", "ಮಕ್ಕಳಿಗೆ", "ಪುಸ್ತಕಗಳನ್ನು", "ಹೋಗುತ್ತಾರೆ", "ಬೆಂಗಳೂರಿನಿಂದ",
                "ಕನ್ನಡ", "ಮನೆ", "ಸರ್ಕಾರದ", "ಮಾಡಲಾಗಿದೆ"]))
