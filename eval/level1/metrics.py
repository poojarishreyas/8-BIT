"""Level 1 (no model) tokenizer metrics.

Per tokenizer x text form, over a list of sentences (parallel across forms):
  tokens_per_sent     mean tokens per sentence
  parity              sum(tokens in form) / sum(tokens in reference form, usually English)
  parity_median       median of per-sentence ratios
  fertility           tokens per whitespace-separated word
  chars_per_token     Unicode code points per token
  bytes_per_token     UTF-8 bytes per token
  akshara_split_rate  share of multi-code-point Kannada aksharas with a token boundary inside them
  partial_char_rate   share of tokens whose bytes are not whole UTF-8 characters
  byte_fallback_rate  share of tokens that are a single byte which is not a full character
                      (SentencePiece <0x..> fallback, or a lone byte in byte-level BPE)
"""
from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass

from tokenizer.akshara import kannada_akshara_byte_spans
from tokenizer.registry import Tok


@dataclass
class SentStats:
    tokens: int
    words: int
    chars: int
    bytes: int
    aksharas: int
    aksharas_split: int
    partial_tokens: int
    byte_tokens: int


def _is_whole_chars(b: bytes) -> bool:
    try:
        b.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def sentence_stats(tok: Tok, text: str) -> SentStats:
    text = tok.normalize(text)
    pieces = tok.token_bytes(text)
    # interior token boundaries as byte offsets
    bounds = set()
    pos = 0
    for p in pieces[:-1]:
        pos += len(p)
        bounds.add(pos)
    spans = kannada_akshara_byte_spans(text)
    split = sum(1 for s, e in spans if any(s < b < e for b in bounds)) if bounds else 0
    partial = sum(1 for p in pieces if not _is_whole_chars(p))
    byte_tok = sum(1 for p in pieces if len(p) == 1 and p[0] >= 0x80)
    return SentStats(
        tokens=len(pieces),
        words=len(text.split()),
        chars=len(text),
        bytes=len(text.encode("utf-8")),
        aksharas=len(spans),
        aksharas_split=split,
        partial_tokens=partial,
        byte_tokens=byte_tok,
    )


def aggregate(stats: list[SentStats], ref: list[SentStats] | None = None) -> dict:
    T = sum(s.tokens for s in stats)
    out = dict(
        n_sent=len(stats),
        tokens_total=T,
        tokens_per_sent=T / len(stats),
        fertility=T / max(1, sum(s.words for s in stats)),
        chars_per_token=sum(s.chars for s in stats) / T,
        bytes_per_token=sum(s.bytes for s in stats) / T,
        akshara_split_rate=(sum(s.aksharas_split for s in stats) / sum(s.aksharas for s in stats))
        if sum(s.aksharas for s in stats) else None,
        partial_char_rate=sum(s.partial_tokens for s in stats) / T,
        byte_fallback_rate=sum(s.byte_tokens for s in stats) / T,
    )
    if ref is not None:
        assert len(ref) == len(stats), "parity needs parallel sentences"
        out["parity"] = T / sum(r.tokens for r in ref)
        out["parity_median"] = statistics.median(s.tokens / max(1, r.tokens) for s, r in zip(stats, ref))
    return out


__all__ = ["SentStats", "sentence_stats", "aggregate", "asdict"]
