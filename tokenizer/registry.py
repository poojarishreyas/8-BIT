"""Uniform access to baseline + our tokenizers.

Every tokenizer is wrapped as `Tok` with:
  .name, .vocab_size, .pattern (pre-tokenizer description, recorded in results)
  .encode(text) -> list[int]           (no special tokens)
  .token_bytes(text) -> list[bytes]    (exact bytes each token covers; b"".join == text bytes)

token_bytes is what boundary metrics (akshara-split rate, partial-char rate) use, so they
are computed the same way for byte-level BPE, SentencePiece-style byte-fallback BPE and tiktoken.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable

# name -> (kind, source). Gated originals have ungated mirrors with identical tokenizer files.
BASELINES: dict[str, tuple[str, str]] = {
    "cl100k_base": ("tiktoken", "cl100k_base"),
    "o200k_base": ("tiktoken", "o200k_base"),
    "qwen3": ("hf", "Qwen/Qwen3-1.7B"),
    "llama3": ("hf", "unsloth/Llama-3.2-1B-Instruct"),  # mirror of meta-llama/Llama-3.2-1B-Instruct
    "gemma3": ("hf", "unsloth/gemma-3-1b-it"),  # mirror of google/gemma-3-1b-it
    "sarvam1": ("hf", "sarvamai/sarvam-1"),
    "indictok256k": ("hf", "vivekvar/indictok-256k"),
}


@lru_cache(maxsize=1)
def _byte_decoder() -> dict[str, int]:
    """Inverse of GPT-2 bytes_to_unicode (used by HF ByteLevel)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_BYTE_TOKEN = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


@dataclass
class Tok:
    name: str
    vocab_size: int
    pattern: str
    encode: Callable[[str], list[int]]
    _piece_bytes: Callable[[list[int]], list[bytes]] = field(repr=False)
    normalize: Callable[[str], str] = field(default=lambda s: s, repr=False)

    def token_bytes(self, text: str) -> list[bytes]:
        ids = self.encode(text)
        pieces = self._piece_bytes(ids)
        target = self.normalize(text).encode("utf-8")
        got = b"".join(pieces)
        if got != target and got[:1] == b" " and got[1:] == target:  # Metaspace prefix space
            pieces = [pieces[0][1:]] + pieces[1:]
            if pieces[0] == b"":
                pieces = pieces[1:]
        return pieces


def _hf_tok(name: str, tk, src: str) -> Tok:
    """Wrap a `tokenizers.Tokenizer` (ByteLevel or Metaspace/byte-fallback)."""
    j = json.loads(tk.to_str())
    pre = j.get("pre_tokenizer") or {}
    pattern = json.dumps(pre, ensure_ascii=False)
    byte_level = "ByteLevel" in json.dumps(j.get("decoder"))
    bd = _byte_decoder()
    id2tok = tk.id_to_token

    def piece_bytes(ids: list[int]) -> list[bytes]:
        out = []
        for i in ids:
            t = id2tok(i)
            if byte_level:
                out.append(bytes(bd[c] for c in t))
            else:
                m = _BYTE_TOKEN.match(t)
                out.append(bytes([int(m.group(1), 16)]) if m else t.replace("▁", " ").encode("utf-8"))
        return out

    norm = (lambda s: unicodedata.normalize("NFC", s)) if (j.get("normalizer") or {}).get("type") == "NFC" else (lambda s: s)
    return Tok(
        name=name,
        vocab_size=tk.get_vocab_size(with_added_tokens=False),
        pattern=f"{src} :: {pattern}",
        encode=lambda s: tk.encode(s, add_special_tokens=False).ids,
        _piece_bytes=piece_bytes,
        normalize=norm,
    )


def load_hf_file(name: str, path: str | Path) -> Tok:
    from tokenizers import Tokenizer

    return _hf_tok(name, Tokenizer.from_file(str(path)), str(path))


@lru_cache(maxsize=None)
def load(name: str) -> Tok:
    if name not in BASELINES:
        p = Path(name)
        if p.exists():
            return load_hf_file(p.parent.name if p.name == "tokenizer.json" else p.stem, p)
        raise KeyError(f"unknown tokenizer {name!r}; known: {list(BASELINES)} or a tokenizer.json path")
    kind, src = BASELINES[name]
    if kind == "tiktoken":
        import tiktoken

        enc = tiktoken.get_encoding(src)
        return Tok(
            name=name,
            vocab_size=enc.n_vocab,
            pattern=f"tiktoken:{src} :: {enc._pat_str}",
            encode=lambda s: enc.encode(s, disallowed_special=()),
            _piece_bytes=lambda ids: [enc.decode_single_token_bytes(i) for i in ids],
        )
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    return _hf_tok(name, Tokenizer.from_file(hf_hub_download(src, "tokenizer.json")), src)
