"""Load a model folder (or hub id) for evaluation with a *verified* tokenizer.

transformers can rebuild a tokenizer's pre-tokenizer from its class defaults (stock Qwen2Tokenizer does,
in transformers 5), which would silently undo our split rule. If the folder has a tokenizer.json, we
check AutoTokenizer against it on probe strings and stop on any mismatch.
"""
from __future__ import annotations

from pathlib import Path

import torch

def dtype_kw(dtype) -> dict:
    """transformers >= 4.56 takes `dtype=`; older versions silently ignore it and need `torch_dtype=`."""
    import transformers
    from packaging.version import Version

    return {"dtype": dtype} if Version(transformers.__version__) >= Version("4.56") else {"torch_dtype": dtype}


PROBES = ["ಮನೆಯಲ್ಲಿ ಇದೆ.", "ಕಾರ್‌ನಲ್ಲಿ ಪ್ರಶ್ನೆ", "naanu nale office ge hogtini", "It is in the house.",
          '<tool_call>\n{"name": "get_weather", "arguments": {"city": "ಬೆಂಗಳೂರು"}}\n</tool_call>']


def load_tokenizer(path: str, tokenizer: str | None = None):
    from transformers import AutoTokenizer

    src = tokenizer or path
    tok = AutoTokenizer.from_pretrained(src)
    p = Path(src) / "tokenizer.json"
    if p.exists():
        from tokenizers import Tokenizer

        raw = Tokenizer.from_file(str(p))
        bad = [s for s in PROBES if tok(s, add_special_tokens=False)["input_ids"] != raw.encode(s, add_special_tokens=False).ids]
        if bad:
            raise SystemExit(f"AutoTokenizer({src}) disagrees with its tokenizer.json on {bad}. "
                             "Re-export with `python -m tokenizer.export_hf`.")
    return tok


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"  # torch.cuda also covers ROCm (AMD) builds


def pick_dtype(dev: torch.device):
    """bf16 where supported (Ampere+/ROCm MI/RDNA3), fp16 on T4 (no bf16), fp32 on CPU."""
    if dev.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load(path: str, device: str | None = None, tokenizer: str | None = None):
    from transformers import AutoModelForCausalLM

    tok = load_tokenizer(path, tokenizer)
    dev = torch.device(device or default_device())
    dtype = pick_dtype(dev)
    model = AutoModelForCausalLM.from_pretrained(path, **dtype_kw(dtype), attn_implementation="sdpa").to(dev).eval()
    if model.get_input_embeddings().weight.shape[0] < len(tok):
        raise SystemExit(f"model has {model.get_input_embeddings().weight.shape[0]} rows < tokenizer {len(tok)}")
    return model, tok


def model_label(path: str) -> str:
    p = Path(path)
    return p.parent.name + "/" + p.name if p.name == "hf" else (p.name or str(path))
