"""Export an extended tokenizer.json as a transformers-loadable folder (chat template, special tokens from Qwen3).

Writes <out>/{tokenizer.json, tokenizer_config.json, special_tokens_map.json} and verifies that
AutoTokenizer produces the same ids as the raw `tokenizers` file (transformers can rebuild a model's
pre-tokenizer from its class defaults, which would silently drop our split rule).

Usage: python -m tokenizer.export_hf --tokenizer artifacts/ext/l123z_16k/tokenizer.json --out tokenizers_hf/l123z_16k
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

CHECK = ["ಮನೆಯಲ್ಲಿ ಇದೆ.", "ಕಾರ್‌ನಲ್ಲಿ ಪ್ರಶ್ನೆ", "naanu nale office ge hogtini", "It is in the house.",
         '<tool_call>\n{"name": "get_weather", "arguments": {"city": "ಬೆಂಗಳೂರು"}}\n</tool_call>', "😀 123 ೧೨೩"]


def export(tokenizer_json: str, out: str, base: str = "Qwen/Qwen3-1.7B") -> Path:
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    o = Path(out)
    o.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(tokenizer_json, o / "tokenizer.json")
    cfg = json.loads(Path(hf_hub_download(base, "tokenizer_config.json")).read_text(encoding="utf-8"))
    # Use the generic fast class so transformers takes pre-tokenizer + vocab from tokenizer.json as-is.
    cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
    cfg.pop("auto_map", None)
    (o / "tokenizer_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    for f in ("special_tokens_map.json", "generation_config.json"):
        try:
            shutil.copyfile(hf_hub_download(base, f), o / f)
        except Exception:
            pass
    raw = Tokenizer.from_file(str(o / "tokenizer.json"))
    hf = AutoTokenizer.from_pretrained(str(o))
    bad = [s for s in CHECK if hf(s, add_special_tokens=False)["input_ids"] != raw.encode(s, add_special_tokens=False).ids]
    if bad:
        raise SystemExit(f"AutoTokenizer differs from tokenizer.json on {bad}")
    assert len(hf) == raw.get_vocab_size(), (len(hf), raw.get_vocab_size())
    msgs = [{"role": "user", "content": "ಬೆಂಗಳೂರು ಹವಾಮಾನ?"}]
    hf.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    print(f"exported {o} (len {len(hf)}, eos {hf.eos_token}, pad {hf.pad_token}); AutoTokenizer == tokenizer.json")
    return o


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B")
    a = ap.parse_args()
    export(a.tokenizer, a.out, a.base)


if __name__ == "__main__":
    main()
