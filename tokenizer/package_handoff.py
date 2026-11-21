"""Part D: dist/bigo_final_tokenizer.zip for the teammate (tokenizer only, no training code).

Contents: bigo_final/ (tokenizer.json, tokenizer_config.json, special_tokens_map.json, chat_template.jinja),
README.md (<= 10 lines), results/level1_matrix.csv, results/kanglish_tax.md.

Usage: python -m tokenizer.package_handoff
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "artifacts/final/bigo_final"
BASE_LEN = 151669


def main() -> None:
    tj = FINAL / "tokenizer.json"
    sha = hashlib.sha256(tj.read_bytes()).hexdigest()
    from tokenizers import Tokenizer

    n_total = Tokenizer.from_file(str(tj)).get_vocab_size(True)
    n_new = n_total - BASE_LEN
    readme = "\n".join([
        "# bigo_final: Qwen3 tokenizer extended for Kannada / Kanglish",
        f"Base vocab = Qwen3 ({BASE_LEN:,} incl. special tokens) + {n_new:,} new Kannada tokens = {n_total:,} total.",
        f"New token IDs start at {BASE_LEN:,}; every original token, merge and ID is unchanged (append-only).",
        "English / JSON / code / tool-call text tokenizes to identical IDs as Qwen3 (tested on 1,400+ strings + all English test lines).",
        f"Resize the model's embeddings to {n_total:,} rows before training; new rows can start as the mean of their Qwen3 sub-token embeddings.",
        "Load with AutoTokenizer.from_pretrained('bigo_final') - tokenizer_config uses PreTrainedTokenizerFast on purpose:",
        "  the stock Qwen2Tokenizer class rebuilds Qwen's split rule and silently undoes ours (transformers 5).",
        f"tokenizer.json SHA-256: {sha}",
        "What it achieves: results/level1_matrix.csv, results/kanglish_tax.md (FLORES devtest Kannada parity vs English).",
    ])
    out = ROOT / "dist/bigo_final_tokenizer.zip"
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"):
            z.write(FINAL / f, f"bigo_final/{f}")
        z.writestr("README.md", readme + "\n")
        for f in ("results/level1_matrix.csv", "results/kanglish_tax.md"):
            if (ROOT / f).exists():
                z.write(ROOT / f, f)
    (ROOT / "results/bigo_final_tokenizer.json").write_text(json.dumps(dict(sha256=sha, total_vocab=n_total, new_tokens=n_new,
                                                                            first_new_id=BASE_LEN, source="l123z_32k"), indent=1))
    print(out, f"{out.stat().st_size / 2**20:.1f} MB", "sha256", sha[:16], "new", n_new)


if __name__ == "__main__":
    main()
