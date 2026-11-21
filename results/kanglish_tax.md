# Kanglish tax (parity = tokens ÷ English tokens for the same sentence; flores_devtest + hand-written Kanglish)

| tokenizer | new vocab | Kannada script | romanized (IndicXlit) | hand-written Kanglish |
|---|---|---|---|---|
| GPT-4 (cl100k) | — | 8.84× | pending | 1.11× |
| GPT-4o (o200k) | — | 1.95× | pending | 1.05× |
| Llama-3 | — | 8.84× | pending | 1.10× |
| Gemma-3 | — | 1.90× | pending | 1.01× |
| Sarvam-1 | — | 1.19× | pending | 1.25× |
| indictok-256k | — | 0.99× | pending | 1.10× |
| Qwen3 | 0.0 | 6.91× | pending | 1.11× |
| std_16k | 15968.0 | 2.45× | pending | 1.11× |

Hand-written Kanglish set: 10 sentences (`data/processed/test/kanglish_handwritten.txt`). **Template only (10 lines) — team set pending; numbers are provisional.**
Source: `results/level1_matrix.csv` (python -m eval.level1.matrix && python -m eval.level1.report).
