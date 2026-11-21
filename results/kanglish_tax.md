# Kanglish tax (parity = tokens ÷ English tokens for the same sentence; flores_devtest + hand-written Kanglish)

| tokenizer | new vocab | Kannada script | romanized (IndicXlit) | hand-written Kanglish |
|---|---|---|---|---|
| GPT-4 (cl100k) | — | 8.84× | 2.19× | 1.13× |
| GPT-4o (o200k) | — | 1.95× | 1.97× | 1.03× |
| Llama-3 | — | 8.84× | 2.18× | 1.12× |
| Gemma-3 | — | 1.90× | 2.00× | 1.03× |
| Sarvam-1 | — | 1.19× | 2.36× | 1.25× |
| indictok-256k | — | 0.99× | 2.05× | 1.15× |
| Qwen3 | 0.0 | 6.91× | 2.17× | 1.13× |
| std_16k | 15968.0 | 2.45× | 2.17× | 1.13× |
| Ours (bigo_final) | 31991.0 | 1.08× | 2.17× | 1.13× |

Hand-written Kanglish set: 181 sentences (`data/processed/test/kanglish_handwritten.txt`).
Source: `results/level1_matrix.csv` (python -m eval.level1.matrix && python -m eval.level1.report).
