# Slide numbers (every number traceable to a file)

| # | claim | number | source |
|---|---|---|---|
| 1 | Kannada token tax, Qwen3 (tokens ÷ English, FLORES devtest) | 6.91× | results/level1_matrix.csv (qwen3, flores_devtest, kan_Knda, parity) |
| 2 | … with our tokenizer (bigo_final, 31991 new tokens) | 1.08× | results/level1_matrix.csv (bigo_final) |
| 3 | … with a standard BPE extension at 16K (fair baseline) | 2.45× | results/level1_matrix.csv (std_16k) |
| 4 | Kannada tokens per word: Qwen3 → ours | 11.85 → 1.85 | results/level1_matrix.csv (fertility) |
| 5 | Kannada sentences in a 4K window: Qwen3 → ours | 17 → 125 | results/context_fit.csv |
| 6 | English/JSON/code/tool-call token IDs changed | 0 | tests/test_tokenizer.py (identity tests, all pass) |
| 7 | Belebele Kannada accuracy: Qwen3 base vs bigo-trained | pending vs pending | results/models/<tag>/level3_belebele.csv |
| 8 | ₹ per 1,000 Kannada answers: Qwen3 base vs bigo-trained | pending vs pending | results/models/<tag>/speed_cost.json |

Regenerate: python -m eval.level1.matrix && python -m eval.level3.context_fit && python -m eval.build_matrix
