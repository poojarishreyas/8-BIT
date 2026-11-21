# Evaluation matrix

Rows = metrics, columns = models. `pending` = model not evaluated yet.

| metric | qwen3_base | bigo_trained | std_ext_trained | sarvam1 |
|---|---|---|---|---|
| tokenizer | qwen3 | bigo_final | std_16k | sarvam1 |
| status | pending | pending | pending | pending |
| L1_kan_Knda_parity | 6.91 | 1.08 | 2.45 | 1.19 |
| L1_kan_Knda_tokens_per_word | 11.85 | 1.85 | 4.20 | 2.33 |
| L1_new_vocab | 0 | 31991 | 15968 |  |
| L1_kan_Latn_xlit_parity | 2.17 | 2.17 | 2.17 | 2.36 |
| L1_kan_Latn_xlit_tokens_per_word | 3.72 | 3.72 | 3.72 | 4.60 |
| L1_kanglish_parity | 1.13 | 1.13 | 1.13 | 1.25 |
| ctx_4K_kannada_sentences | 17 | 125 | 54 | pending |
| ctx_8K_kannada_sentences | 36 | 258 | 114 | pending |
| ctx_32K_kannada_sentences | 168 | 1067 | 476 | pending |

Sources: results/models/<tag>/ (python -m eval.run_all), results/level1_matrix.csv, results/context_fit.csv.
Rebuild: python -m eval.build_matrix
