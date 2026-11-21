# Big-O Token Tax: a fairer Qwen3 tokenizer for Kannada and Kanglish

The Big Hack 2.0, Bengaluru, track 03 (AI research), problem R3 "The token tax".

**Hypothesis.** Most of Kannada's token tax in open models comes from pre-tokenization. A mark-aware,
akshara-safe, suffix-aware extension of Qwen3's BPE cuts the Kannada-to-English token ratio by more than half
compared with a standard BPE extension at the same vocabulary size, and leaves English, JSON, code and tool-call
tokens unchanged.

## Results (Level 1, tokenizer only; FLORES-200 devtest)

| tokenizer | new tokens | Kannada ÷ English tokens | Kannada tokens / word | Kannada sentences in 4K context |
|---|---|---|---|---|
| Qwen3 original | 0 | 6.91× | 11.85 | 17 |
| Qwen3 + standard BPE extension | 16K (saturates at ~24K) | 2.45× | 4.20 | 54 |
| **Qwen3 + ours (`bigo_final`)** | 32K | **1.08×** | **1.85** | **125** |
| GPT-4o (o200k) / Gemma-3 / Sarvam-1 (reference) | — | 1.95× / 1.90× / 1.19× | | |

- The split rule alone (layer 1) takes the standard extension from 2.45× to 1.12× at 16K new tokens.
- English, JSON, code and tool-call token IDs are identical to Qwen3 (`tests/test_tokenizer.py`, all pass).
- Akshara safety: no akshara that is in our inventory is ever split in running text. 282 of 155,165
  multi-codepoint test aksharas are split, all out-of-inventory (rare conjuncts, unusual ZWNJ use).
- Every number: `results/level1_matrix.csv`, `results/context_fit.csv`, `results/SLIDE_NUMBERS.md`.
  Charts: `results/charts/`. Kanglish tax: `results/kanglish_tax.md`. Failures with counts: `results/failures.md`.

## Method (four layers, each an ablation step)

1. **Mark-aware split rule.** Qwen3's `\p{L}+` becomes `[\p{L}\p{M}]+`, so vowel signs and virama stay inside
   words. ZWJ/ZWNJ may continue a word only right after a Kannada character (`mark_zw`), so emoji, Persian and
   Devanagari text are unchanged.
2. **Akshara units.** Merges first build each Kannada character, then each akshara (BPE on aksharas as isolated
   words), then merge only whole aksharas. Akshara rules come from `akshara-tokenizer`, not regex `\X`.
3. **Suffix-aware.** Hard stem|suffix splits in the training words, from a hand-written seed list of case endings.
4. **Parity-aware merges.** Wired in through `swiss-ai/parity-aware-bpe` (`--parity on`; English is the fixed
   reference; optional English-safe romanized merges). The final build was **not** completed (stopped for low
   memory), so `bigo_final` = layers 1–3 at 32K new tokens.

Extension is append-only: new IDs start at 151,669, and every original token, merge and ID is unchanged.
Merges are learned with `parity-aware-bpe` through `tokenizer/bpe_glue.py`, which maps units to private-use
characters.

## Reproduce

```
python -m data.download --only belebele flores in22 kan_Knda kan_Knda_wiki kan_Latn eng_Latn   # HF login for FLORES/IN22
python -m data.prepare_test && python -m data.clean && python -m data.parity_dev
python -m tokenizer.build_extension --pattern mark_zw --units akshara --suffix seed --new-tokens 32000 \
    --train data/processed/train/kan_Knda_mix.txt --out artifacts/ext/l123z_32k
python -m tokenizer.export_hf --tokenizer artifacts/ext/l123z_32k/tokenizer.json --out artifacts/final/bigo_final
pytest tests                                   # identity, round trip, akshara safety
python -m eval.level1.matrix && python -m eval.level1.report && python -m eval.level3.context_fit
python -m eval.run_all --model <hf-folder> --tag <name>      # model matrix (GPU); notebooks/kaggle_eval_baseline.ipynb
python -m eval.build_matrix                                  # results/eval_matrix.{csv,md}, SLIDE_NUMBERS.md
python -m tokenizer.package_handoff                          # dist/bigo_final_tokenizer.zip
```

## Limitations

- **No model training here.** A teammate trains Qwen3-1.7B on `bigo_final` on his own AMD GPU. The model matrix
  (`eval/run_all.py`) is ready and shows "pending" until that model and the Qwen3 baseline run arrive.
- **Level 2 (tiny-model bits per character) was skipped**, so there is no model-quality evidence for the
  layers yet. Layer 3 costs about 4.5% more tokens at Level 1, and any benefit would need BPC or downstream
  results to show.
- **Layer 4 is implemented but the final build was not completed** (stopped by hand at 11,700 of 26,770 merges
  after 61 min; about 20× slower than plain BPE because it re-scores the dev sets after every merge). With English
  held fixed and only Kannada merges allowed, parity-aware selection reduces to standard BPE plus a stopping point
  at parity, so we expect no gain (reasoned, not measured).
- **Kanglish is not improved by `bigo_final`.** Latin text is left untouched to keep English identical. The
  romanized variant (`--latn-train`) is implemented but not built. Romanized (IndicXlit) FLORES costs 2.17× and
  the 181-line Kanglish set 1.13× for every Qwen3 variant. The Kanglish set was written by Claude and checked
  by a native speaker on the team.
- **Unseen aksharas still split.** An unseen akshara must be split somewhere, and Qwen3's own old merge `Ġà²`
  fuses a word's leading space into a partial character.
- **Training-data ZWNJ.** Sangraha `verified/kan` has ZWNJ stripped, so Kannada Wikipedia was mixed in.

See `docs/EVAL_INSTRUCTIONS.md` (current scope), `docs/PHASE2_NOTES.md` (design facts) and `CLAUDE.md`.
