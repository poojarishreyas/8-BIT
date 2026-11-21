# Phase 2 notes: how the layers are actually built

Facts below were measured in this repo; numbers live in `results/` and `artifacts/ext/*/build_report.json`.

## Facts about Qwen3's tokenizer that shape the design
- Only **25** Qwen3 tokens contain Kannada, all single characters. Most Kannada characters are byte pieces.
- In 150K Kannada word types, the only old token that crosses a Kannada character boundary is
  `' \xe0\xb2'` (a space glued to the first two bytes of the next Kannada character). So with a leading
  space, `ಮ` is `[Ġà², ®]`.
- Kannada UTF-8 always starts with `E0 B2`/`E0 B3`, and those byte pairs appear in no other script.
  Every new merge must contain them, which makes English/JSON/code identity hold by construction.
  The identity test still checks it.
- New IDs start at **151,669**, after the added special tokens.
  Gotcha: HF `tokenizers` re-numbers added tokens after `model.vocab` when there is an ID gap. The fix is
  to put the 26 special tokens into `model.vocab` at their original IDs.
- The pattern has **one** `\p{L}+`, not two. One replacement is enough.

## Layer 2 (akshara-safe): units, not pre-splitting
Splitting training text at every akshara boundary would forbid all multi-akshara tokens. We use units instead:
- **a0**: deterministic merges that turn every Kannada character (and `' '+char`) into one token,
  built from its old pieces.
- **a1**: BPE (via parity-aware-bpe, one language) on *aksharas as isolated words*, run until every
  akshara with frequency >= 5 is a single token. These merges cannot fire across an akshara boundary:
  a pair with a consonant on its right only occurs inside an akshara after a virama, and a mark never
  starts an akshara.
- **b**: BPE over whole-akshara units only. Rare aksharas (not single tokens after a1) act as barriers in training.
- Known leak: a rare akshara's partial piece can equal a common akshara string, so stage-b merges can then
  split it at inference. The akshara-integrity test measures this; it is part of the failure analysis.

## Layer 3 (suffix-aware)
A hand-written seed list in `tokenizer/suffix.py`, applied as a hard split in training words
(stem >= 2 aksharas, longest match). Sandhi makes boundaries approximate (`ಊರು+ಇನಲ್ಲಿ` -> `ಊರಿ|ನಲ್ಲಿ`).
At inference, merges can still form across a stem|suffix boundary when the joined string was learned elsewhere.
Morfessor is the planned data-driven alternative.

## Layer 4 (parity-aware): where it applies
- **From-scratch tokenizers** (Level 2 / quick-win style): eng + kan_Knda + kan_Latn, with
  parity-aware-bpe picking the worst-compressed language at each step. The parallel dev set is Sangraha
  `synthetic/kan_Knda` and `synthetic/kan_Latn` (parallel by `doc_id`, no FLORES).
- **Qwen3 extension**: only Kannada-script merges are allowed, because Latin merges would change English
  tokenization. So with one language, parity-aware selection reduces to standard BPE.
  Romanized-Kannada merges are possible only if each one is checked to never fire on English text.
  That is a measured trade-off still to try.

## Data finding (ZWNJ)
Sangraha `verified/kan` has ZWNJ stripped: **0.00** per 10K characters, against **33** in FLORES devtest
and **80** in IN22-Gen. Kannada Wikipedia keeps it (13 per 10K). Kannada training text for the extension
is therefore 15M characters of Sangraha plus 15M of Kannada Wikipedia (`data/processed/train/kan_Knda_mix.txt`).
The Phase 1 quick-win numbers used Sangraha only.

## ZWNJ/ZWJ in the split rule (layer 1 refinement, `mark_zw`)
With `[\p{L}\p{M}]+`, 91% of the remaining akshara splits on FLORES were aksharas ending in ZWNJ/ZWJ
(`ನ್‌`, `ರ್‌`). ZW joiners are category Cf, so the rule cut the word just before them. `mark_zw` lets ZWJ/ZWNJ
continue a word only right after a Kannada character (a lookbehind on U+0C80–U+0CFF), so emoji ZWJ sequences,
Persian ZWNJ and Devanagari stay byte-identical (`test_identity_zw_outside_kannada`).
Result at 16K: akshara splits 1.3% -> 0.1% (FLORES), 2.9% -> 0.3% (IN22-Gen), with no extra tokens.

## Ablation at 16K appended tokens (results/level1_ablation16k_*.csv)
| build | Kannada ÷ English (FLORES / IN22) | akshara split (FLORES / IN22) |
|---|---|---|
| Qwen3 | 6.91 / 6.83 | 100% / 100% |
| standard extension (original rule) | 2.45 / 2.41 | 100% / 100% |
| + L1 `mark_zw` | 1.12 / 1.19 | 10.8% / 12.9% |
| + L2 akshara units | 1.14 / 1.21 | 0.1% / 0.3% |
| + L3 seed suffixes | 1.19 / 1.25 | 0.1% / 0.3% |
- L1 alone cuts the ratio by 54% against the standard extension at equal vocab (hypothesis: >= 50%).
- L2 costs +2.5% Kannada tokens for near-total akshara integrity.
- L3 costs a further +4.5% tokens at Level 1. Any benefit has to show up in BPC (Level 2) or downstream;
  if it doesn't, that is a negative result to report.
- Remaining splits (`results/akshara_failures_l12z_16k.json`): rare foreign-name conjuncts (`ರ್ಬ್ಯು`, `ನ್ಫ್ರೇ`)
  and nonstandard ZWNJ after vowel signs (`ನ್ನು‌`).
