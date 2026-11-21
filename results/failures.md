# Failure analysis: where bigo_final is no better (or worse)

Words from FLORES-200 devtest + IN22-Gen Kannada (and the hand-written Kanglish file), each tokenized with a
leading space as in running text. *No better than Qwen3* = ours uses as many or more tokens than the original;
*worse than std* = more tokens than the standard BPE extension at 16K (the fair baseline). Counts, not anecdotes.

| category | words | Qwen3 tok/word | std ext tok/word | ours tok/word | ours ≥ Qwen3 | ours > std ext | ours = 1 token |
|---|---|---|---|---|---|---|---|
| all_kannada | 34,252 | 12.35 | 4.21 | 1.77 | 6 (0.0%) | 84 (0.2%) | 49.5% |
| rare_kannada (train freq<5) | 9,652 | 16.97 | 5.82 | 2.83 | 2 (0.0%) | 53 (0.5%) | 1.7% |
| loanword (ends in virama) | 2,331 | 12.51 | 4.58 | 2.25 | 0 (0.0%) | 5 (0.2%) | 30.1% |
| number | 475 | 5.06 | 3.83 | 3.83 | 256 (53.9%) | 0 (0.0%) | 0.0% |
| english_in_kannada_line | 114 | 3.18 | 2.51 | 2.33 | 101 (88.6%) | 0 (0.0%) | 50.9% |
| kanglish_word | 68 | 1.79 | 1.79 | 1.79 | 68 (100.0%) | 0 (0.0%) | 47.1% |

Mixed-script lines (Kannada + Latin letters): 83 of 2036; tokens Qwen3 16,231 → ours 2,873 (0.18×). English words inside them tokenize exactly as in Qwen3 by design (identity test).

## Worst examples per category (ours vs std ext vs Qwen3 tokens)

**all_kannada**: `ಉಭಯಚರಗಳ` 3/1/10, `ಏಕದಳ` 3/1/6, `ಒಳಪಡದ` 3/1/7, `ಕಡಲಕರೆಯಾಚೆಯ` 6/4/15, `ಸದಿಶಗಳ` 4/2/9, `ಮಹ್ಕಮಹ್‌` 4/3/11, `ಸರಳೀಕರಣವಾದ` 4/3/15, `ಶಿಬೀರಗಳ` 4/3/12

**rare_kannada (train freq<5)**: `ಸದಿಶಗಳ` 4/2/9, `ಉಭಯಚರಗಳ` 3/1/10, `ಕಡಲಕರೆಯಾಚೆಯ` 6/4/15, `ಕಡಲಕರೆಯ` 3/2/9, `ಮೆಸಪಟೇಮಿಯದ` 6/5/14, `ಶಪಥಪತ್ರದ` 4/3/10, `‌ಔಷಧೀಯ` 5/4/11, `ನಗೋಯ` 3/2/6

**loanword (ends in virama)**: `ಗಢ್‌` 3/2/7, `ಮಹ್ಕಮಹ್‌` 4/3/11, `ರತನ್‌` 3/2/7, `ಭಜನ್‌` 3/2/8

**number**: `150` 4/4/4, `17` 3/3/3, `80` 3/3/3, `2020` 5/5/5, `1885` 5/5/5, `1945` 5/5/5, `2002` 5/5/5, `2010` 5/5/5

**english_in_kannada_line**: `PC` 1/1/1, `DNAಯ` 2/2/2, `FIFA` 1/1/1, `AI` 1/1/1, `USOC` 2/2/2, `r` 1/1/1, `802.11n` 8/8/8, `802.11b` 8/8/8

**kanglish_word**: `sakkath` 3/3/3, `traffic` 1/1/1, `aagutte` 4/4/4, `ee` 1/1/1, `call` 1/1/1, `helidru` 3/3/3, `madona` 2/2/2, `medicine` 1/1/1

Notes:
- Numbers: Qwen3's split rule gives every digit its own token (`\p{N}`); we keep that rule, so numbers
  are unchanged by design.
- English inside Kannada and romanized/Kanglish words: Latin text is untouched in bigo_final (English identity),
  so these get no benefit — the Kanglish tax remains (see kanglish_tax.md).
- Loanwords ending in virama and rare words are where the akshara inventory and suffix list run out.

Source: `python -m eval.level1.report` → results/failures.csv
