# CLAUDE.md — Big-O Token Tax (The Big Hack 2.0, problem R3)

> **SCOPE CHANGE (2026-11-20 23:30): we do NOT train models anymore. Follow `docs/EVAL_INSTRUCTIONS.md` first.**

Read this first. Full detail lives in `docs/PROJECT_PLAN.md`; the task checklist is `docs/TASKS.md`.

## What we are building

A **fairer tokenizer for Kannada and Kanglish** (Kannada typed in English letters, often mixed with English), and proof that it helps.

- Hackathon: The Big Hack 2.0, Bengaluru. Track 03 "AI research", problem **R3 "The token tax"**.
- Challenge text: *Reduce the cost or quality gap for Indian languages in language models, and measure exactly what the improvement costs you.*
- Build time: **48 hours**. Compute: **free only** (Kaggle 2× T4, laptops). No rented GPUs.

## Track rules we must satisfy (R3 is a research track)

1. A hypothesis or method stated in one or two sentences.
2. An evaluation that compares our result with a baseline.
3. An analysis of where our method fails. A properly measured negative result counts.
4. No "just prompt ChatGPT/Gemini" submissions. Do not publish harmful outputs.

Judging: Innovation 25, Technical depth 25, Evidence 20, Impact 15, Presentation 15.

## Hypothesis

Most of Kannada's token tax in open models comes from pre-tokenization. A mark-aware, akshara-safe, suffix-aware, parity-aware BPE will cut the Kannada-to-English token ratio by at least half against standard BPE at the same vocabulary size, with no loss in bits per character on held-out Kannada text.

## The tokenizer: four layers (each one is an ablation step)

1. **Mark-aware split rule.** Replace the pre-tokenizer's `\p{L}+` with `[\p{L}\p{M}]+` so Kannada vowel signs and virama (Unicode category M) stay inside words.
   - Verified example: `ಮನೆಯಲ್ಲಿ ಇದೆ` → letters-only rule gives 8 pieces `ಮನ|ೆ|ಯಲ|್|ಲ|ಿ|ಇದ|ೆ`; mark-aware gives 2 `ಮನೆಯಲ್ಲಿ|ಇದೆ`.
2. **Akshara-safe pieces.** Token boundaries only between aksharas (grapheme clusters), never inside one (e.g. `ಕ್ರಿ` = ಕ + ್ + ರ + ಿ is one unit).
3. **Suffix-aware merges.** No merges across stem + case-ending boundaries (ಮನೆ + ಯಲ್ಲಿ / ಗೆ / ಯಿಂದ). Source: suffix list and/or Morfessor.
4. **Parity-aware merges.** Each merge step picks the merge that most helps the currently worst-compressed language (max–min), per Parity-aware BPE (arXiv 2508.04796, code: swiss-ai/parity-aware-bpe).

Training text: Kannada script + English + Kanglish/romanized Kannada.

## Key design decision: EXTEND, do not replace

For the real model we keep every original token, merge and token ID, and **append** new Kannada vocab/merges. Consequences:
- English, JSON, code and tool-call tokenization must be **byte-identical** to the original tokenizer. This is a test (see below), not an assumption.
- Edit `tokenizer.json` directly (pre_tokenizer pattern, append to `model.vocab` and `model.merges`). Do **not** use `tokenizer.add_tokens()` for Kannada subwords: it creates added-token string matches, not BPE merges.

## Base model

- **Qwen3-1.7B** (instruct, tool calling, Apache-2.0). **Checked 20 Nov 2026:** its `tokenizer.json` pre_tokenizer is `Sequence[Split(regex), ByteLevel(use_regex=false)]` with the letters-only pattern `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`. Base vocab 151,643 entries + added special tokens, 151,387 merges. Sanity numbers with the original tokenizer: `ಮನೆಯಲ್ಲಿ ಇದೆ.` = 17 tokens vs `It is in the house.` = 6; `ಕನ್ನಡ ಒಂದು ಸುಂದರ ಭಾಷೆ` = 29 tokens.
- The fix for layer 1 is to replace `\p{L}+` with `[\p{L}\p{M}]+` in **both** places it appears in that pattern. Verified: HF `tokenizers` `Split(Regex(...))` accepts `\p{M}`.
- Backup: Llama-3.2-1B-Instruct (gated licence).
- Not as base (SentencePiece, no regex split): Gemma-3, Sarvam-1. Use them as baselines only.

## Hardware constraints (Kaggle T4) — always respect

- Two **separate** 16 GB T4s, not one 32 GB GPU. ~30 GPU-hours/week per account, 12-hour sessions; quota counts wall-clock time, so shut idle sessions down.
- **fp16 only. No bf16. No FlashAttention-2** (needs Ampere). Use PyTorch SDPA attention and AMP loss scaling.
- 1.7B: train only new embedding rows + new output rows + LoRA. Do not train the whole ~150K × 2048 embedding matrix with Adam (~5 GB). Use PEFT trainable-token indices if available, else a separate small embedding table for new tokens.
- Checkpoint every 20–30 min. Pre-tokenize data on a laptop and upload as a Kaggle dataset.
- Small models: DDP across both T4s. Demo: original model on GPU 0, ours on GPU 1.
- Never convert our model to GGUF/llama.cpp/Ollama: they may not recognise our custom pre-tokenizer. Serve with plain `transformers`.

## Evaluation (summary — details in docs/PROJECT_PLAN.md)

- **Level 1 (no model):** same sentences in English / Kannada / romanized Kannada / Kanglish. Metrics: tokens per sentence, parity (÷ English), fertility, chars per token, akshara-split rate, byte-fallback rate.
- **Level 2 (tiny models, 20–50M params, from scratch):** identical except tokenizer, matched on training **characters**, 2 seeds. Metric: bits per character (BPC).
- **Level 3 (Qwen3-1.7B):** original vs standard Kannada extension vs ours, same training stages. Belebele (kan_Knda + English), FLORES chrF++ (en↔kn), tokens/s, ₹ per 1,000 answers, context fit. Plus the three-route test: native Kannada vs translate-to-English vs Kanglish.
- **Agent:** 150 labelled queries (50 Kannada, 50 Kanglish, 50 English): tool choice, argument accuracy, task success, tokens per task, time per task, context fit. Also run with all 52 harness tools vs our 6–8.

Baselines for every table: cl100k_base, o200k_base, Llama-3, Gemma-3, Sarvam, indictok-256k, plain BPE at the same vocab size.

## Data hygiene (do not break)

- FLORES-200 **devtest** is test-only. Belebele passages come from FLORES, so **exclude all FLORES text from training data**.
- The hand-written Kanglish test set is written before any training and never trained on.
- Always report vocabulary size next to every result. Compare tokenizers at equal vocabulary size.

## Mandatory tokenizer tests (write these early, keep them green)

1. Old-vs-new identity: 1,000+ English, JSON, code and tool-call strings tokenize to identical IDs.
2. Round trip: `decode(encode(x)) == x` for Kannada, Kanglish, English, mixed text, emojis, numbers, ZWJ/ZWNJ.
3. Akshara integrity: no token boundary falls inside a grapheme cluster (Python `regex` `\X`), checked on a Kannada sample.

## Proposed repo layout

```
tokenizer/      # layer 1-4 training code, merge/extend tokenizer.json
data/           # download + clean scripts (no raw data committed)
eval/
  level1/       # token counts, parity, fertility
  level2/       # tiny-model training + BPC
  level3/       # Belebele, chrF++, speed, cost
  agent/        # agent test set + scorer
model/          # extend Qwen3, FOCUS/mean init, stage A/B/C training
server/         # OpenAI-compatible /v1/chat/completions, constrained tool calls
demo/           # token visualizer + side-by-side dashboard
notebooks/      # Kaggle notebooks (thin wrappers around the code above)
tests/          # the mandatory tokenizer tests
results/        # CSV/JSON outputs that feed slides
docs/
```

## Conventions

- Python 3.10+. Libraries: `tokenizers`, `sentencepiece`, `transformers`, `peft`, `accelerate`, `datasets`, `regex`, `sacrebleu`, `lm-eval`, `deepfocus`, `outlines` (or `lm-format-enforcer`/`xgrammar`), `fastapi`, `tiktoken`.
- Every experiment writes its numbers to `results/` as CSV/JSON with: tokenizer name, vocab size, seed, data split, date. Slides are filled from these files only — never from memory.
- Scripts take CLI args; no hard-coded Kaggle paths outside `notebooks/`.
- Keep Kannada text UTF-8 and NFC-normalized consistently across train and test. Do not strip ZWJ/ZWNJ.

## Reuse, don't rebuild

We assemble the tokenizer from existing open-source parts. Write glue code, not new algorithms. Check each licence before vendoring; prefer `pip install` over copying code.

| Need | Reuse | Licence | How we use it |
| --- | --- | --- | --- |
| Tokenizer engine, file format | Hugging Face `tokenizers` (`Split`, `ByteLevel`, `BpeTrainer`, `Tokenizer.from_file`) | Apache-2.0 | Train merges, edit and save `tokenizer.json` |
| Base vocabulary | Qwen3-1.7B `tokenizer.json` | Apache-2.0 | Kept whole; we only append |
| Layer 1 split rule | `tokenizers` regex with `\p{M}` | — | One-line pattern change (verified works) |
| Layer 2 akshara segmentation | `akshara-tokenizer` (IYRA; clone + `pip install .`) — `segment_aksharas()` handles Kannada conjuncts via virama | Apache-2.0 | Akshara-split-rate metric; filter new tokens to akshara-aligned ones; training-time boundaries |
| Layer 3 suffix boundaries | `morfessor` (pip) unsupervised segmentation; Kannada suffix rules from the `Sahana-M/shabdkosh` stemmer as a seed list (check licence) | Morfessor BSD | Stem/suffix boundaries applied to training text only |
| Layer 4 parity-aware merges | `swiss-ai/parity-aware-bpe` (`pip install git+https://github.com/swiss-ai/parity-aware-bpe.git`) | MIT | Learns merges from per-language corpora + parallel dev set; outputs `merges.txt`/`vocab.json` that we convert into `tokenizer.json` |
| Reference Indic tokenizer + benchmark | `vivekvar-dl/indictok` (code + 137 validated script rules + `benchmark_multilingual.py`, covers Kannada) | Apache-2.0 | Strong baseline; reuse its benchmark script and script rules instead of writing our own |
| Grapheme-BPE reference | `ghilli` (GPE-BPE on HF tokenizers, Tamil config) | Apache-2.0 | Pattern to copy if we try a grapheme-alphabet variant for Kannada |
| Parity tables | `AleksandarPetrov/tokenization-fairness` | see repo | Parity on FLORES-200 |
| Intrinsic scores | `tokenization-scorer` (Rényi efficiency), `catherinearnett/byte-premium-tool` | see repos | Extra Level 1 columns |
| Normalization | `indic-nlp-library` | MIT | Optional normalization ablation |
| Romanization | AI4Bharat IndicXlit (`ai4bharat-transliteration`) | MIT | Build romanized-Kannada text |
| Embedding init | `deepfocus` (FOCUS) | see repo | New-token embeddings |
| Tiny models | `karpathy/nanoGPT` | MIT | Level 2 trainer |
| Eval | `lm-evaluation-harness` (Belebele), `sacrebleu` (chrF++) | MIT / Apache-2.0 | Level 3 |
| Constrained tool calls | `outlines` or `lm-format-enforcer` | Apache-2.0 / MIT | Server |

Important gotchas found while checking these:
- **Do not use regex `\X` as "akshara".** Tested: HF `tokenizers` `\X` (Unicode grapheme clusters) splits the conjunct `ಕ್ರಿ` into `ಕ್` + `ರಿ`. Use `akshara-tokenizer`'s virama-aware rules for akshara boundaries.
- `parity-aware-bpe` documents Whitespace + ByteLevel pre-tokenization only. Feed it text that our pre-tokenizer has already split (one pre-token per line or space-joined), and run the identical pre-tokenizer at inference.
- `indictok` and `akshara-tokenizer` are character-level / SentencePiece, while Qwen3 is byte-level BPE. Their merges cannot be appended to Qwen3 directly; use them as baselines, rule sources and metrics. (Optional experiment only: convert char-level Kannada merges to byte-level by first adding byte merges that build each Kannada character.)
- New token IDs must start after the highest existing ID including added special tokens; then resize the model's embeddings.

## Three phases

Work is gated: finish a phase's exit check before moving on. Tracks run in parallel inside each phase. Hours are for a 48-hour build.

### Phase 1 — Measure and set up (hours 0–12)
Goal: the "before" picture, all data ready, every reused tool installed and working.
- Scaffold repo, `requirements.txt`, `tests/`; install the reuse table's packages.
- Download FLORES-200 devtest, IN22-Gen, Belebele, Sangraha Kannada subset, English sample; strip FLORES text from training data.
- Team writes the Kanglish test set (150–200 sentences) now, before any training.
- Level 1 script for baselines (cl100k, o200k, Llama-3, Gemma-3, Sarvam, indictok-256k, Qwen3): tokens/sentence, parity, fertility, chars/token, akshara-split rate (via `akshara-tokenizer`), byte-fallback rate. Reuse `indictok`'s benchmark and Petrov's parity code where they fit.
- Romanize Kannada with IndicXlit; measure the Kanglish tax.
- Quick win: two BPE tokenizers with HF `tokenizers`, same data and vocab size, original Qwen3 pattern vs `[\p{L}\p{M}]+`.
- Write the three mandatory tokenizer tests (identity, round trip, akshara integrity).
- Kaggle: verify accounts, test a GPU notebook, measure 1.7B tokens/s.
**Exit check:** baseline table + quick-win numbers in `results/`; tests run; Kanglish set frozen.

### Phase 2 — Build the tokenizer and train (hours 12–36)
Goal: our tokenizer assembled from reused parts, extended into Qwen3, models trained.
- Layer 2+3 training text: split Kannada words at akshara boundaries (`akshara-tokenizer`) and stem/suffix boundaries (`morfessor` + seed list) so merges never cross them.
- Layer 4: learn new Kannada merges with `parity-aware-bpe` on that pre-split text (byte-level, mark-aware pattern); fall back to HF `BpeTrainer` if integration fails, and report which was used.
- Filter new tokens to akshara-aligned strings; append vocab + merges to Qwen3's `tokenizer.json`; swap in the mark-aware pattern; tests must stay green.
- Ablation: add layers one at a time; vocab sweep 8K–128K new merges.
- Level 2: nanoGPT tiny models (plain BPE, layer-1-only, full, byte-level), 2 seeds, matched characters — run overnight on Kaggle.
- Level 3: resize Qwen3 embeddings, FOCUS vs mean init, stages A (new rows) and B (LoRA continued pretraining); same for the standard-extension baseline.
- Romanized-Kannada tokenizer (HF `tokenizers`, Akshar-32k recipe) for the Kanglish comparison.
**Exit check:** ablation table, BPC table, two trained 1.7B models saved to Kaggle/HF Hub.

### Phase 3 — Evaluate, demo and pitch (hours 36–48)
Goal: evidence, the live agent demo and the slides.
- Stage C instruction + tool-call tuning (both extended models).
- Level 3 evals: Belebele, FLORES chrF++, speed, ₹ per 1,000 answers, context fit, three-route test, blind human rating.
- Server (FastAPI + transformers + constrained tool calls), tunnel, laptop fallback.
- Harness composition (web_search, web_fetch, read, ask_user_question) + MCP tools (weather, BMTC, mandi, Kannada Wikipedia); agent test set + scorer; 52-tool vs 8-tool comparison.
- Dashboard: token visualizer + side-by-side agent run; record backup video.
- Hour 40: freeze experiments. Fill slides from `results/`; failure analysis with real examples; rehearse.
**Exit check:** every slide number traceable to a file in `results/`; demo runs on the backup path too.

## Start here (first hour)

1. Scaffold the layout above, `requirements.txt` (including the reuse table's packages), and `tests/`.
2. Write `eval/level1` token counter + parity for the baseline tokenizers on FLORES-200 (kan_Knda, eng_Latn).
3. The quick win from Phase 1.
