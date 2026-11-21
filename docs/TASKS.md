# Task checklist (by phase)

> **Scope change 2026-11-20 23:30:** no model training here. Every Qwen3 training item below (Stage A/B/C, FOCUS, LoRA, SFT, Level 3 training) is **moved to teammate** (AMD GPU). Current scope: `docs/EVAL_INSTRUCTIONS.md`.

Finish a phase's exit check before starting the next. Put every number into `results/`.

## Phase 1 — Measure and set up (hours 0–12)

Setup
- [ ] Phone-verify all four Kaggle accounts (GPU + internet)
- [x] Repo scaffold, `requirements.txt`, `tests/`
- [ ] Install reused tools: `tokenizers`, `tiktoken`, `akshara-tokenizer` (clone + `pip install .`), `morfessor`, `parity-aware-bpe` (pip from GitHub), `indic-nlp-library`, `ai4bharat-transliteration`, `deepfocus`, `tokenization-scorer`, `lm-eval`, `sacrebleu`
- [x] Clone `vivekvar-dl/indictok` and `AleksandarPetrov/tokenization-fairness` for their benchmark/parity scripts

Data
- [x] FLORES-200 devtest (eng_Latn, kan_Knda), IN22-Gen, Belebele, Sangraha Kannada subset, English sample
- [x] Remove any FLORES text from training data; dedupe
- [ ] Team writes the Kanglish test set (150–200 sentences) — freeze it
- [ ] Romanize Kannada sentences with IndicXlit

Measure
- [x] Level 1 script: tokens/sentence, parity, fertility, chars/token, akshara-split rate, byte-fallback rate
- [ ] Baselines: cl100k, o200k, Llama-3, Gemma-3, Sarvam, indictok-256k, Qwen3 (record each one's pre-tokenizer pattern)
- [ ] Kanglish tax table (Kannada vs romanized vs Kanglish vs English)
- [x] Quick win: HF BPE with original Qwen3 pattern vs `[\p{L}\p{M}]+`, same data, same vocab

Tests and compute
- [x] Tests: old-vs-new identity, round trip, akshara integrity (using `akshara-tokenizer`, not `\X`)
- [x] Kaggle smoke test: load Qwen3-1.7B in fp16, measure training tokens/s (756 tok/s per T4, see results/kaggle_smoke_Tesla_T4_2026-11-20.json)

**Exit check:** baseline + quick-win numbers in `results/`; tests run; Kanglish set frozen.

## Phase 2 — Build the tokenizer and train (hours 12–36)

Tokenizer (assembled from reused parts)
- [x] Akshara units (a0 chars, a1 aksharas) instead of pre-splitting — see docs/PHASE2_NOTES.md (`akshara-tokenizer`)
- [~] Stem/suffix boundaries: seed list done; Morfessor + hand-check 100 words TODO
- [x] Learn new merges with `parity-aware-bpe` (used for every build; see build_report merge_learner)
- [x] Filter new tokens: must contain Kannada bytes (identity by construction); akshara-aligned by construction
- [x] Append vocab + merges to Qwen3 `tokenizer.json` after the highest existing ID; mark-aware pattern in both places
- [x] Tests green (identity on English/JSON/code/tool calls)
- [x] Ablation: layers 1→3 (L4 implemented, final build not completed — killed for low memory)
- [x] Vocab sweep: 8K, 16K, 32K, 64K (128K dropped)
- [ ] Romanized-Kannada tokenizer (HF `tokenizers`) for the Kanglish comparison

Tiny models (Level 2)
- [ ] nanoGPT trainer, 20–50M params, fp16, DDP on 2× T4, budget matched by characters
- [ ] Runs: plain BPE ×2, ours ×2 (overnight)
- [ ] Runs: layer-1-only, byte-level
- [ ] BPC on held-out Kannada, English, romanized, Kanglish

Real model (Level 3, stages A–B)
- [moved to teammate] Resize Qwen3-1.7B embeddings; mean init and FOCUS init
- [moved to teammate] Train only new rows (PEFT trainable tokens or separate table)
- [moved to teammate] Stage A (new rows), Stage B (LoRA continued pretraining)
- [moved to teammate] Same stages for the standard-extension baseline
- [moved to teammate] Save both models (Kaggle output / HF Hub, private)

**Exit check:** ablation table, BPC table, two trained 1.7B models saved.

## Phase 3 — Evaluate, demo and pitch (hours 36–48)

Model evaluation
- [moved to teammate] Stage C instruction + tool-call tuning (both extended models)
- [ ] Belebele kan_Knda + eng_Latn + romanized; FLORES chrF++ both directions
- [ ] Speed, ₹ per 1,000 answers, context fit
- [ ] Three-route test: native Kannada vs translate-to-English vs Kanglish
- [ ] Blind human rating, 20 Kannada + 20 Kanglish prompts

Agent demo
- [ ] Model server: OpenAI-style endpoint, constrained tool calls, token + time fields
- [ ] Tunnel + laptop CPU fallback
- [ ] Harness composition: web_search, web_fetch, read, ask_user_question (+ todo_write)
- [ ] MCP tools: weather, BMTC bus, mandi prices, Kannada Wikipedia
- [ ] Agent test set (150 queries) + scorer; 52-tool vs 8-tool comparison
- [ ] Dashboard: token visualizer + side-by-side agent run
- [ ] Record backup demo video

Pitch
- [ ] Hour 40: freeze experiments
- [ ] Fill slide placeholders from `results/`
- [ ] Failure analysis slide with real examples
- [ ] Rehearse pitch and demo

**Exit check:** every slide number traceable to `results/`; backup demo path works.

## Scope change (EVAL_INSTRUCTIONS.md) — status
- [x] Part A: final build exported (`artifacts/final/bigo_final` = l123z_32k); tests pass (58)
- [ ] Part A: L4 final build (l1234) — rerun alone when memory allows; romanized variant (needs romanized_out.zip)
- [x] Part B: Level 1 matrix, Kanglish tax, 5 charts, failure analysis (counts)
- [ ] Part B: romanized columns (needs romanized_out.zip); team Kanglish set (150–200 lines)
- [x] Part C: eval/run_all.py (Belebele, chrF++, speed/₹, three-route, agent), context fit, 50 agent queries, human rating sheet, eval/build_matrix.py
- [ ] Part C: run Qwen3 baseline on Kaggle (notebooks/kaggle_eval_baseline.ipynb); teammate's model later
- [x] Part D: dist/bigo_final_tokenizer.zip
