# Scope change: tokenizer + evaluation only (read this before anything else)

Date: 2026-11-20, ~23:30 IST. About 12 hours left.

**We no longer train any model.** Drop every Stage A / B / C task, the FOCUS init, LoRA and SFT work,
and anything in `docs/TASKS.md` that trains Qwen3. A teammate will train Qwen3-1.7B on OUR tokenizer
on his own AMD GPU and hand us the trained model later.

Our job, in order:

- **Part A:** finish the tokenizer.
- **Part B:** evaluate the tokenizer.
- **Part C:** build the full evaluation matrix. Run it now on baselines, and make it plug-and-play for the trained model when it arrives.
- **Part D:** zip the final tokenizer for the teammate (no training code).

Keep every rule in CLAUDE.md:
- no FLORES/Belebele text in any training data
- datasets stay Private
- free compute only
- never disable PyTorch `weights_only` safety; for IndicXlit we pin `torch<2.6` in a separate py3.10 venv

Put every number in `results/`, and write everything so one command regenerates it.

---

## Inputs arriving soon

`romanized_out.zip` from Kaggle (IndicXlit, beam 4). Unzip it as follows:
- `romanized/*_kan_Latn_xlit.txt` into `data/processed/test/`
  - the files are flores_dev, flores_devtest, in22gen, belebele_passages and parity_dev
- `results/romanize_xlit_*.json` into `results/`

Do NOT re-run IndicXlit locally; this is the frozen romanized set.

**Hand-written Kanglish test set (team task):**
- Write 150–200 sentences into `data/processed/test/kanglish_handwritten.txt`, one sentence per line, with an aligned English line in `kanglish_handwritten_eng.txt`.
- If it does not exist yet, create an empty template with 10 example lines and tell the user, then continue with everything else.

---

## Part A: Finish the tokenizer (target: 3–4 h)

Already done:
- the layer 1 mark-aware rule
- the layer 2 akshara units (`tokenizer/akshara.py`)
- seed suffix list (`tokenizer/suffix.py`)
- ablation builds in `artifacts/ext/` (l1, l12, l12z, l123, l123z at 8K–64K)
- `tokenizer/extend.py`, `tokenizer/export_hf.py`

Remaining:

1. **Layer 4, parity-aware merges.** Use `swiss-ai/parity-aware-bpe` on the pre-split Kannada + English training text. Use `data/processed/dev` or `parity_dev` as the parallel dev set for the parity criterion, never FLORES devtest. If it does not install or run within 45 minutes, fall back to the HF `BpeTrainer` with Kannada-upweighted sampling, and record which method was used in `results/builds/`.
2. **Final build: `l1234`.**
   - Build it at 16K and 32K new tokens, appended after Qwen3's 151,669.
   - Export it as a Hugging Face tokenizer folder: `artifacts/final/bigo_qwen3_ext_{16k,32k}/`.
   - Include `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json` and Qwen3's chat template.
   - Pick the better size on Level 1 parity vs vocab cost, and call it `bigo_final`.
3. **Romanized tokenizer for Kanglish (optional, only if time allows).**
   - Train a small BPE (8K) on `data/processed/train/kan_Latn.txt` and add it as an extension variant `l1234+latn`.
   - Skip it if it costs more than 1 hour; then report Kanglish with the base tokenizer only.
4. **Three mandatory tokenizer tests** (pytest, `tests/test_tokenizer.py`). All must pass before Part B.
   - **Unchanged on non-Kannada:** English, code, JSON and Qwen3 tool-call text tokenize to identical IDs under Qwen3 and `bigo_final`. Use at least 500 English lines, 50 JSON / tool-call samples and some code.
   - **Akshara-safe:** no token boundary falls inside an akshara on all test Kannada. Report the violation count, which should be 0.
   - **Lossless round trip:** decode(encode(x)) == x on all test sets, in all forms.

## Part B: Tokenizer evaluation (target: 2–3 h, CPU only)

**B1. Level 1 matrix.** Extend `eval/level1/run.py` to produce one long table: `results/level1_matrix.csv`.

Rows are tokenizers:
- the baselines: cl100k, o200k, Llama-3, Gemma-3, Sarvam-1, indictok-256k, Qwen3
- the ablation steps: Qwen3 + l1, l12, l12z, l123, l123z, l1234
- the vocab sweep: 8K, 16K, 32K, 64K
- standard BPE extension at the same size (the fair baseline)
- `bigo_final`

Columns, one set per form (eng_Latn, kan_Knda, kan_Latn_xlit, kanglish_handwritten) and per test set (flores_devtest, in22gen, belebele_passages):
- tokens/word (fertility)
- tokens/char
- parity vs English (same sentence)
- bytes premium
- Rényi efficiency (α=2.5, `tokenization-scorer`)
- % words split into 1 token
- akshara violations
- new vocab size

**B2. Kanglish tax table** (`results/kanglish_tax.csv` + `.md`). For each tokenizer, compare English, Kannada, romanized and hand-written Kanglish. This is a headline slide.

**B3. Charts** (`results/charts/*.png`, readable at slide size):
1. Parity bar chart: baselines vs `bigo_final`.
2. Ablation waterfall: tokens/word as layers 1→4 are added.
3. Vocab sweep curve: parity vs number of new tokens.
4. Kanglish tax grouped bars.
5. Token visualizer image of the same sentence under Qwen3 vs ours, with colored token boundaries.

**B4. Level 2 (optional; GPU needed; skip if behind).**
- `eval/level2/` already exists.
- Prepare the data for Qwen3, standard extension and `bigo_final`, and zip it as `dist/bigo-level2.zip`.
- Give the user one Kaggle command per run: 2× T4 with torchrun, about 40 minutes each.
- Report BPC on held-out Kannada, English, romanized and Kanglish in `results/level2_bpc.csv`.
- If skipped, say so in the README limitations.

**B5. Failure analysis** (`results/failures.md`). Real examples where `bigo_final` is no better or worse:
- rare words
- loanwords
- numbers
- mixed-script lines
- English inside Kannada
- dialect spellings in Kanglish

Do this with counts, not anecdotes. R3 requires it.

## Part C: Full evaluation matrix for models (build now, fill later)

Write it so the same command runs on any model folder:

```
python -m eval.run_all --model <hf-path-or-folder> --tag <name> --out results/models/<tag>/
```

Default device:
- `cuda` if present (this also covers ROCm on AMD)
- else `cpu`

Precision:
- fp16 on T4
- bf16 when supported

Tasks:

1. **Belebele (reading comprehension)** in kan_Knda, eng_Latn and kan_Latn (romanized passages from the zip).
   - Use log-likelihood scoring of the 4 options, no generation. Extend the existing `eval/level3/belebele.py`.
   - Use all 900 questions; add a `--limit` flag.
2. **FLORES chrF++** in both directions (kan→eng and eng→kan) on devtest, plus Kanglish→eng.
   - Use greedy decoding and batched generation. Extend `eval/level3/flores_chrf.py`.
   - Add `--limit` (default 200 for quick runs, full run 1012).
3. **Cost and speed.** Measure the following and convert to "₹ per 1,000 answers" with a stated price assumption:
   - prompt tokens, output tokens
   - tokens/s for prefill and decode
   - latency per answer
4. **Context fit.** Count how many Kannada FLORES sentences fit in a 4K / 8K / 32K window, Qwen3 vs ours. This one is tokenizer-only and can run now.
5. **Three-route test** on the same 200 questions:
   1. native Kannada
   2. translate to English → answer → translate back
   3. Kanglish

   Report accuracy, total tokens and latency for each route.
6. **Agent eval** (`eval/agent/`, currently empty).
   - 150 queries in `data/processed/test/agent_queries.jsonl`, each with: query, form (kn / kanglish / en), expected tool and expected key arguments.
   - Run against the harness tool list in Qwen3 tool-call format.
   - Score:
     - tool-choice accuracy
     - argument exact-match / F1
     - valid-JSON rate
     - tokens per query
   - Write the queries now; 50 is acceptable if short on time.
7. **Human rating sheet.** Create `results/human_rating_template.csv` with 20 Kannada and 20 Kanglish prompts, blind A/B columns. No automation is needed.

**Run now (baselines)** on a Kaggle T4 (prepare a notebook and give the user the commands):
- `Qwen/Qwen3-1.7B` (the original)
- Optional, if time allows: `sarvamai/sarvam-1` or another Indic baseline.

**Later (when the teammate's model arrives):**
- `--model <trained-bigo-model> --tag bigo_trained`.
- If he also trains a standard-BPE-extension model, run it as `--tag std_ext_trained`.

**Final matrix:** `results/eval_matrix.csv` + `results/eval_matrix.md`, built by `python -m eval.build_matrix`.
- Rows: models.
- Columns: every metric above × form.
- Plus the Level 1 headline columns for each model's tokenizer.
- Missing models show "pending", not blank.
- Generate `results/SLIDE_NUMBERS.md`, with the 6–8 numbers that go on slides and the file each comes from.

## Part D: Handoff package (tokenizer only — NO training code)

Do not write any training code, training spec or model check script. Create `dist/bigo_final_tokenizer.zip`. It contains:

- `bigo_final/`: the HF tokenizer folder (`tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, Qwen3 chat template).
- `README.md`: 10 lines max. Base vocab = Qwen3 (151,669) + N new tokens; new IDs start at 151,669; English/JSON/tool-call IDs are unchanged; the tokenizer's SHA-256.
- `results/level1_matrix.csv` and `results/kanglish_tax.md`, so the teammate sees what the tokenizer achieves.

That is all. The teammate handles training himself.

---

## Order of work and checkpoints

1. Unzip the romanization results; run Level 1 on the existing ablation builds. This gives a quick first matrix. **Tell the user the headline parity numbers right away; he needs them for a presentation.**
2. Part A layer 4 + final build + tests.
3. Part B full matrix, charts, Kanglish tax, failures.
4. Part C scripts. Hand the user the Kaggle commands for the Qwen3 baseline run.
5. Part D: zip the final tokenizer.
6. Update the README: method, how to reproduce, results table, limitations (including "model trained by teammate on AMD; Level 2 skipped" if so).

Update `docs/TASKS.md`: mark training tasks "moved to teammate", and tick items as they are done.
