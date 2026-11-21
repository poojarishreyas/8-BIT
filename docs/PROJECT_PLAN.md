# Project plan — The Token Tax (R3)

Status: plan, 20 Nov 2026. Results below marked `[__]` are not measured yet.

## 1. The problem

Language models read **tokens**. Tokenizers learned mostly from English cut Kannada into many small pieces, so the same sentence costs more tokens. Effects for Kannada and Kanglish users:

- **Cost:** paid APIs charge per token.
- **Speed:** each token takes time to generate.
- **Context:** less Kannada text fits in the model's window.

Reference numbers (cite in the pitch):

| Fact | Source |
| --- | --- |
| Kannada needs ~12× the tokens of English with cl100k (GPT-4) | Big Hack 2.0 booklet, R3 (arXiv 2607.24276) |
| Under cl100k, Indian languages average an 8.0× tax; o200k and XLM-R cut it by ~73%; merge failure correlates with the tax at r = 0.89 | Srivastava, "The Tokenizer Tax", arXiv 2607.24276 |
| On 22 Indian languages, parity: GPT-4 7.43×, Llama-3 6.36×, Llama-4 Scout 3.76×, GPT-4o 3.32×, Gemma-3 2.53×, Sarvam-30B 1.83×, IndicBERT-v2 1.55×, MuRIL 1.47×, indictok-256k 1.42× | huggingface.co/vivekvar/indictok-256k (1,024 IN22-Gen sentences) |
| Kannada floor: 6.09 tokens/word under `\p{L}+` vs 1.25 under `[\p{L}\p{M}]+` (calculated bound, not trained) | "Vowel Signs Are Not Letters", arXiv 2608.26449 |
| 63.3% of the most-downloaded text-generation repos use the letters-only pattern | same paper |
| Fewer tokens alone did not give better models across 64 LMs | "Tokenization Is More Than Compression", EMNLP 2024 |

## 2. Root cause

Tokenization = (1) pre-tokenization by regex, then (2) BPE merges **only inside** each pre-token. The common GPT-2-style pattern `\p{L}+` matches letters only. Kannada vowel signs (ಾ ಿ ೆ …) and virama (್) are Unicode marks (Mn/Mc), so the regex splits every word at them, and BPE can never repair it.

Verified: `ಮನೆಯಲ್ಲಿ ಇದೆ` ("it is in the house") → `ಮನ|ೆ|ಯಲ|್|ಲ|ಿ|ಇದ|ೆ` (8) with `\p{L}+`, `ಮನೆಯಲ್ಲಿ|ಇದೆ` (2) with `[\p{L}\p{M}]+`.

Note: GPT-4o's o200k pattern already has the mark-aware fix. Our gain shows against letters-only tokenizers (Qwen, Llama-3 style). Name which baseline uses which pattern.

## 3. Solution: four-layer tokenizer

| Layer | Change | Prior work | Our twist |
| --- | --- | --- | --- |
| 1 Mark-aware split | `[\p{L}\p{M}]+` | arXiv 2608.26449 (Nepali trained, Kannada bound only) | First trained Kannada result |
| 2 Akshara-safe | No boundary inside a grapheme cluster | GPE (COLING 2025), GAT-K Kannada tokenizer | Combined with 3 and 4 |
| 3 Suffix-aware | No merges across stem/case-ending boundaries | MorphBPE; Unigram aligns better with morphology on Indic | Kannada suffix list / Morfessor |
| 4 Parity-aware | Max–min merge choice across languages | Parity-aware BPE (ACL 2026) | With Indic pre-tokenization |

Optional variants to ablate: Unigram vs BPE, Indic normalization (IndicNLP), BPE-dropout, vocab sizes 8K–128K.

Stretch ideas (only if time): factored akshara embeddings (consonant cluster + vowel sign as two summed embeddings); vocabulary-budget allocation per language as an explicit optimisation with a parity constraint.

### Extend, don't replace

For the real model: keep all original tokens/merges/IDs; append new Kannada merges; change the split pattern. English/JSON/tool-call tokenization stays identical (tested).

## 4. Data

| Use | Dataset (verify exact HF names) |
| --- | --- |
| Parallel test text | FLORES-200 devtest (eng_Latn, kan_Knda); IN22-Gen |
| Training text | AI4Bharat Sangraha (Kannada subset), IndicCorp v2, Kannada Wikipedia; English sample |
| Romanization | AI4Bharat IndicXlit / Aksharantar (~2.9M Kannada pairs) |
| Kanglish test set | 150–200 FLORES-equivalent sentences written by the team (new) |
| Reading test | Belebele kan_Knda + eng_Latn |
| Tool-call tuning | Salesforce xlam-function-calling-60k (queries translated to Kannada/Kanglish), ~300 team-written demo examples, English examples |
| Kannada instruction data | e.g. Aya dataset (check Kannada coverage) |

Hygiene: no FLORES text in any training data (Belebele is built on FLORES). Kanglish set written before training.

## 5. Evaluation protocol

### Level 1 — no model
For each tokenizer × each form (English, Kannada, romanized Kannada, Kanglish): tokens per sentence, parity = tokens ÷ English tokens for the same sentence, fertility (tokens/word), chars per token, akshara-split rate, byte-fallback rate, Rényi efficiency (`tokenization-scorer`), morphological boundary accuracy on 100–200 hand-labelled words. Report vocab size.

Win: Kannada and Kanglish parity near 1.0×, English unchanged.

### Level 2 — tiny models from scratch
- nanoGPT-style 20–50M params; variants: plain BPE, layer-1-only, full tokenizer, byte-level.
- Same data mix, **matched on training characters** (not tokens), 2 seeds each.
- Metric on held-out text in each form:

```
BPC = sum_i( -ln p(t_i | t_<i) ) / ( ln 2 × number_of_characters )
```

Report mean ± spread. This is the proof that the tokenizer helps rather than just compresses. Never cut this.

### Level 3 — Qwen3-1.7B
Three models, identical training stages: (a) original, (b) standard Kannada extension (SentencePiece/BPE on Kannada, merged — Kan-LLaMA style), (c) ours.

Training stages for (b) and (c):
- A: new embedding/output rows only, ~1 h, Kannada text.
- B: + LoRA on attention/MLP, continued pretraining on mostly Kannada + some English + Kanglish, as many tokens as one session allows (measure tokens/s first).
- C: instruction + tool-call tuning, 1–2 h, Kannada/Kanglish/English, including "no tool needed" examples.

Tests: Belebele (lm-evaluation-harness, log-likelihood scoring) kan_Knda + eng_Latn + IndicXlit-romanized Kannada; FLORES en→kn and kn→en chrF++ (sacrebleu); tokens/s and time-to-first-token; ₹ per 1,000 answers at a stated price per million tokens; Kannada words that fit in 4K context; 20 Kannada + 20 Kanglish prompts rated blind by the team.

Three-route test on one question set: (1) native Kannada with our tokenizer, (2) translate → English → answer → translate back, (3) Kanglish. Report cost vs quality.

### Agent evaluation
Test set: 150 queries (50 Kannada, 50 Kanglish, 50 English), each labelled with correct tool + arguments.
Metrics: tool-choice accuracy, argument accuracy (exact / normalized), task success, tokens per task (all steps), time per task, steps before context fills.
Extra: same test with all 52 harness tools mounted vs our 6–8 (accuracy + prompt tokens).
Pitch point: the tax compounds — every step resends tools, history and tool results.

## 6. Agent demo

- Harness: our prebuilt harness, custom composition with only: `web_search`, `web_fetch`, `read`, `ask_user_question` (optional `todo_write`), plus MCP tools we write: Karnataka weather, BMTC bus times, mandi prices, Kannada Wikipedia (mock data allowed — say so).
- Drop: edit/str_replace_editor, bash/pwsh, run_code, jobs, subagents, workflow/ralph/goals, agent-team, Cordis, schedules. Turn off run_code mode. Cap loops at 2–3 steps. Trim the system prompt.
- Server: FastAPI + transformers, OpenAI-style `/v1/chat/completions` returning `tool_calls`; constrained decoding from each tool's JSON schema; response also carries token count and generation time. Original model on GPU 0, ours on GPU 1.
- Expose via Gradio share link or `cloudflared tunnel --url`. Backups: same server on laptop CPU; recorded video.
- Dashboard: same query to both models → answer, tokens per step, cumulative tokens, time, ₹. First screen = token visualizer that needs no model.

## 7. Compute (free only)

| Job | Where |
| --- | --- |
| Tokenizer training, Level 1 | Laptops (CPU) |
| Level 2 tiny models | Kaggle, one variant per teammate account, DDP on 2× T4 |
| Level 3 1.7B | Kaggle T4, fp16, LoRA + new rows only |
| Evaluation | Separate Kaggle sessions, one shared script |
| Demo | Kaggle T4 + tunnel; laptop CPU backup |

Phone-verify all Kaggle accounts now (needed for GPU + internet). Quota resets Sunday 00:00 UTC.

Rough T4 training estimates (measure!): 1.7B ≈ 10M tokens/h; 3–4B ≈ 4–5M tokens/h; 7B QLoRA ≈ 1–2M tokens/h. Stay at 1.7B.

## 8. Three phases

The build runs in three gated phases (full task list per phase in `CLAUDE.md` and `docs/TASKS.md`):

| Phase | Hours | Goal | Exit check |
| --- | --- | --- | --- |
| 1 Measure and set up | 0–12 | Baselines, data, reused tools installed, Kanglish set written, quick-win split-rule result, tests | Baseline + quick-win numbers in `results/`; Kanglish set frozen |
| 2 Build and train | 12–36 | Tokenizer assembled from reused parts, extended into Qwen3, tiny models and 1.7B stages A–B trained | Ablation table, BPC table, two trained 1.7B models saved |
| 3 Evaluate, demo, pitch | 36–48 | Stage C, Level 3 + agent evals, server, harness, dashboard, slides | Every slide number traceable to `results/`; backup demo works |

### Reused components (do not rebuild)

Hugging Face `tokenizers` (engine), Qwen3 `tokenizer.json` (base vocab), `akshara-tokenizer` (Kannada akshara rules, Apache-2.0), `morfessor` (suffix boundaries), `swiss-ai/parity-aware-bpe` (parity merges, MIT), `vivekvar-dl/indictok` (baseline, benchmark script, script rules, Apache-2.0), `ghilli` (grapheme-BPE reference), Petrov's `tokenization-fairness`, `tokenization-scorer`, `byte-premium-tool`, IndicXlit, `deepfocus`, nanoGPT, `lm-evaluation-harness`, `sacrebleu`, `outlines`. Details and gotchas: `CLAUDE.md` → "Reuse, don't rebuild".

Verified facts from checking the parts (20 Nov 2026):
- Qwen3-1.7B uses the letters-only pattern; `ಮನೆಯಲ್ಲಿ ಇದೆ.` = 17 tokens vs 6 for the English sentence.
- HF `tokenizers` accepts `[\p{L}\p{M}]+`; it keeps `ಮನೆಯಲ್ಲಿ`, `ಇದೆ`, `ಕ್ರಿಕೆಟ್` whole.
- HF `\X` grapheme clusters split the conjunct `ಕ್ರಿ` into `ಕ್` + `ರಿ`, so they are not aksharas; use `akshara-tokenizer`.

## 8b. Track view of the same schedule

| Track (owner) | Day 1 (0–24 h) | Day 2 (24–48 h) |
| --- | --- | --- |
| 1 Tokenizer [__] | Baselines measured; layer 1 quick win; add layers 2–4 one at a time | Vocab sweep 8K–128K; ablation table; trade-off curve |
| 2 Tiny models [__] | Training code; plain BPE vs ours, 2 seeds, overnight | Layer-1-only + byte-level variants; BPC table |
| 3 Real model [__] | Extend Qwen3-1.7B; init; stage A; stage B overnight | Stage C; Belebele, chrF++, speed, cost; three-route test |
| 4 Kanglish + demo [__] | Kanglish set written; IndicXlit romanization; Kanglish tax measured; server skeleton | Romanized-Kannada tokenizer; harness composition; MCP tools; dashboard; agent eval |

Hours 40–48: freeze experiments, fill slides from `results/`, failure analysis, rehearse.
Cut order if short on GPU: stretch ideas → Level 3 stage B length → second seed. Never cut Level 2 BPC.

## 9. Risks and failure analysis (report all)

- Fewer tokens but no quality gain (BPC/Belebele decide).
- English/JSON/code cost rises (report counts per variant).
- Tiny-model noise (2 seeds, show spread).
- Bad akshara/suffix splits on rare conjuncts, loanwords, sandhi (show 10 hand-checked failures).
- Kanglish spelling variation; IndicXlit errors on vowels, similar consonants, gemination.
- Weak baselines (must include o200k and indictok).
- Novelty: pieces are published; claim the combination for Kannada, the Kanglish measurement and the agent compounding measurement.
- Small model tool errors on ambiguous queries (report even if accuracy dips).

## 10. Deliverables

- Code repo with tests green.
- `results/` CSVs feeding: parity table, ablation table, BPC table, Level 3 table, agent table, tokens-per-task chart.
- Slide deck (already drafted in Claude; fill `[__]` placeholders from results).
- Idea description (submitted text) and live demo + backup video.

## Key references

- Petrov et al., Language Model Tokenizers Introduce Unfairness Between Languages, NeurIPS 2023 (code: AleksandarPetrov/tokenization-fairness)
- Foroutan et al., Parity-Aware BPE, ACL 2026, arXiv 2508.04796
- Velayuthan & Sarveswaran, Grapheme Pair Encoding, COLING 2025
- Vowel Signs Are Not Letters, arXiv 2608.26449
- Srivastava, The Tokenizer Tax, arXiv 2607.24276
- Schmidt et al., Tokenization Is More Than Compression, EMNLP 2024
- Dobler & de Melo, FOCUS, EMNLP 2023 (`pip install deepfocus`)
- Feher et al., Retrofitting LLMs with Dynamic Tokenization, ACL 2025
- Husain et al., RomanSetu, ACL 2024
- Madhani et al., Aksharantar / IndicXlit
- Karthika et al., Multilingual Tokenization through the Lens of Indian Languages, arXiv 2506.17789
- Kan-LLaMA (Tensoic), Ambari (CognitiveLab), Sarvam-1, indictok-256k
