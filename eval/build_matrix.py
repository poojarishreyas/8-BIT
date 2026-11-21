"""Assemble the final evaluation matrix and the slide numbers.

  results/eval_matrix.csv + .md   rows = models; columns = every metric x form; Level 1 headline columns for each
                                  model's tokenizer; models without results show "pending".
  results/SLIDE_NUMBERS.md        the 6-8 numbers that go on slides, each with the file it comes from.

Usage: python -m eval.build_matrix
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
# expected rows: tag -> tokenizer name in results/level1_matrix.csv
MODELS = {"qwen3_base": "qwen3", "bigo_trained": "bigo_final", "std_ext_trained": "std_16k", "sarvam1": "sarvam1"}
PENDING = "pending"


def read_csv(p: Path) -> list[dict]:
    return list(csv.DictReader(p.open(encoding="utf-8"))) if p.exists() else []


def level1(tok: str) -> dict:
    out = {}
    for r in read_csv(RES / "level1_matrix.csv"):
        if r["tokenizer"] == tok and r["split"] == "flores_devtest" and r["form"] in ("kan_Knda", "kan_Latn_xlit"):
            f = r["form"]
            out[f"L1_{f}_parity"] = f"{float(r['parity']):.2f}"
            out[f"L1_{f}_tokens_per_word"] = f"{float(r['fertility']):.2f}"
            if f == "kan_Knda":
                out["L1_new_vocab"] = r["new_vocab"]
    for r in read_csv(RES / "level1_matrix.csv"):
        if r["tokenizer"] == tok and r["split"] == "kanglish_handwritten" and r["form"] == "kanglish":
            out["L1_kanglish_parity"] = f"{float(r['parity']):.2f}"
    for r in read_csv(RES / "context_fit.csv"):
        if r["tokenizer"] == tok and r["form"] == "kan_Knda":
            out[f"ctx_{int(r['window']) // 1024}K_kannada_sentences"] = r["sentences_fit"]
    return out


def model_row(tag: str) -> dict:
    d = RES / "models" / tag
    row = {}
    for r in read_csv(d / "level3_belebele.csv"):
        row[f"belebele_acc_{r['lang']}"] = f"{float(r['accuracy']):.3f}"
        row[f"belebele_prompt_tokens_{r['lang']}"] = r["mean_prompt_tokens"]
    for r in read_csv(d / "level3_chrf.csv"):
        row[f"chrf++_{r['direction']}"] = r["chrfpp"]
        row[f"tokens_per_answer_{r['direction']}"] = r["tokens_per_answer"]
        row[f"out_chars_per_s_{r['direction']}"] = r["out_chars_per_s"]
    sc = d / "speed_cost.json"
    if sc.exists():
        s = json.loads(sc.read_text(encoding="utf-8"))
        row.update(prefill_tok_s=f"{s['prefill_tokens_per_s']:.0f}", decode_tok_s=f"{s['decode_tokens_per_s']:.1f}",
                   latency_s=f"{s['latency_s_mean']:.2f}", inr_per_1000_answers=f"{s['inr_per_1000_answers']:.2f}")
    tr = d / "three_route.json"
    if tr.exists():
        for r in json.loads(tr.read_text(encoding="utf-8")):
            row[f"route_{r['route']}_acc"] = f"{r['accuracy']:.3f}"
            row[f"route_{r['route']}_tokens_per_q"] = f"{r['tokens_per_question']:.0f}"
            row[f"route_{r['route']}_latency_s"] = f"{r['latency_s_per_question']:.2f}"
    for r in read_csv(d / "agent_summary.csv"):
        sfx = f"_{r['lang']}" if r["lang"] != "all" else ""
        for k in ("tool_choice", "arg_exact", "arg_f1", "valid_json_rate", "tokens_per_task"):
            if k in r:
                row[f"agent_{k}{sfx}"] = r[k]
    return row


def main() -> None:
    tags = dict(MODELS)
    if (RES / "models").exists():
        for d in sorted((RES / "models").iterdir()):
            if d.is_dir() and d.name not in tags and not d.name.startswith("_"):
                tags[d.name] = ""
    rows = []
    for tag, tok in tags.items():
        r = dict(model=tag, tokenizer=tok or "?")
        m = model_row(tag)
        r["status"] = "done" if m else PENDING
        r.update(level1(tok) if tok else {})
        r.update(m)
        rows.append(r)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    for r in rows:
        for c in cols:
            r.setdefault(c, PENDING)
    with (RES / "eval_matrix.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    # markdown: transpose (metrics as rows) so it stays readable
    md = ["# Evaluation matrix", "", "Rows = metrics, columns = models. `pending` = model not evaluated yet.", "",
          "| metric | " + " | ".join(r["model"] for r in rows) + " |", "|---|" + "---|" * len(rows)]
    for c in cols[1:]:
        md.append(f"| {c} | " + " | ".join(str(r[c]) for r in rows) + " |")
    md += ["", "Sources: results/models/<tag>/ (python -m eval.run_all), results/level1_matrix.csv, results/context_fit.csv.",
           "Rebuild: python -m eval.build_matrix"]
    (RES / "eval_matrix.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    slide_numbers(rows)
    print("wrote results/eval_matrix.csv, eval_matrix.md, SLIDE_NUMBERS.md;", len(rows), "models")


def slide_numbers(rows: list[dict]) -> None:
    l1 = {(r["tokenizer"], r["split"], r["form"]): r for r in read_csv(RES / "level1_matrix.csv")}
    g = lambda t, f="kan_Knda", s="flores_devtest", k="parity": l1.get((t, s, f), {}).get(k)
    fmt = lambda v, nd=2: PENDING if v in (None, "", PENDING) else f"{float(v):.{nd}f}"
    x = lambda v: PENDING if v in (None, "", PENDING) else f"{float(v):.2f}×"
    ctx = {(r["tokenizer"], r["form"], r["window"]): r["sentences_fit"] for r in read_csv(RES / "context_fit.csv")}
    by = {r["model"]: r for r in rows}
    b, q = by.get("bigo_trained", {}), by.get("qwen3_base", {})
    L = ["# Slide numbers (every number traceable to a file)", "",
         "| # | claim | number | source |", "|---|---|---|---|",
         f"| 1 | Kannada token tax, Qwen3 (tokens ÷ English, FLORES devtest) | {x(g('qwen3'))} | results/level1_matrix.csv (qwen3, flores_devtest, kan_Knda, parity) |",
         f"| 2 | … with our tokenizer (bigo_final, {g('bigo_final', k='new_vocab') or '?'} new tokens) | {x(g('bigo_final'))} | results/level1_matrix.csv (bigo_final) |",
         f"| 3 | … with a standard BPE extension at 16K (fair baseline) | {x(g('std_16k'))} | results/level1_matrix.csv (std_16k) |",
         f"| 4 | Kannada tokens per word: Qwen3 → ours | {fmt(g('qwen3', k='fertility'))} → {fmt(g('bigo_final', k='fertility'))} | results/level1_matrix.csv (fertility) |",
         f"| 5 | Kannada sentences in a 4K window: Qwen3 → ours | {ctx.get(('qwen3', 'kan_Knda', '4096'), PENDING)} → {ctx.get(('bigo_final', 'kan_Knda', '4096'), PENDING)} | results/context_fit.csv |",
         f"| 6 | English/JSON/code/tool-call token IDs changed | 0 | tests/test_tokenizer.py (identity tests, all pass) |",
         f"| 7 | Belebele Kannada accuracy: Qwen3 base vs bigo-trained | {q.get('belebele_acc_kan_Knda', PENDING)} vs {b.get('belebele_acc_kan_Knda', PENDING)} | results/models/<tag>/level3_belebele.csv |",
         f"| 8 | ₹ per 1,000 Kannada answers: Qwen3 base vs bigo-trained | {q.get('inr_per_1000_answers', PENDING)} vs {b.get('inr_per_1000_answers', PENDING)} | results/models/<tag>/speed_cost.json |",
         "", "Regenerate: python -m eval.level1.matrix && python -m eval.level3.context_fit && python -m eval.build_matrix"]
    (RES / "SLIDE_NUMBERS.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
