"""Agent evaluation: labelled queries -> tool calls -> scores. Runs on any HF model folder (original or ours).

Loop per query (max --max-turns): chat template with tools -> greedy generate -> parse Qwen-style
<tool_call>{"name":..,"arguments":{..}}</tool_call> blocks -> mock tool results -> continue until a turn
has no tool call.

Scores per query
  tool_choice    first predicted tool name == first gold tool name (or both "no tool")
  arg_accuracy   share of the first gold call's arguments matched (normalised; gold may list alternatives)
  task_success   predicted call sequence matches gold names in order, with all gold args matched
  tokens         prompt + generated tokens summed over turns (cost); prompt_tokens_turn1 = context cost
  seconds        wall time
Query file (jsonl): {"id", "lang": "kan_Knda|kanglish|eng_Latn", "query", "gold": [{"name", "arguments": {k: v|[alts]}}]}
gold = [] means the model should answer without a tool.

Usage:
  python -m eval.agent.run --model /kaggle/working/ours_C/hf --queries eval/agent/queries_v1.jsonl \
      --out /kaggle/working/results [--extra-tools eval/agent/harness_extra_tools.json --toolset 52]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import time
import unicodedata
from pathlib import Path

import torch

from eval.agent.tools import TOOLS, call_mock, load_extra
from model.hf_load import load, model_label

CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def norm(v) -> str:
    return unicodedata.normalize("NFC", " ".join(str(v).lower().split())).strip(" .")


def arg_ok(gold, pred) -> bool:
    alts = gold if isinstance(gold, list) else [gold]
    return pred is not None and norm(pred) in {norm(x) for x in alts}


def parse_calls(text: str) -> list[dict]:
    calls = []
    for m in CALL.finditer(text):
        try:
            c = json.loads(m.group(1))
            args = c.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            calls.append({"name": c.get("name"), "arguments": args if isinstance(args, dict) else {}})
        except json.JSONDecodeError:
            calls.append({"name": None, "arguments": {}, "malformed": True})
    return calls


def render(tok, msgs, tools):
    try:
        return tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def run_query(model, tok, q: dict, tools: list, dev, max_turns: int, max_new: int) -> dict:
    msgs = [{"role": "user", "content": q["query"]}]
    calls, tokens, first_prompt, final = [], 0, None, ""
    t0 = time.perf_counter()
    for _ in range(max_turns):
        enc = tok(render(tok, msgs, tools), return_tensors="pt", add_special_tokens=False).to(dev)
        n_in = enc["input_ids"].shape[1]
        first_prompt = first_prompt or n_in
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        new = g[0, n_in:].tolist()
        tokens += n_in + len(new)
        text = tok.decode(new, skip_special_tokens=False)
        text = text.split("<|im_end|>")[0].split("</think>")[-1]
        turn = parse_calls(text)
        if not turn:
            final = CALL.sub("", text).strip()
            break
        calls += turn
        msgs.append({"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": c["name"] or "unknown", "arguments": c["arguments"]}} for c in turn]})
        for c in turn:
            msgs.append({"role": "tool", "content": call_mock(c["name"], c["arguments"])})
    secs = time.perf_counter() - t0
    gold = q.get("gold", [])
    first_pred = calls[0]["name"] if calls else None
    first_gold = gold[0]["name"] if gold else None
    if gold:
        ga = gold[0].get("arguments", {})
        pa = calls[0]["arguments"] if calls else {}
        arg_acc = sum(arg_ok(v, pa.get(k)) for k, v in ga.items()) / len(ga) if ga else float(bool(calls))
    else:
        arg_acc = 1.0 if not calls else 0.0
    ga = gold[0].get("arguments", {}) if gold else {}
    pa = calls[0]["arguments"] if calls else {}
    tp = sum(arg_ok(v, pa.get(k)) for k, v in ga.items())
    prec = tp / len(pa) if pa else (1.0 if not ga else 0.0)
    rec = tp / len(ga) if ga else 1.0
    arg_f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    arg_exact = int(bool(gold) and bool(calls) and tp == len(ga))
    seq_ok = [c["name"] for c in calls[: len(gold)]] == [g["name"] for g in gold] and \
        all(all(arg_ok(v, c["arguments"].get(k)) for k, v in g.get("arguments", {}).items()) for g, c in zip(gold, calls))
    success = seq_ok if gold else not calls
    return dict(id=q["id"], lang=q["lang"], tool_choice=int(first_pred == first_gold), arg_accuracy=round(arg_acc, 3),
                task_success=int(success), arg_exact=arg_exact, arg_f1=round(arg_f1, 3),
                attempted_call=int(bool(calls) or "<tool_call>" in text),
                valid_json=int(not any(c.get("malformed") for c in calls)),
                n_calls=len(calls), malformed=sum(1 for c in calls if c.get("malformed")),
                tokens=tokens, prompt_tokens_turn1=first_prompt, seconds=round(secs, 2),
                pred_calls=json.dumps(calls, ensure_ascii=False), final=final[:300])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--queries", default="data/processed/test/agent_queries.jsonl")
    ap.add_argument("--extra-tools", default=None, help="JSON list of extra tool schemas (e.g. to reach 52 tools)")
    ap.add_argument("--toolset", default=None, help="label for results, default = number of tools")
    ap.add_argument("--max-turns", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default=None, help="default: cuda if available (incl. ROCm), else cpu")
    a = ap.parse_args()

    model, tok = load(a.model, a.device, a.tokenizer)
    dev = next(model.parameters()).device
    tools = TOOLS + load_extra(a.extra_tools)
    toolset = a.toolset or str(len(tools))
    qs = [json.loads(l) for l in Path(a.queries).read_text(encoding="utf-8").splitlines() if l.strip()][: a.limit]
    for q in qs:  # data/processed/test/agent_queries.jsonl format: form / expected_tool / expected_args
        if "expected_tool" in q:
            q["gold"] = [{"name": q["expected_tool"], "arguments": q.get("expected_args", {})}] if q["expected_tool"] else []
            q["lang"] = q.get("form", q.get("lang"))
    label = model_label(a.model)
    rows = []
    for q in qs:
        r = run_query(model, tok, q, tools, dev, a.max_turns, a.max_new_tokens)
        rows.append(dict(model=label, toolset=toolset, **r))
        print(f"{q['id']} {q['lang']} choice={r['tool_choice']} args={r['arg_accuracy']} ok={r['task_success']} "
              f"tok={r['tokens']} {r['seconds']}s", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{label.replace('/', '_')}_{toolset}tools"
    with (out / f"agent_{tag}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summ = []
    for lang in sorted({r["lang"] for r in rows}) + ["all"]:
        rs = [r for r in rows if lang == "all" or r["lang"] == lang]
        m = lambda k: round(sum(r[k] for r in rs) / len(rs), 4)
        summ.append(dict(model=label, toolset=toolset, lang=lang, n=len(rs), tool_choice=m("tool_choice"),
                         arg_accuracy=m("arg_accuracy"), arg_exact=m("arg_exact"), arg_f1=m("arg_f1"),
                         valid_json_rate=(sum(r["valid_json"] for r in rs if r["attempted_call"]) /
                                          max(1, sum(r["attempted_call"] for r in rs))),
                         task_success=m("task_success"), tokens_per_task=m("tokens"),
                         prompt_tokens_turn1=m("prompt_tokens_turn1"), seconds_per_task=m("seconds"),
                         tokenizer_len=len(tok), date=dt.date.today().isoformat()))
    f = out / "agent_summary.csv"
    new = not f.exists()
    with f.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0]))
        if new:
            w.writeheader()
        w.writerows(summ)
    for s in summ:
        print(s)


if __name__ == "__main__":
    main()
