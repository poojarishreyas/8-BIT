"""Run the full model evaluation matrix on one model (any HF folder or hub id). Same command for every model:

  python -m eval.run_all --model <hf-path-or-folder> --tag <name> --out results/models/<tag>/ [--data <test dir>]

Device: cuda if present (also covers ROCm/AMD), else cpu. Precision: bf16 where supported, fp16 on T4 (model/hf_load.py).
Tasks (each a separate step; a failure is logged and the rest continue):
  belebele     kan_Knda, eng_Latn, kan_Latn (romanized passages), log-likelihood of A-D, all 900 (--limit)
  chrf         FLORES devtest eng->kan, kan->eng (+ hand-written Kanglish->eng), greedy, batched (--chrf-limit, default 200)
  speed_cost   prefill/decode tokens/s, latency, ₹ per 1,000 answers (stated price assumption)
  three_route  native Kannada vs translate-to-English vs Kanglish on the same 200 Belebele questions
  agent        tool choice, argument exact/F1, valid-JSON rate, tokens per query (data/processed/test/agent_queries.jsonl)
Then `python -m eval.build_matrix` assembles results/eval_matrix.{csv,md}.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ["belebele", "chrf", "speed_cost", "three_route", "agent"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default=None, help="default results/models/<tag>/")
    ap.add_argument("--data", default=None, help="test-set folder (default: data/raw/test + data/processed/test via a merged view)")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    ap.add_argument("--limit", type=int, default=None, help="Belebele questions per language (default all 900)")
    ap.add_argument("--chrf-limit", type=int, default=200, help="FLORES sentences (1012 = full devtest)")
    ap.add_argument("--route-limit", type=int, default=200)
    ap.add_argument("--agent-queries", default=str(ROOT / "data/processed/test/agent_queries.jsonl"))
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    out = Path(a.out or ROOT / "results/models" / a.tag)
    out.mkdir(parents=True, exist_ok=True)
    data = a.data
    if data is None:  # local layout: test files live in two folders; link them into one view
        view = ROOT / "artifacts/test_view"
        view.mkdir(parents=True, exist_ok=True)
        for d in (ROOT / "data/raw/test", ROOT / "data/processed/test"):
            for f in d.glob("*"):
                if f.is_file() and not (view / f.name).exists():
                    (view / f.name).write_bytes(f.read_bytes())
        data = str(view)
    common = ["--model", a.model] + (["--tokenizer", a.tokenizer] if a.tokenizer else []) + \
             (["--device", a.device] if a.device else [])
    py = [sys.executable, "-m"]
    cmds = {
        "belebele": py + ["eval.level3.belebele", *common, "--data", data, "--out", str(out)] +
                    (["--limit", str(a.limit)] if a.limit else []),
        "chrf": py + ["eval.level3.flores_chrf", *common, "--data", data, "--out", str(out), "--limit", str(a.chrf_limit)],
        "speed_cost": py + ["eval.level3.speed_cost", *common, "--data", data, "--out", str(out)],
        "three_route": py + ["eval.level3.three_route", *common, "--data", data, "--out", str(out), "--limit", str(a.route_limit)],
        "agent": py + ["eval.agent.run", *common, "--queries", a.agent_queries, "--out", str(out)],
    }
    status = {}
    for t in a.tasks:
        t0 = time.time()
        print(f"=== {t}: {' '.join(cmds[t])}", flush=True)
        r = subprocess.run(cmds[t], cwd=ROOT)
        status[t] = dict(ok=r.returncode == 0, seconds=round(time.time() - t0, 1))
        print(f"=== {t} {'ok' if r.returncode == 0 else 'FAILED'} in {status[t]['seconds']}s", flush=True)
    (out / "run_meta.json").write_text(json.dumps(dict(tag=a.tag, model=a.model, tokenizer=a.tokenizer, tasks=status,
                                                       date=dt.datetime.now().isoformat(timespec="seconds")), indent=1),
                                      encoding="utf-8")


if __name__ == "__main__":
    main()
