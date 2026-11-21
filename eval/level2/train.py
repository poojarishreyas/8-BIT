"""Level 2 trainer: tiny GPT from scratch on pre-tokenized data, matched on training characters.

- One pass over the shared training documents (same chars for every tokenizer); steps are derived from
  the tokenizer's own token count, so a better-compressing tokenizer trains on fewer tokens.
- fp16 autocast + GradScaler, SDPA attention, AdamW, warmup + cosine. DDP via torchrun (2x T4).
- Checkpoints every --ckpt-minutes and resumes from <out>/ckpt.pt.
- At the end: BPC on every eval form in eval.json -> <results>/level2_bpc.csv (appends rows).

Usage (Kaggle, both GPUs):
  torchrun --nproc_per_node 2 -m eval.level2.train --data /kaggle/input/bigo-level2/l123_32k \
      --out /kaggle/working/runs/l123_32k_s0 --seed 0 --results /kaggle/working/results
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from eval.level2.bpc import bpc
from eval.level2.model import GPT, GPTConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir from eval.level2.prepare (train.bin, val.bin, eval.json, meta.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--results", default="results")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--n-layer", type=int, default=8)
    ap.add_argument("--n-head", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--global-batch-tokens", type=int, default=65536)
    ap.add_argument("--micro-batch", type=int, default=16, help="sequences per GPU per micro-step")
    ap.add_argument("--epochs", type=float, default=1.0, help="passes over the training characters")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--ckpt-minutes", type=float, default=25)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--max-steps", type=int, default=None, help="debug cap")
    a = ap.parse_args()

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local = 0, 1, 0
    dev = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    master = rank == 0
    torch.manual_seed(a.seed)

    data = Path(a.data)
    meta = json.loads((data / "meta.json").read_text())
    dt_ = np.dtype(meta["dtype"])
    tr = np.memmap(data / "train.bin", dtype=dt_, mode="r")
    va = np.memmap(data / "val.bin", dtype=dt_, mode="r")
    V, eos = meta["n_vocab_with_eos"], meta["eos_id"]

    cfg = GPTConfig(vocab_size=V, ctx=a.ctx, n_layer=a.n_layer, n_head=a.n_head, d_model=a.d_model)
    model = GPT(cfg).to(dev)
    tokens_per_micro = a.micro_batch * a.ctx
    accum = max(1, a.global_batch_tokens // (tokens_per_micro * world))
    step_tokens = accum * tokens_per_micro * world
    total_steps = math.ceil(a.epochs * len(tr) / step_tokens)
    if a.max_steps:
        total_steps = min(total_steps, a.max_steps)

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95), eps=1e-8)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ck = out / "ckpt.pt"
    step, train_seconds = 0, 0.0
    if ck.exists():
        s = torch.load(ck, map_location=dev)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); scaler.load_state_dict(s["scaler"])
        step, train_seconds = s["step"], s["train_seconds"]
        if master:
            print(f"resumed at step {step}")
    raw = model
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local])

    def lr_at(i: int) -> float:
        if i < a.warmup:
            return a.lr * (i + 1) / a.warmup
        p = (i - a.warmup) / max(1, total_steps - a.warmup)
        return a.lr * (a.min_lr_frac + (1 - a.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))

    def batch(arr, gen):
        ix = gen.integers(0, len(arr) - a.ctx - 1, a.micro_batch)
        x = np.stack([arr[i:i + a.ctx + 1].astype(np.int64) for i in ix])
        t = torch.from_numpy(x).to(dev, non_blocking=True)
        return t[:, :-1], t[:, 1:]

    @torch.no_grad()
    def val_loss(n=20) -> float:
        raw.eval()
        g = np.random.default_rng(12345 + rank)
        ls = []
        for _ in range(n):
            x, y = batch(va, g)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                ls.append(float(torch.nn.functional.cross_entropy(raw(x).float().flatten(0, 1), y.flatten())))
        raw.train()
        return sum(ls) / len(ls)

    if master:
        print(f"{meta['name']}: V={V} params={raw.n_params():,} (non-emb {raw.n_params(True):,}) "
              f"train_tokens={len(tr):,} train_chars={meta['train_chars']:,} steps={total_steps} "
              f"step_tokens={step_tokens} accum={accum} world={world}", flush=True)
    log = (out / "log.csv").open("a", newline="")
    lw = csv.writer(log)
    last_ck = time.time()
    model.train()
    while step < total_steps:
        t0 = time.time()
        gen = np.random.default_rng([a.seed, rank, step])
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        for m in range(accum):
            x, y = batch(tr, gen)
            sync = (m == accum - 1) or not ddp
            ctxm = model.no_sync() if (ddp and not sync) else torch.enable_grad()
            with ctxm:
                with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                    loss = torch.nn.functional.cross_entropy(model(x).float().flatten(0, 1), y.flatten()) / accum
                scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        step += 1
        if dev.type == "cuda":
            torch.cuda.synchronize()
        train_seconds += time.time() - t0
        if master and (step % 50 == 0 or step == total_steps):
            vl = val_loss() if step % a.eval_every == 0 or step == total_steps else ""
            lw.writerow([step, float(loss.detach()) * accum, vl, lr_at(step), round(train_seconds, 1)])
            log.flush()
            print(f"step {step}/{total_steps} loss {float(loss.detach())*accum:.4f} val {vl} {step_tokens*step/train_seconds:,.0f} tok/s", flush=True)
        if time.time() - last_ck > a.ckpt_minutes * 60 or step == total_steps:
            if master:
                torch.save(dict(model=raw.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(),
                                step=step, train_seconds=train_seconds, cfg=vars(cfg)), ck)
            last_ck = time.time()

    if master:
        ev = json.loads((data / "eval.json").read_text())
        rows = []
        for form, e in ev.items():
            r = bpc(raw, e, eos, a.ctx, dev)
            rows.append(dict(tokenizer=meta["name"], vocab_size=meta["vocab_size"], seed=a.seed, form=form.split(".")[0],
                             split=form.split(".", 1)[1] if "." in form else "", **r,
                             params=raw.n_params(), params_non_embedding=raw.n_params(True),
                             train_chars=meta["train_chars"], train_tokens=len(tr), steps=total_steps,
                             step_tokens=step_tokens, train_seconds=round(train_seconds, 1),
                             d_model=a.d_model, n_layer=a.n_layer, ctx=a.ctx, gpus=world,
                             date=dt.date.today().isoformat()))
            print(f"BPC {form}: {r['bpc']:.4f}", flush=True)
        res = Path(a.results)
        res.mkdir(parents=True, exist_ok=True)
        f = res / "level2_bpc.csv"
        new = not f.exists()
        with f.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            if new:
                w.writeheader()
            w.writerows(rows)
        (out / "bpc.json").write_text(json.dumps(rows, indent=1))
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
