"""Train Qwen3-1.7B with an extended tokenizer on one T4 (fp16, SDPA): Stage A / B / C.

  --stage A   new embedding rows only (tied: they are also the new output rows). Base model frozen.
  --stage B   LoRA continued pretraining (+ new rows keep training at a lower LR).
  --stage C   instruction + tool-call tuning on chat jsonl (loss on assistant turns only), LoRA + new rows.

New rows live in a separate fp32 table (ExtEmbedding / ExtHead): we never put the whole
~150K x 2048 matrix under Adam (CLAUDE.md). Without --tokenizer, the original Qwen3 is trained (no new
rows; stage A is then skipped), which gives the "same training stages" baseline.

Checkpoints (every --save-minutes, default 30, and at the end) go to --out:
  new_rows.pt        fp32 new rows + n_old/n_total + init method
  adapter/           PEFT LoRA (stages B, C)
  trainer_state.pt   optimizer, scaler, step, tokens/chars seen, stage  (same-stage resume only)
  log.csv, run.json
--resume DIR (a Kaggle input dataset or a previous /kaggle/working/out) loads weights; if DIR holds a
trainer_state.pt for the *same* stage, training continues from that step.
--export writes a merged fp16 HF model (+ tokenizer) to <out>/hf for eval/level3 and eval/agent.

Data: stages A/B read <data>/{train.bin,val.bin,meta.json} from eval.level2.prepare (--eos-token
"<|endoftext|>"); stage C reads a jsonl of {"messages": [...], "tools": [...]?}.
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
import torch.nn as nn
import torch.nn.functional as F

BASE = "Qwen/Qwen3-1.7B"


def dtype_kw(dtype) -> dict:
    """transformers >= 4.56 takes `dtype=`; older versions silently ignore it and need `torch_dtype=`."""
    import transformers
    from packaging.version import Version

    return {"dtype": dtype} if Version(transformers.__version__) >= Version("4.56") else {"torch_dtype": dtype}


# ------------------------------------------------------------------ extended vocab modules
class ExtEmbedding(nn.Module):
    """ids < n_old -> frozen original rows; ids >= n_old -> trainable fp32 rows."""

    def __init__(self, old_weight: torch.Tensor, new_weight: nn.Parameter):
        super().__init__()
        self.old = old_weight  # frozen buffer-like tensor (shared with ExtHead when tied)
        self.new = new_weight
        self.n_old = old_weight.shape[0]

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        is_new = ids >= self.n_old
        old = F.embedding(ids.clamp(max=self.n_old - 1), self.old)
        if self.new.numel() == 0 or not bool(is_new.any()):
            return old
        new = F.embedding((ids - self.n_old).clamp(min=0), self.new).to(old.dtype)
        return torch.where(is_new.unsqueeze(-1), new, old)


class ExtHead(nn.Module):
    def __init__(self, old_weight: torch.Tensor, new_weight: nn.Parameter):
        super().__init__()
        self.old, self.new = old_weight, new_weight

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        lo = F.linear(h, self.old.to(h.dtype))
        if self.new.numel() == 0:
            return lo
        return torch.cat([lo, F.linear(h, self.new.to(h.dtype))], dim=-1)


def load_tokenizers(tok_dir: str | None, base: str):
    from transformers import AutoTokenizer

    base_tok = AutoTokenizer.from_pretrained(base)
    if not tok_dir:
        return base_tok, base_tok
    tok = AutoTokenizer.from_pretrained(tok_dir)
    # transformers can silently rebuild a tokenizer's pre-tokenizer from class defaults; verify against the file
    from tokenizers import Tokenizer

    raw = Tokenizer.from_file(str(Path(tok_dir) / "tokenizer.json"))
    for s in ["ಮನೆಯಲ್ಲಿ ಇದೆ.", "ಕಾರ್‌ನಲ್ಲಿ", "It is in the house.", '{"a": 1}']:
        if tok(s, add_special_tokens=False)["input_ids"] != raw.encode(s, add_special_tokens=False).ids:
            raise SystemExit(f"AutoTokenizer({tok_dir}) differs from its tokenizer.json on {s!r}; "
                             "export it with `python -m tokenizer.export_hf`")
    return tok, base_tok


def mean_init(model, base_tok, tok, n_old: int, n_total: int) -> torch.Tensor:
    """Each new token = mean of the original embeddings of its pieces under the original tokenizer."""
    from tokenizers import Tokenizer

    base_raw = Tokenizer.from_str(base_tok.backend_tokenizer.to_str())
    E = model.get_input_embeddings().weight.detach().float()
    rows = torch.empty(n_total - n_old, E.shape[1])
    for i in range(n_old, n_total):
        piece = tok.convert_ids_to_tokens(i)  # byte-level string; may be any byte sequence
        old_ids = [base_raw.token_to_id(t.value) for t in base_raw.model.tokenize(piece)]
        rows[i - n_old] = E[old_ids].mean(0)
    return rows


def focus_init(model, base_tok, tok, n_old: int, data_txt: str) -> torch.Tensor:
    from deepfocus import FOCUS  # pip install deepfocus (needs fasttext training on data_txt)

    W = FOCUS(source_embeddings=model.get_input_embeddings().weight.detach().float(), source_tokenizer=base_tok,
              target_tokenizer=tok, target_training_data_path=data_txt)
    return W[n_old:].float().cpu()


def build_model(a, dev):
    from transformers import AutoModelForCausalLM

    tok, base_tok = load_tokenizers(a.tokenizer, a.base)
    dtype = torch.float16 if dev.type == "cuda" else torch.float32  # CPU only for smoke tests
    model = AutoModelForCausalLM.from_pretrained(a.base, **dtype_kw(dtype), attn_implementation="sdpa")
    n_old = len(base_tok)
    n_total = len(tok)
    assert n_total >= n_old
    tied = bool(getattr(model.config, "tie_word_embeddings", False))
    resume = Path(a.resume) if a.resume else None
    init = "none"
    if n_total > n_old:
        if resume and (resume / "new_rows.pt").exists():
            s = torch.load(resume / "new_rows.pt", map_location="cpu")
            assert s["n_old"] == n_old and s["n_total"] == n_total, "new_rows.pt was made for another tokenizer"
            rows, rows_out, init = s["rows"], s.get("rows_out"), s["init"]
        elif a.init == "focus":
            rows, rows_out, init = focus_init(model, base_tok, tok, n_old, a.focus_data), None, "focus"
        else:
            rows, rows_out, init = mean_init(model, base_tok, tok, n_old, n_total), None, "mean"
    else:
        rows, rows_out = torch.empty(0, model.config.hidden_size), None
    model.to(dev)
    for p in model.parameters():
        p.requires_grad_(False)
    old_in = model.get_input_embeddings().weight.data[:n_old]
    new_in = nn.Parameter(rows.to(dev, torch.float32))
    model.model.embed_tokens = ExtEmbedding(old_in, new_in)
    if tied:
        model.lm_head = ExtHead(old_in, new_in)
        new_out = None
    else:
        old_out = model.lm_head.weight.data[:n_old]
        new_out = nn.Parameter((rows_out if rows_out is not None else rows).to(dev, torch.float32))
        model.lm_head = ExtHead(old_out, new_out)
    model.config.vocab_size = n_total
    model.config.tie_word_embeddings = False  # we tie by sharing tensors ourselves; stop HF re-tying
    info = dict(n_old=n_old, n_total=n_total, n_new=n_total - n_old, tied=tied, init=init)
    return model, tok, new_in, new_out, info


# ------------------------------------------------------------------ data
class BinData:
    def __init__(self, d: Path, seq_len: int):
        self.meta = json.loads((d / "meta.json").read_text())
        dtp = np.dtype(self.meta["dtype"])
        self.tr = np.memmap(d / "train.bin", dtype=dtp, mode="r")
        self.va = np.memmap(d / "val.bin", dtype=dtp, mode="r")
        self.L = seq_len

    def batch(self, arr, n: int, gen, dev):
        ix = gen.integers(0, len(arr) - self.L - 1, n)
        x = torch.from_numpy(np.stack([arr[i:i + self.L + 1].astype(np.int64) for i in ix])).to(dev)
        return x[:, :-1], x[:, 1:]


class ChatData:
    """Stage C: render with the chat template; labels only on assistant turns (header excluded, <|im_end|> included)."""

    def __init__(self, path: Path, tok, seq_len: int):
        self.items = []
        start, end = "<|im_start|>assistant\n", "<|im_end|>"
        for line in path.read_text(encoding="utf-8").splitlines():
            ex = json.loads(line)
            text = tok.apply_chat_template(ex["messages"], tools=ex.get("tools"), tokenize=False)
            enc = tok(text, add_special_tokens=False, return_offsets_mapping=True, truncation=True, max_length=seq_len + 1)
            spans, k = [], 0
            while (s := text.find(start, k)) >= 0:
                e = text.find(end, s)
                e = len(text) if e < 0 else e + len(end)
                spans.append((s + len(start), e))
                k = e
            lab = [t if any(a <= o0 and o1 <= b for a, b in spans) else -100
                   for t, (o0, o1) in zip(enc["input_ids"], enc["offset_mapping"])]
            if any(l != -100 for l in lab[1:]):
                self.items.append((enc["input_ids"], lab))
        self.pad = tok.pad_token_id

    def batch(self, n: int, gen, dev):
        pick = [self.items[i] for i in gen.integers(0, len(self.items), n)]
        L = max(len(x) for x, _ in pick)
        X = torch.full((n, L), self.pad, dtype=torch.long)
        Y = torch.full((n, L), -100, dtype=torch.long)
        for r, (x, y) in enumerate(pick):
            X[r, :len(x)] = torch.tensor(x)
            Y[r, :len(y)] = torch.tensor(y)
        return X[:, :-1].to(dev), Y[:, 1:].to(dev)


# ------------------------------------------------------------------ save / export
def save(out: Path, model, new_in, new_out, info, state: dict | None, lora: bool):
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "new_rows.pt.tmp"
    torch.save(dict(rows=new_in.detach().float().cpu(), rows_out=None if new_out is None else new_out.detach().float().cpu(),
                    **info), tmp)
    os.replace(tmp, out / "new_rows.pt")
    if lora:
        model.save_pretrained(str(out / "adapter"))
    if state is not None:
        torch.save(state, out / "trainer_state.pt.tmp")
        os.replace(out / "trainer_state.pt.tmp", out / "trainer_state.pt")


def export(a, out: Path):
    """Merged fp16 HF model with resized embeddings, for evaluation/serving with plain transformers."""
    from transformers import AutoModelForCausalLM

    src = Path(a.resume) if a.export_only else out
    tok, base_tok = load_tokenizers(a.tokenizer, a.base)
    model = AutoModelForCausalLM.from_pretrained(a.base, **dtype_kw(torch.float16), attn_implementation="sdpa")
    n_old, n_total = len(base_tok), len(tok)
    if (src / "adapter").exists():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(src / "adapter")).merge_and_unload()
    if n_total > n_old:
        s = torch.load(src / "new_rows.pt", map_location="cpu")
        model.resize_token_embeddings(n_total, mean_resizing=False)
        with torch.no_grad():
            model.get_input_embeddings().weight[n_old:n_total] = s["rows"].half()
            if s.get("rows_out") is not None:
                model.get_output_embeddings().weight[n_old:n_total] = s["rows_out"].half()
    hf = out / "hf"
    model.save_pretrained(str(hf), safe_serialization=True)
    tok.save_pretrained(str(hf))
    if a.tokenizer:  # keep the verified tokenizer files byte-identical
        import shutil

        for f in ("tokenizer.json", "tokenizer_config.json"):
            shutil.copyfile(Path(a.tokenizer) / f, hf / f)
    print("exported", hf)


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["A", "B", "C"], default=None)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--tokenizer", default=None, help="HF tokenizer dir from tokenizer.export_hf (omit = original Qwen3)")
    ap.add_argument("--data", default=None, help="A/B: prepared bin dir; C: chat jsonl")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", choices=["mean", "focus"], default="mean")
    ap.add_argument("--focus-data", default=None, help="Kannada text file for FOCUS fasttext")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr-rows", type=float, default=None, help="LR for new rows (default A 1e-3, B/C 1e-4)")
    ap.add_argument("--lr-lora", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--train-chars", type=float, default=None, help="A/B: stop after this many training characters")
    ap.add_argument("--steps", type=int, default=None, help="C (or override): number of optimizer steps")
    ap.add_argument("--max-hours", type=float, default=11.0, help="stop + save before the 12 h session limit")
    ap.add_argument("--save-minutes", type=float, default=30)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--export", action="store_true", help="after training, write merged HF model to <out>/hf")
    ap.add_argument("--export-only", action="store_true", help="only export from --resume (no training)")
    a = ap.parse_args()
    out = Path(a.out)
    if a.export_only:
        export(a, out)
        return
    if not a.stage or not a.data:
        raise SystemExit("--stage and --data are required for training")

    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    model, tok, new_in, new_out, info = build_model(a, dev)
    if a.stage == "A" and info["n_new"] == 0:
        raise SystemExit("stage A trains new rows only; the original tokenizer has none (skip to stage B)")

    lora = a.stage in ("B", "C")
    if lora:
        from peft import LoraConfig, PeftModel, get_peft_model

        res = Path(a.resume) if a.resume else None
        if res and (res / "adapter").exists():
            model = PeftModel.from_pretrained(model, str(res / "adapter"), is_trainable=True)
        else:
            model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.0, task_type="CAUSAL_LM",
                                                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                                     "gate_proj", "up_proj", "down_proj"]))
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()  # fp32 master weights for LoRA
    for p in (new_in, new_out):
        if p is not None and p.numel():
            p.requires_grad_(True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False

    lr_rows = a.lr_rows if a.lr_rows is not None else (1e-3 if a.stage == "A" else 1e-4)
    groups = [{"params": [p for p in (new_in, new_out) if p is not None and p.numel()], "lr": lr_rows, "weight_decay": 0.0}]
    if lora:
        groups.append({"params": [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n],
                       "lr": a.lr_lora, "weight_decay": 0.0})
    groups = [g for g in groups if g["params"]]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95))
    base_lrs = [g["lr"] for g in groups]
    cuda = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)

    if a.stage == "C":
        data = ChatData(Path(a.data), tok, a.seq_len)
        total_steps = a.steps or math.ceil(len(data.items) / (a.micro_batch * a.grad_accum))
        cpt = None
    else:
        data = BinData(Path(a.data), a.seq_len)
        m = data.meta
        if m.get("n_vocab_with_eos", 0) > info["n_total"] or (m["vocab_size"] != info["n_total"]):
            raise SystemExit(f"data was tokenized with vocab {m['vocab_size']}, model has {info['n_total']}")
        cpt = m["train_chars"] / m["train_tokens"]
        chars = a.train_chars or m["train_chars"]
        total_steps = a.steps or math.ceil(chars / cpt / (a.micro_batch * a.grad_accum * a.seq_len))

    step, tokens_seen, train_seconds = 0, 0, 0.0
    res = Path(a.resume) if a.resume else None
    if res and (res / "trainer_state.pt").exists():
        st = torch.load(res / "trainer_state.pt", map_location="cpu", weights_only=False)
        if st["stage"] == a.stage:
            opt.load_state_dict(st["opt"])
            scaler.load_state_dict(st["scaler"])
            step, tokens_seen, train_seconds = st["step"], st["tokens_seen"], st["train_seconds"]
            print(f"resumed stage {a.stage} at step {step}/{total_steps}")

    run = dict(**info, stage=a.stage, base=a.base, tokenizer=a.tokenizer, data=a.data, total_steps=total_steps,
               seq_len=a.seq_len, micro_batch=a.micro_batch, grad_accum=a.grad_accum, lr_rows=lr_rows,
               lr_lora=a.lr_lora if lora else None, lora_r=a.lora_r if lora else None,
               chars_per_token=cpt, date=dt.date.today().isoformat(), gpu=torch.cuda.get_device_name(dev) if cuda else "cpu")
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps(run, indent=1))
    print(json.dumps(run), flush=True)
    log = (out / "log.csv").open("a", newline="")
    lw = csv.writer(log)

    def lr_mult(i):
        if i < a.warmup:
            return (i + 1) / a.warmup
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, (i - a.warmup) / max(1, total_steps - a.warmup))))

    @torch.no_grad()
    def val_loss(n=10):
        if a.stage == "C":
            return ""
        model.eval()
        g = np.random.default_rng(999)
        ls = []
        for _ in range(n):
            x, y = data.batch(data.va, a.micro_batch, g, dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
                lg = model(input_ids=x).logits
            ls.append(float(F.cross_entropy(lg.float().flatten(0, 1), y.flatten())))
        model.train()
        return sum(ls) / len(ls)

    t_start, last_save = time.time(), time.time()
    model.train()
    stopped = "done"
    while step < total_steps:
        t0 = time.time()
        gen = np.random.default_rng([a.seed, step])
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * lr_mult(step)
        tot = 0.0
        for _ in range(a.grad_accum):
            if a.stage == "C":
                x, y = data.batch(a.micro_batch, gen, dev)
            else:
                x, y = data.batch(data.tr, a.micro_batch, gen, dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
                lg = model(input_ids=x).logits
            loss = F.cross_entropy(lg.float().flatten(0, 1), y.flatten(), ignore_index=-100) / a.grad_accum
            scaler.scale(loss).backward()
            tot += float(loss.detach())
            tokens_seen += int((y != -100).sum())
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        step += 1
        if cuda:
            torch.cuda.synchronize(dev)
        train_seconds += time.time() - t0
        if step % 10 == 0 or step == total_steps:
            vl = val_loss() if (step % a.eval_every == 0 or step == total_steps) else ""
            lw.writerow([step, tot, vl, tokens_seen, round(train_seconds, 1), opt.param_groups[0]["lr"]])
            log.flush()
            print(f"step {step}/{total_steps} loss {tot:.4f} val {vl} {tokens_seen / max(1e-9, train_seconds):,.0f} tok/s "
                  f"mem {(torch.cuda.max_memory_allocated(dev) if cuda else 0) / 1e9:.1f}GB", flush=True)
        state = None
        if time.time() - last_save > a.save_minutes * 60 or step == total_steps:
            state = dict(stage=a.stage, opt=opt.state_dict(), scaler=scaler.state_dict(), step=step,
                         tokens_seen=tokens_seen, train_seconds=train_seconds)
            save(out, model, new_in, new_out, info, state, lora)
            last_save = time.time()
            print(f"saved {out} at step {step}", flush=True)
        if (time.time() - t_start) / 3600 > a.max_hours:
            stopped = "max_hours"
            if state is None:
                save(out, model, new_in, new_out, info, dict(stage=a.stage, opt=opt.state_dict(), scaler=scaler.state_dict(),
                     step=step, tokens_seen=tokens_seen, train_seconds=train_seconds), lora)
            print(f"stopping at {a.max_hours} h; resume with --resume {out}", flush=True)
            break
    run.update(final_step=step, tokens_seen=tokens_seen, train_seconds=round(train_seconds, 1), stopped=stopped)
    (out / "run.json").write_text(json.dumps(run, indent=1))
    if a.export and stopped == "done":
        del model
        torch.cuda.empty_cache()
        a.resume = None
        export(a, out)


if __name__ == "__main__":
    main()
