"""Kaggle T4 smoke test: load Qwen3-1.7B in fp16 (SDPA attention, no bf16, no FA2) and measure
  - training tokens/s: LoRA + AMP loss scaling + gradient checkpointing, random tokens
  - generation tokens/s: greedy decode
  - peak GPU memory
Writes results/kaggle_smoke_<gpu>_<date>.json.

Usage (on Kaggle): python -m model.smoke_test --seq-len 1024 --batch 4 --steps 10
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--gen-tokens", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default="results")
    a = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = torch.device(a.device)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16, attn_implementation="sdpa").to(dev)
    gpu = torch.cuda.get_device_name(dev)
    res = dict(model=a.model, gpu=gpu, dtype="fp16", attn="sdpa", seq_len=a.seq_len, batch=a.batch,
               date=dt.date.today().isoformat(), torch=torch.__version__)

    # generation speed (base model, before LoRA wrap)
    model.eval()
    prompt = tok.apply_chat_template([{"role": "user", "content": "ಬೆಂಗಳೂರಿನ ಬಗ್ಗೆ ಒಂದು ಪ್ಯಾರಾ ಬರೆಯಿರಿ."}],
                                     tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(prompt, return_tensors="pt").to(dev)
    with torch.no_grad():
        model.generate(**ids, max_new_tokens=8, do_sample=False)
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        g = model.generate(**ids, max_new_tokens=a.gen_tokens, min_new_tokens=a.gen_tokens, do_sample=False)
        torch.cuda.synchronize(dev)
    n_new = g.shape[1] - ids["input_ids"].shape[1]
    res["gen_tokens_per_s"] = n_new / (time.perf_counter() - t0)

    # training throughput: LoRA, fp32 master LoRA weights, fp16 activations via autocast + GradScaler
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                             "gate_proj", "up_proj", "down_proj"]))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    x = torch.randint(0, tok.vocab_size, (a.batch, a.seq_len), device=dev)
    torch.cuda.reset_peak_memory_stats(dev)
    times = []
    for step in range(a.steps):
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(input_ids=x, labels=x).loss
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(dev)
        times.append(time.perf_counter() - t0)
    steady = times[2:] or times
    res["train_tokens_per_s"] = a.batch * a.seq_len / (sum(steady) / len(steady))
    res["peak_mem_gb"] = torch.cuda.max_memory_allocated(dev) / 1e9
    res["final_loss"] = float(loss)
    res["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(json.dumps(res, indent=1))
    out = Path(a.out_dir) / f"kaggle_smoke_{gpu.replace(' ', '_')}_{res['date']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
