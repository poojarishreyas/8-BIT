# Kaggle smoke test (Phase 1)

Settings: Accelerator **GPU T4 x2**, Internet **on** (needs a phone-verified account).

Cell 1: get the code (upload the repo as a Kaggle dataset, or clone it once it's on GitHub)
```
!cp -r /kaggle/input/big-o-token-tax /kaggle/working/repo
%cd /kaggle/working/repo
!pip install -q peft
```

Cell 2: run on GPU 0
```
!python -m model.smoke_test --seq-len 1024 --batch 4 --steps 10 --device cuda:0
```

Cell 3: download `results/kaggle_smoke_Tesla_T4_*.json` and commit it to `results/`.
**Then stop the session**, because the quota counts wall-clock time.

If it goes OOM at batch 4, use `--batch 2`.

**Known issue (found 20 Nov 2026):** the Kaggle image ships torchao 0.10.0, which makes peft 0.21.2 fail in `get_peft_model` ("incompatible version of torchao"). Run `!pip uninstall -y -q torchao` before any PEFT/LoRA code, in every Kaggle notebook.

**Measured (single T4, LoRA r=16, grad checkpointing, seq 1024, batch 4):** 756 training tokens/s (~2.7M tokens/hour per GPU), 21 generation tokens/s, 11.6 GB peak memory.
