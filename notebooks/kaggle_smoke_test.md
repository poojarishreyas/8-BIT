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
