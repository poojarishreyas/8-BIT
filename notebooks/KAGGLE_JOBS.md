# Kaggle jobs: exact commands

**Kaggle has 2 separate 16 GB T4s.** Every job below runs **two models at once, one per GPU**
(`--device cuda:0` / `cuda:1`). fp16 only, SDPA attention. Sessions max out at 12 h, so trainers stop
themselves at `--max-hours 11` and save. They also save every 30 min to `/kaggle/working/<run>`.

## 0. Upload once (Datasets → New Dataset → upload zip → **Private**)
| dataset slug | file | contents |
|---|---|---|
| `bigo-code` | `dist\bigo-code.zip` | code + exported tokenizers (`tokenizers_hf/`) |
| `bigo-test` | `dist\bigo-test.zip` | FLORES devtest, IN22-Gen, Belebele, parity dev (test only) |
| `bigo-train` | `dist\bigo-train.zip` | `A/<tok>/`, `B/<tok>/` token bins, `C/sft_v1.jsonl` |
| `bigo-level2` | `dist\bigo-level2.zip` | Level 2 bins (from-scratch 32K set, bytes, qwen3, l123z_16k) |

Rebuild zips any time with `.venv\Scripts\python notebooks\make_bundles.py`. **After a code change, upload
`bigo-code.zip` as a new dataset version.**

Run long jobs with **Save Version → "Save & Run All (Commit)"**. That runs headless for up to 12 h and keeps
`/kaggle/working` as the notebook's output. To resume, add that output as an input to the next notebook (it
mounts at `/kaggle/input/<notebook-slug>/...`) and pass it as `--resume`.

Tokenizer names: `l123z_16k` = **ours** (layers 1–3, ZW-aware split rule, 16K new tokens);
`std_16k` = **standard extension** baseline; no `--tokenizer` = **original Qwen3**.
`l12z_16k` (without suffix layer) is also shipped. To use it as "ours", swap the name everywhere.

## Common first cell (every notebook)
Settings: Accelerator **GPU T4 x2**, Internet **On**, Persistence **Files only**.
```python
import os, glob, subprocess
CODE  = glob.glob('/kaggle/input/bigo-code*')[0]
TRAIN = (glob.glob('/kaggle/input/bigo-train*') or [None])[0]
TEST  = (glob.glob('/kaggle/input/bigo-test*') or [None])[0]
W = '/kaggle/working'
if not os.path.exists(f'{W}/repo'):
    subprocess.run(['cp', '-r', CODE, f'{W}/repo'], check=True)
os.chdir(f'{W}/repo')
!pip uninstall -y -q torchao          # torchao 0.10 breaks peft's get_peft_model on the Kaggle image
!pip install -q peft sacrebleu
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-1.7B')  # download once, before two processes race for the cache
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def two(cmd0, cmd1, log0, log1):
    """Run two jobs, one per GPU, and wait for both (needed in commit mode)."""
    p = [subprocess.Popen(c + ['--device', d], stdout=open(f'{W}/{l}', 'w'), stderr=subprocess.STDOUT)
         for c, d, l in ((cmd0, 'cuda:0', log0), (cmd1, 'cuda:1', log1)) if c]
    for x in p: x.wait()
    for l in (log0, log1):
        if l: print(l, open(f'{W}/{l}').read()[-1500:])
PY = ['python', '-m', 'model.train_qwen']
OURS, STD = 'tokenizers_hf/l123z_16k', 'tokenizers_hf/std_16k'
TRAIN_CHARS = '40e6'   # same characters for all three models in stage B (fits one session each)
```

## Job 1: Stage A (new rows only), ours + standard extension (~2–4 h)
```python
two(PY + ['--stage', 'A', '--tokenizer', OURS, '--data', f'{TRAIN}/A/l123z_16k', '--out', f'{W}/ours_A'],
    PY + ['--stage', 'A', '--tokenizer', STD,  '--data', f'{TRAIN}/A/std_16k',   '--out', f'{W}/std_A'],
    'log_ours_A.txt', 'log_std_A.txt')
```
Mean init is the default. For the FOCUS ablation, add `'--init', 'focus', '--focus-data', <kannada.txt>` after
`!pip install -q deepfocus`, and give it a separate `--out`.

## Job 2: Stage B (LoRA continued pretraining), ours + standard extension
Add Job 1's output as an input; it mounts as `/kaggle/input/<job1-slug>/`.
```python
J1 = glob.glob('/kaggle/input/*/ours_A')[0].rsplit('/', 1)[0]
two(PY + ['--stage', 'B', '--tokenizer', OURS, '--data', f'{TRAIN}/B/l123z_16k', '--resume', f'{J1}/ours_A',
          '--train-chars', TRAIN_CHARS, '--out', f'{W}/ours_B'],
    PY + ['--stage', 'B', '--tokenizer', STD,  '--data', f'{TRAIN}/B/std_16k',   '--resume', f'{J1}/std_A',
          '--train-chars', TRAIN_CHARS, '--out', f'{W}/std_B'],
    'log_ours_B.txt', 'log_std_B.txt')
```
At about 756 tokens/s per T4 (measured), 40M characters is about **4.2 h ours / 5.4 h standard / 9.6 h original**.

## Job 3: Stage B for the original Qwen3 (same characters, LoRA only; no stage A)
GPU 1 is free here, so it can run the Level 2 notebook in another session, or the next eval.
```python
two(PY + ['--stage', 'B', '--data', f'{TRAIN}/B/qwen3', '--train-chars', TRAIN_CHARS, '--out', f'{W}/orig_B'],
    None, 'log_orig_B.txt', None)
```
If it stops at 11 h: rerun the same cell in a new session with `'--resume', '<prev output>/orig_B'` added.
It continues from the saved step.

## Job 4: Stage C (instruction + tool calls) + export merged models (~1–2 h each)
Inputs: Job 2 output (and Job 3 output for the original).
```python
J2 = glob.glob('/kaggle/input/*/ours_B')[0].rsplit('/', 1)[0]
C = f'{TRAIN}/C/sft_v1.jsonl'
two(PY + ['--stage', 'C', '--tokenizer', OURS, '--data', C, '--resume', f'{J2}/ours_B', '--seq-len', '2048', '--micro-batch', '1', '--grad-accum', '16',
          '--out', f'{W}/ours_C', '--export'],
    PY + ['--stage', 'C', '--tokenizer', STD,  '--data', C, '--resume', f'{J2}/std_B', '--seq-len', '2048', '--micro-batch', '1', '--grad-accum', '16',
          '--out', f'{W}/std_C', '--export'],
    'log_ours_C.txt', 'log_std_C.txt')
# original Qwen3 (after Job 3):
# J3 = glob.glob('/kaggle/input/*/orig_B')[0].rsplit('/', 1)[0]
# two(PY + ['--stage','C','--data',C,'--resume',f'{J3}/orig_B','--seq-len','2048','--micro-batch','1','--grad-accum','16','--out',f'{W}/orig_C','--export'], None, 'log_orig_C.txt', None)
```
Each run writes `<out>/hf/` (merged fp16 model + tokenizer, about 3.5 GB) for the evals.

## Job 5: Level 3 + agent evals (any model folder or hub id)
Inputs: `bigo-test` and the Job 4 output(s). Results go to `/kaggle/working/results/*.csv`.
```python
J4 = glob.glob('/kaggle/input/*/ours_C')[0].rsplit('/', 1)[0]
OUT = f'{W}/results'
def evals(model):
    return [['python', '-m', 'eval.level3.belebele', '--model', model, '--data', TEST, '--out', OUT],
            ['python', '-m', 'eval.level3.flores_chrf', '--model', model, '--data', TEST, '--out', OUT],
            ['python', '-m', 'eval.agent.run', '--model', model, '--queries', 'eval/agent/queries_v1.jsonl', '--out', OUT]]
A, B = evals(f'{J4}/ours_C/hf'), evals(f'{J4}/std_C/hf')
for ca, cb in zip(A, B):
    two(ca, cb, 'log_eval_a.txt', 'log_eval_b.txt')
# Reference: untuned original Qwen3 straight from the hub, and orig_C once Job 4b exists:
# for c in evals('Qwen/Qwen3-1.7B'): two(c, None, 'log_eval_orig.txt', None)
```
- `queries_v1.jsonl` is the team's 150 hand-written agent queries (format: `eval/agent/queries_example.jsonl`).
  Until it exists, use `eval/agent/queries_example.jsonl` for a dry run.
- 52 tools vs 8: add `'--extra-tools', 'eval/agent/harness_extra_tools.json', '--toolset', '52'` to the agent
  command (the file is a JSON list of 44 extra tool schemas; not written yet).
- Quick check before a full run: add `'--limit', '50'` to each command.

Download `/kaggle/working/results/` and copy the CSVs into the repo's `results/`.

## Job 6: Level 2 tiny models
Use `notebooks/kaggle_level2.ipynb` with inputs `bigo-code` + `bigo-level2`. Run once with `QUICK=True`
(timing), then `QUICK=False`. Both T4s are used by one DDP run at a time.

## Job 7: Romanization (IndicXlit)
`notebooks/kaggle_romanize_xlit.ipynb` with inputs `bigo-code` + `bigo-test`, **GPU T4 x1**.
