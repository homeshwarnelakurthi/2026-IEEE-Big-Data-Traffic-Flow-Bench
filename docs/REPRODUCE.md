# Reproducing the solution

## 1. What you need

| Input | Content |
|---|---|
| Competition data | the official release folder `kaggle_public` (holds `corridors/`, `task1/`, `task2/`, `task4/`) |
| Trained weights | folder with `models/` (Kaggle dataset `homeshwarrao/tfb-solution`, private, shared on request): `models/nn` (16 network checkpoints), `models/catboost` (4), `models/t2_members` (7), `models/public_queue` (3) |
| This repository | the code |

Library versions are pinned in `requirements.txt` (Python 3.13, CUDA 12.8, numpy 2.1.3, pandas 2.3.3, pyarrow 23.0.1, scipy 1.16.3, lightgbm 4.6.0, catboost 1.2.10, torch 2.11.0). On Kaggle nothing needs installing.

## 2. Rebuild the final submission

```bash
export TFB_DATA_ROOT=/path/to/kaggle_public
export TFB_SOLUTION=/path/to/tfb-solution      # folder that holds models/
export TFB_OUT=./run                           # work and output folder
python tfb/run_all.py                          # writes ./run/submission.csv
```

- Stages can be run separately: `--stages t1,t2,t4,merge` (and `t1mix` to re-blend Task 1 only).
- Optional settings (defaults give the final submission): `TFB_W_NN` (network share on normal days, 0.8), `TFB_W_V3` (network v3 share inside the blocks, 0.65), `TFB_GF_RAMP` (on-ramp sensor in the gap-filler, 1), `TFB_NN_WEIGHTS` (which checkpoints to average).
- On Kaggle, `kaggle/reproduce/run.py` finds the three inputs automatically and runs the script.

**Run time:** about 70 minutes on Kaggle 2× T4. A CPU-only run works but takes several hours.

**Small run-to-run differences** (far below 1%): the gap-filler is retrained inside every run (LightGBM with 4 threads), and network inference uses fp16 on GPU.

## 3. Retrain the models

| Model | Command | Hardware / time |
|---|---|---|
| Network v1, seeds 1–2 | `TFB_SEEDS=1,2 TFB_BLOCK_P=0 python tfb/t1_nn.py` | 2× T4, ≈6 h |
| Network v2, seeds 1–8 | `TFB_SEEDS=1,2 python tfb/t1_nn.py` (then `3,4`, `5,6`, `7,8`; the last two pairs used `TFB_MAX_HOURS=6`) | 2× T4, ≈6 h per pair |
| Network v3, seeds 5–10 | `TFB_SEEDS=5,6 TFB_MAX_HOURS=4 python train/t1_nn_v3.py` (then `7,8`, `9,10`) | 2× T4, ≈4 h per pair |
| CatBoost residual models | `python train/t1_cat.py` | 2× T4, ≈55 min |
| Gap-filler | retrained automatically inside `run_all.py` (`tfb/t1_gapfill.py final`) | CPU, ≈20 min |
| Task 2 members | `python train/t2_member.py v11` (members `v11`–`v16`, `v20`) | CPU, ≈45 min each |
| Public queue models | released by the public notebook author; trainers in `third_party/public_src/train_*.py` | – |

Network checkpoints go to `models/nn/{v1,v2,v3}_seed*.pt`, Task 2 members to `models/t2_members/t2_<member>/`.

## 4. Checks

- `kaggle/reproduce/run.py` rebuilt the final submission exactly: every Task 1 cell within 1e-6 relative, Task 2 and Task 4 identical.
- `kaggle/t2_truncation_check/run.py` deletes every row after each Task 2 window's origin and re-runs Task 2. The output did not change (0 of 174,000 cells), so Task 2 uses no future data.

## 5. Running the experiment scripts

The scripts in `experiments/` were used for local validation. They import helpers from `tfb/`, so run them with:

```bash
PYTHONPATH=tfb python experiments/t1_blend.py
```

Some of them read intermediate outputs saved from my Kaggle training runs (folder set by `TFB_WORKSPACE`); those outputs are not part of this repository.
