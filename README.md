# 2026 IEEE Big Data Cup – Traffic Flow Bench

My solution to the **TrafficFlowBench** competition (IEEE Big Data Cup 2026, hosted on Kaggle).
Author: **Homeshwar Nelakurthi**, Kent State University.

---

## The problem in one minute

The data come from ten simulated freeway corridors in California, recorded every 5 minutes.
There are four tasks, combined into one score:

| Task | What we must predict | Weight |
|---|---|---:|
| **1. State** | Speed and flow at sensor readings that were hidden (20%, 30% or 50% of cells removed) | 35% |
| **2. Queue** | Which road links will be jammed 5–30 minutes from now | 30% |
| **3. Physics** | Nothing extra; it checks that the Task 1 answers obey traffic physics (conservation of vehicles) | 15% |
| **4. OD estimation** | How many trips go from each origin to each destination, given only link counts | 20% |

Training data covers 9 months. The test months (March and April) have no labels.

---

## My approach in short

| Task | Method |
|---|---|
| **1. State** | An **axial-attention neural network** I designed and trained from scratch. It looks at a whole day × whole corridor at once and fills the hidden cells. 16 trained copies (seeds) are averaged and lightly blended with a CatBoost model. Inside the 90-minute gaps that the organisers cut out for Task 2, a special network version plus a LightGBM "gap-filler" take over. |
| **2. Queue** | Gradient-boosting models (LightGBM + CatBoost) that read the last hour of traffic and decide link by link whether a jam will be present. Onset (new jam) and ongoing (existing jam) windows have separate models. |
| **3. Physics** | Improves automatically when Task 1 gets more accurate. I built a local scorer for it to check this. |
| **4. OD** | A regularised least-squares solver (L-BFGS-B) that matches link counts while staying close to the given prior. |

The whole final submission is rebuilt by **one script**: `tfb/run_all.py` (about 70 minutes on a Kaggle 2× T4 GPU machine).

The full story of what I tried, what worked and what did not is in **[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)**.

---

## Key ideas

1. **Build a trustworthy local test first.** The test labels are hidden, so I recreated the official scoring on a held-out training month (February), wrote a copy of the official Task 2 window picker (it finds 79 of 80 official windows), and derived a formula that scores Task 3 locally, which the organisers said was not possible.
2. **A neural network that sees the whole picture.** Traffic jams in this data are flat "plateaus" that stretch over many links. A model that attends along the full day and the full corridor can copy the jam from the parts it can see. It cut speed error inside jams by about 60% compared with CatBoost.
3. **Find the hidden failure.** About 2% of the Task 1 cells sit inside the 90-minute blank blocks made for Task 2, yet they held about 40% of the error. Training a network version on exactly those cells, plus a dedicated gap-filler that uses an on-ramp "jam sensor", fixed it.
4. **Keep it reproducible.** Every model is retrained from published code, all weights are saved, and a Kaggle run rebuilds the final file to within 1e-6.

---

## Repository layout

```
.
├── tfb/                  final pipeline (this is what produces the submission)
│   ├── run_all.py        ONE entry script: Task 1 + Task 2 + Task 4 → submission.csv
│   ├── t1_nn.py          axial-attention imputer: model, training (v1/v2) and inference
│   ├── t1_gapfill.py     LightGBM gap-filler for the Task 2 blank blocks
│   ├── t4_variants.py    Task 4 L2 OD solver
│   ├── common.py         data loading, paths, cached array views
│   ├── t1_common.py      Task 1 targets and the official scoring rule
│   ├── make_submission.py  merges the task files and runs validity checks
│   └── build_cache.py    optional: pre-builds the data cache
├── train/                scripts that retrain every saved model
│   ├── t1_nn_v3.py       network v3 (trained on the blank-block cells)
│   ├── t1_cat.py         CatBoost residual model for Task 1
│   └── t2_member.py      Task 2 LightGBM ensemble members
├── experiments/          local validation and every experiment I ran (not needed for the final run)
├── eda/                  first data exploration scripts
├── kaggle/               Kaggle launchers: full rebuild, and the Task 2 "no future data" check
├── third_party/          public notebook code I reused (credited in NOTICE)
├── docs/
│   ├── EXPERIMENTS.md    all experiments, results and lessons
│   └── REPRODUCE.md      step-by-step rebuild and retraining guide
├── requirements.txt      pinned library versions
└── NOTICE                credits for third-party code
```

---

## Quick start

**On Kaggle (recommended, no install needed):**

1. Create a notebook with **GPU T4 ×2**, internet off.
2. Attach: the competition data, the dataset `homeshwarrao/tfb-code` (this code) and `homeshwarrao/tfb-solution` (trained weights).
3. Run `kaggle/reproduce/run.py`. It writes `/kaggle/working/run/submission.csv`.

The trained weights (about 900 MB) are too large for GitHub, so they live in the Kaggle dataset `tfb-solution`, which is private and shared on request. Every model can also be retrained from scratch with the scripts in `train/` (see [docs/REPRODUCE.md](docs/REPRODUCE.md)).

**Anywhere else:**

```bash
pip install -r requirements.txt
export TFB_DATA_ROOT=/path/to/kaggle_public     # official data release
export TFB_SOLUTION=/path/to/tfb-solution       # folder that holds models/
export TFB_OUT=./run                            # work/output folder
python tfb/run_all.py                           # → ./run/submission.csv
```

A CPU-only run works too but takes several hours. Details, retraining commands and run times are in [docs/REPRODUCE.md](docs/REPRODUCE.md).

---

## Tools used

Python 3.13, PyTorch (fp16 on GPU), LightGBM, CatBoost, NumPy, pandas, SciPy. Training ran on Kaggle (2× Tesla T4).

## Credits

Built on the organisers' [TrafficFlowBench repository](https://github.com/jacky850/trafficflowbench-public) and two public Kaggle notebooks (anubisyy and yeashusemwal), whose code is in `third_party/` and credited in [NOTICE](NOTICE). Everything else is my own work.
