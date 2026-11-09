# Experiments and findings

This page records what I tried for each task, how I tested it, what I kept and what I dropped.
All numbers below come from my own **local validation** on held-out training data, not from the competition leaderboard.

Contents:

1. [Data exploration](#1-data-exploration)
2. [Local validation framework](#2-local-validation-framework)
3. [Task 1: state reconstruction](#3-task-1-state-reconstruction)
4. [Task 3: physics](#4-task-3-physics)
5. [Task 2: queue forecasting](#5-task-2-queue-forecasting)
6. [Task 4: OD estimation](#6-task-4-od-estimation)
7. [Final pipeline](#7-final-pipeline)
8. [Reproducibility checks](#8-reproducibility-checks)
9. [Lessons learned](#9-lessons-learned)

---

## 1. Data exploration

Scripts: `eda/`.

- **The traffic is simulated** by a cell-transmission (CTM/LWR) model with a triangular speed–flow relation. Per-link parameters (free speed, capacity, jam density, lanes, length) are released. Jams look the way such a model produces them: speed inside a jam is a flat plateau, the jam head stays at the bottleneck and the tail creeps upstream.
- **Sensor noise sets the ceiling.** In free flow the speed noise is 1.0–1.4 km/h on the D12 corridors and 2.3–4.0 km/h on D7. Estimating that noise per corridor (`experiments/t1_noise_floor.py`) shows that even a perfect noise-free answer scores only about **0.958** on Task 1.
- **Congestion is rare and recurrent.** It is almost all between 13:00 and 20:00, ranges from 0.4% of cells (I10 West) to about 10% (D7 I405 North), and is more common on weekends.
- **Task 2 structure.** Every "onset" window first shows a jam at exactly T+30 (true for all 120 training windows), and the released row at time T may be used.
- **Task 4 structure.** Each origin–destination pair has exactly one path. Counts exist only on mainline links, and the given prior is badly scaled (about 2.2× the counts on most corridors).

---

## 2. Local validation framework

The test labels are hidden and leaderboard feedback is limited, so the first job was a local test that follows the official rules.

| Part | What it does | Code |
|---|---|---|
| Task 1 scorer | Holds out **February** completely and scores it on the official masks with a re-implementation of the official rule (per-regime RMSE, flow per lane, 0.54/0.46 weights, family averaging) | `tfb/t1_common.py`, `experiments/t1_blend.py` |
| Task 2 window picker | Re-creates the official window selector (queue = speed ≤ 0.6 × free speed, coverage ≥ 70%, onset and ongoing rules). It recovers **79 of the 80** official training windows, so I can generate thousands of extra windows for training and testing | `experiments/t2_data.py`, `experiments/t2_check_selector.py` |
| Task 3 proxy | A closed-form local version of the conservation score (see §4). The organisers said Task 3 could not be scored locally | `experiments/t3_proxy.py` |
| Blank-block simulation | Copies the real Task 2 blank-block schedule into February, so Task 1 can be tested under test-time conditions | `experiments/t1_blocks_eval*.py` |

Reference points on February (Task 1 score):

| Method | Score |
|---|---:|
| Linear interpolation in time and space (`experiments/t1_baseline_interp.py`) | 0.913 |
| Perfect noise-free answer (ceiling) | ≈ 0.958 |

---

## 3. Task 1: state reconstruction

### 3.1 Models compared

| Model | Description | February score |
|---|---|---:|
| Interpolation | linear in time and space | 0.913 |
| CatBoost residual (public budget) | public feature pipeline; predicts the correction to an interpolation baseline | 0.9416 |
| CatBoost residual (3× iterations) | same features, trained much longer because the public model stopped at its iteration cap | 0.9424 |
| **Axial-attention network (mine)** | see below | **0.9545** (2 seeds) |

### 3.2 The axial-attention imputer (`tfb/t1_nn.py`)

- **Input:** a 14-channel image of one corridor-day (288 time steps × up to 64 links): observed speed and flow (zero where hidden), occupancy, an observed-mask channel, typical weekday/weekend profiles, time of day, weekend flag and static link features.
- **Architecture:** about 1M parameters; 6 blocks, each with a local 5×5 convolution, attention along time (each link sees its whole day), attention along links (each moment sees the whole corridor) and an MLP. It predicts the *difference from the typical profile*, not raw values.
- **Training:** each sample is a random training day with a fresh random mask at 20%, 30% or 50%, so the model never sees the same puzzle twice. The loss is written in score units. AdamW, cosine schedule, fp16, 40,000 steps, about 6 hours per seed on one T4 GPU.

Learning curve (February score):

| Step | Seed 1 | Seed 2 |
|---:|---:|---:|
| 10,000 | 0.9450 | 0.9457 |
| 20,000 | 0.9511 | 0.9503 |
| 30,000 | 0.9536 | 0.9529 |
| 40,000 | 0.9542 | 0.9540 |

**The network wins on all 10 corridors**, by +0.007 to +0.016.

### 3.3 Error analysis

| | CatBoost speed RMSE | Network speed RMSE |
|---|---:|---:|
| Free flow (96% of cells) | 1.54 km/h | 1.19 km/h |
| **Congested (4% of cells)** | **3.95 km/h** | **1.60 km/h** |
| 20% hidden | 1.52 | 1.14 |
| 50% hidden | 1.78 | 1.23 |

- The network's advantage is largest **inside jams** (60% lower error). It can copy the flat jam speed from visible parts of the same queue.
- It barely gets worse when more cells are hidden (+8% from 20% to 50% masking, against +17% for CatBoost), because it uses the whole day and corridor, not only nearby cells.

### 3.4 The hidden failure: Task 2 blank blocks

On test days that host Task 2 windows, the organisers blank **every link for 90 minutes** after each forecast origin. These blocks do not exist in training data, so my February test could not see them.

| Step | What happened |
|---|---|
| Network v1 | Never saw such a block and broke down there (14–25 km/h errors). First fix: use CatBoost on those days. |
| Network v2 | Trained with simulated 90-minute blocks (probability 0.4). No cost on normal days (four seeds 0.9540–0.9543). Better on block days in simulation, yet still worse than expected in practice. |
| **Root cause** | About **2% of the Task 1 targets lie inside the blocks**, and in a realistic simulation they hold about **40% of all squared speed error**. v2 had been trained with those cells excluded, and my simulation had excluded them from scoring too. |
| Per-cell split | CatBoost inside the blocks, network v2 elsewhere on those days. |
| LightGBM gap-filler | A dedicated model that predicts across the gap from the values just before and after it, neighbours and position in the block (`tfb/t1_gapfill.py`). |
| Network v3 | Same architecture, but the block cells are part of the loss, and the hour before a block stays visible (it is republished as Task 2 history). Weaker overall (0.946), so used only inside blocks. |
| On-ramp jam sensor | When an on-ramp's flow drops below 20% of its normal level, the mainline next to it is almost always jammed (62–84% vs 0.5–4% base rate). Added as a gap-filler feature. |

Inside the blocks (68,555 February cells):

| Method | Speed RMSE (km/h) | Flow RMSE (veh/h/lane) | Score |
|---|---:|---:|---:|
| Network v2 | – | – | 0.556 |
| Time-of-day profile | – | – | 0.620 |
| Interpolation across the gap | – | – | 0.752 |
| CatBoost | 7.26 | 89.0 | 0.775 |
| Gap-filler | 6.61 | 63.9 | 0.808 |
| Gap-filler + ramp sensor | 6.22 | 62.7 | 0.818 |
| Network v3 (2 seeds) | 5.08 | 59.2 | 0.845 |
| 0.5 × v3 + 0.5 × gap-filler (ramp) | 5.29 | 57.4 | 0.842 |
| **0.65 × v3 + 0.35 × gap-filler (ramp), final** | – | – | **≈0.845** |

### 3.5 Ensembling

- More seeds help a little and steadily: in the end 2 v1 + 8 v2 seeds on normal days and 6 v3 seeds inside the blocks.
- A small CatBoost share (20%) is kept on normal days as a hedge between months. On February it neither helped nor hurt.

### 3.6 Things that did not work

| Idea | Result |
|---|---|
| Re-derive flow from interpolated density (to help physics) | Both Task 1 and the physics proxy got worse (`experiments/t3_density_test.py`) |
| Short-gap density post-processing (pull density toward neighbours) | Worse in every setting tried (`experiments/t1_density_pp.py`) |
| Blending in density space instead of speed/flow | Negligible change (`experiments/t1_density_blend.py`) |
| Adding network v3 to the normal-day average | Worse; v3 is a block specialist |
| CTM physics simulation to fill the blocks (`experiments/t1_ctm_gap.py`) | Far worse (21–44 km/h error). The released ramp flows do not balance the mainline, and real queues sit off the released speed–flow curve, so simulated queues dissolve at once |

---

## 4. Task 3: physics

Reading the official scorer closely shows that the organiser boundary flows are fitted to the released observations. On training data the observed state therefore conserves vehicles exactly, and the score for a submission reduces to:

```text
S_LWR(proxy) = 1 − Σ |ΔN_sub − ΔN_obs| / Σ |ΔN_obs|
```

where N is the number of vehicles on a link (density × length) and Δ is the change over 5 minutes (`experiments/t3_proxy.py`). Submitting the truth gives exactly 1.0.

What this means:

- Density is computed as flow ÷ speed, so **a speed error inside a jam counts far more** than the same error in free flow. Jam cells dominate the physics score.
- What matters is the *jump* in density error from one step to the next, so smooth, consistent predictions score better than noisy cell-by-cell ones.
- **Physics can only improve through better Task 1 accuracy.** No post-processing trick helps (confirmed in §3.6).

| Model | Task 1 score | Physics proxy |
|---|---:|---:|
| CatBoost | 0.9416 | 0.633 |
| CatBoost (longer training) | 0.9424 | 0.637 |
| Network, 2 seeds | 0.9545 | 0.685 |
| Network ensemble, 6 seeds + CatBoost | 0.9546 | 0.687 |

Each point of Task 1 gained by the network brings about 4 points on the physics proxy.

---

## 5. Task 2: queue forecasting

### 5.1 My own model (`experiments/t2_exp.py`, `experiments/t2_final.py`)

Separate LightGBM models for onset and ongoing windows, one row per link, with links in travel order. Features:

- recent history: last speed and flow, lags, 15-minute means and trends, minimum speed, queued steps;
- **the released row at time T** (no public notebook used it);
- a space-time patch of ±6 neighbouring links;
- distance to the nearest jam upstream and downstream;
- a long-run "bottleneck memory": how often each link is jammed at that time of day in training;
- time of day, day of week, number of active jams on the corridor.

Onset windows predict only the T+30 row (a jam is guaranteed to appear exactly there).

**Ablation** (window IoU on generated windows from held-out months):

| Features | Onset Jun | Onset Feb | Ongoing Jun | Ongoing Feb |
|---|---:|---:|---:|---:|
| Base | 0.918 | 0.902 | 0.869 | 0.859 |
| + row at T | 0.925 | 0.903 | 0.884 | 0.890 |
| + space-time patch | 0.923 | 0.907 | 0.900 | 0.905 |
| **+ bottleneck memory** | **0.938** | **0.923** | **0.903** | **0.909** |
| + same-month memory | 0.938 | 0.914 | 0.902 | 0.907 |

Locally it beat the public queue models clearly (ongoing IoU 0.898 vs 0.808 on February).

### 5.2 Why it was not used

On the March test windows my model did worse than the public models, although every local test favoured it. I checked several explanations:

| Hypothesis | Test | Result |
|---|---|---|
| Dead or stuck detectors | counted them | only 0.1% of cells; ruled out |
| Wrong jam threshold | compared with the official scorer | identical; ruled out |
| Different persistence filter | re-filtered February windows | my model still ahead; ruled out |
| Row at T is sparser at test time (Task 1 masking) | masked it in training too (`experiments/t2_rowT.py`) | real but small effect (+0.006 to +0.011 IoU) |
| Labels from noisy vs noise-free speeds, window sampling, month shift, few test windows | – | plausible, not resolved |

Only 80 windows are scored per test month, so the Task 2 score on one month is noisy.

### 5.3 What the final pipeline uses

To stay fully reproducible, I **retrained a public ensemble from its published training code** (7 LightGBM members, `train/t2_member.py`) and combined it with the public CatBoost/LightGBM queue models:

- **Onset:** public onset models (0.5 LightGBM + 0.5 CatBoost), T+30 row only, threshold 0.55, at least one link always set.
- **Ongoing:** 0.5 × my retrained 7-member ensemble + 0.5 × public ongoing LightGBM, threshold 0.45.

The retrained ensemble agrees with the original published predictions on 99.9% of cells.

I also tested ongoing thresholds (0.25 / 0.45 / 0.65), onset thresholds (0.45 / 0.55 / 0.65), ensemble weights (0.3 / 0.5 / 0.7), and feeding the row at T into the public model. On the March windows the middle settings were best and were kept.

---

## 6. Task 4: OD estimation

The baseline solves a prior-regularised non-negative least-squares problem:

```text
min_f  ½‖A f − c‖² + ½ λ ‖f − b‖²     subject to f ≥ 0
```

(A = link–path incidence, c = counts, b = prior), solved with L-BFGS-B from a closed-form starting point (`tfb/t4_variants.py`).

| Variant | Idea | Outcome |
|---|---|---|
| **L2, λ = 0.05** | additive correction of the prior | **kept** |
| IPF / KL to the prior | multiplicative fit that keeps the prior's shape | much worse |
| IPF from a 3-month averaged prior | less noisy prior | even worse |
| L2 with λ = 0.005 or 0.5 | weaker or stronger prior | no change (solutions differ by < 0.5%, `experiments/t4_lambda.py`) |

The hidden reference sits close to the additive L2 answer. Multiplicative methods rescale every path, including those without counts, and overshoot.

---

## 7. Final pipeline

Produced by `tfb/run_all.py`:

| Cells | Prediction |
|---|---|
| Task 1, normal days | 0.8 × mean(network v1 seeds 1–2, v2 seeds 1–8) + 0.2 × CatBoost |
| Task 1, Task 2 days outside the blocks | 0.8 × mean(network v2 seeds 1–8) + 0.2 × CatBoost |
| Task 1, inside the 90-minute blocks (≈2%) | 0.65 × mean(network v3 seeds 5–10) + 0.35 × LightGBM gap-filler with on-ramp sensor |
| Task 2 | public onset models + 50/50 blend of my retrained ensemble and the public ongoing model |
| Task 3 | nothing to submit (follows Task 1) |
| Task 4 | L2 OD solver, λ = 0.05 |

```text
official data + saved models
   └─ tfb/run_all.py
        t1    16 network checkpoints, CatBoost, per-cell blend, gap-filler retrained, block blend
        t2    retrained ensemble + public queue models
        t4    L2 solver
        merge validity checks → submission.csv
```

---

## 8. Reproducibility checks

- **Full rebuild on Kaggle** (`kaggle/reproduce/run.py`, 2× T4, about 70 minutes): compared cell by cell with the submitted file, all 6,735,795 Task 1 cells agree within 1e-6 relative (max 4.4e-8), and Task 2 and Task 4 are identical.
- **Task 2 uses no future data** (`kaggle/t2_truncation_check/run.py`): with every row after each window's origin deleted, all 160 windows gave identical output (0 of 174,000 cells changed, max probability change 0.0).
- Pinned versions: `requirements.txt` (and the full `pip freeze` of the Kaggle image in `requirements-kaggle-freeze.txt`).

---

## 9. Lessons learned

1. **The validation set must copy the test inputs, not just the test targets.** One wrong assumption ("cells inside the blank blocks are not scored") hid the biggest Task 1 error source through three rounds of fixes.
2. **Task 1 improvements are stable across months**; the network's gains were consistent on every corridor and held on the unseen test months.
3. **Task 2 is noisy.** With only 80 scored windows per month, small tuning choices made on one month did not carry over to the next. Features that rely on training-month habits (time-of-day bottleneck memory) are risky when the test month has new incidents. Simpler, less month-specific Task 2 settings would have been the safer choice.
4. **Physics follows accuracy.** Once the scoring is understood, the best physics result comes from the most accurate state reconstruction, not from post-processing.

## Future work

- Task 2 trained on denoised labels (from the network's reconstruction) and with less dependence on training-month time-of-day patterns.
- Longer network training (the learning curve was still rising at 40k steps) and a loss that weights jam density errors the way the physics score does.
- Better Task 4 priors that keep each month's own structure.
