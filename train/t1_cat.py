"""Retrain the public CatBoost Task 1 residual ensemble (d8 + d10, speed + flow) on Kaggle GPUs,
score it on the official February masks, and write February/validation/private predictions.

Uses the public pipeline source (tfb_source) mounted from the anubisyy notebook output."""
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA = glob.glob("/kaggle/input/**/kaggle_public", recursive=True)[0]
# the public pipeline source ships with this package as public_src/ (also in the anubisyy notebook output as tfb_source/)
SRC = os.path.dirname(sorted(glob.glob("/kaggle/input/**/public_src/prepare_state.py", recursive=True)
                             + glob.glob("/kaggle/input/**/tfb_source/prepare_state.py", recursive=True))[0])
OUT = Path("/kaggle/working")
env = dict(os.environ, TFB_DATA_ROOT=DATA, TFB_OUTPUT_ROOT=str(OUT), TFB_THREADS="4", PYTHONPATH=SRC)
t0 = time.time()


def sh(args, **kw):
    print(">>", " ".join(args), flush=True)
    return subprocess.Popen([sys.executable, *args], cwd=SRC, env=env, **kw)


if not (OUT / "train.npz").exists():
    assert sh(["prepare_state.py"]).wait() == 0
print(f"features ready {time.time()-t0:.0f}s", flush=True)

# speed and flow train side by side, one GPU each; d8 first, then d10
for depth, iters in ((8, 27000), (10, 13500)):
    procs = [sh(["train_state.py", "--channel", ch, "--device", str(g), "--depth", str(depth),
                 "--iterations", str(iters), "--tag", f"d{depth}"]) for g, ch in enumerate(("speed", "flow"))]
    assert all(p.wait() == 0 for p in procs)
    print(f"d{depth} trained {time.time()-t0:.0f}s", flush=True)

sys.path.insert(0, SRC)
os.environ.update(env)
import state_model as base  # noqa: E402
from prepare_state import features  # noqa: E402
from state_inference import StatePredictor  # noqa: E402

pred = StatePredictor(OUT)


def predict_frame(name, pi, frame, target):
    panel = base.Panel(name, pi)
    times, v, q, e = base.matrices(frame, panel)
    rows = times.get_indexer(target.timestamp); cols = pd.Index(panel.links).get_indexer(target.link_id)
    assert (rows >= 0).all() and (cols >= 0).all()
    regime = target.mask_regime.map({"R1": 0, "R2": 1, "R3": 2}).to_numpy()
    X, bv, bq = features(panel, pi, times, v, q, e, rows, cols, regime)
    target = target.copy()
    target["speed_kmh"] = np.clip(bv + pred.residual(X, "speed"), 1, 180)
    target["flow_vph"] = np.clip((bq + pred.residual(X, "flow")) * panel.lanes[cols], 0, 150000)
    return target, panel


cols_ = ["timestamp", "station_id", "link_id", "speed_kmh", "flow_vph", "is_score_eligible"]
feb, scores = [], {}
for pi, name in enumerate(base.all_panels()):
    paths = sorted(Path(DATA, "corridors", name, "train", "mainline_states_masked").glob("*/synthetic_mainline_2031_02_*.parquet"))
    frame = pd.concat([pd.read_parquet(p, columns=cols_) for p in paths], ignore_index=True)
    frame["link_id"] = frame.link_id.astype(str); frame["station_id"] = frame.station_id.astype(str)
    tpl = pd.read_csv(Path(DATA, "task1", name, "train", "sample_submission_state.csv"), dtype={"station_id": str, "link_id": str})
    tpl = tpl[tpl.timestamp.str.startswith("2031-02")]
    t, panel = predict_frame(name, pi, frame, tpl)
    truth = pd.concat([pd.read_parquet(p.as_posix().replace("mainline_states_masked", "mainline_states").rsplit("/", 2)[0]
                                       + "/year_month=2031-02/" + p.name, columns=["timestamp", "link_id", "speed_kmh", "flow_vph"])
                       for p in paths]).rename(columns={"speed_kmh": "tv", "flow_vph": "tq"})
    truth["link_id"] = truth.link_id.astype(str)
    m = t.merge(truth, on=["timestamp", "link_id"])
    lanes = pd.Series(panel.lanes, index=panel.links).reindex(m.link_id).to_numpy()
    s = []
    for r in ("R1", "R2", "R3"):
        k = (m.mask_regime == r).to_numpy()
        if k.any():
            rv = np.sqrt(np.mean((m.speed_kmh[k] - m.tv[k]) ** 2)); rq = np.sqrt(np.mean(((m.flow_vph[k] - m.tq[k]) / lanes[k]) ** 2))
            s.append(0.54 * max(0, 1 - rv / 25) + 0.46 * max(0, 1 - rq / 600))
    scores[name] = float(np.mean(s)); print(name, "FEB S_state", round(scores[name], 5), flush=True)
    feb.append(t)
pd.concat(feb).to_csv(OUT / "feb_state_catboost.csv", index=False)
fam = {}
for k, v in scores.items():
    fam.setdefault(k.rsplit("_", 1)[0], []).append(v)
print("FEB family-mean S_state", round(float(np.mean([np.mean(v) for v in fam.values()])), 5), flush=True)

for split in ("validation", "private"):
    parts = []
    for pi, name in enumerate(base.all_panels()):
        frame = base.month_frame(name, "", split=split)
        tpl = pd.read_csv(Path(DATA, "task1", name, split, "sample_submission_state.csv"), dtype={"station_id": str, "link_id": str})
        parts.append(predict_frame(name, pi, frame, tpl)[0])
    pd.concat(parts).to_csv(OUT / f"state_{split}_catboost.csv", index=False)
    print(split, "written", flush=True)
for f in ("train.npz", "valid.npz"):
    (OUT / f).unlink(missing_ok=True)  # keep the output small
print(f"total {time.time()-t0:.0f}s")
