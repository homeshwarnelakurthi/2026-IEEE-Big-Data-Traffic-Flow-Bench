# Source: yeashusemwal, "TrafficFlowBench Trees + U-Net + CNN" (Kaggle notebook, 2026),
# kaggle.com/code/yeashusemwal/trafficflowbench-trees-u-net-cnn-lb-0-87. Redistributed for reproduction with credit; see NOTICE.
"""Inference-only replay of the Task 2 LightGBM members (no training).

Three code paths produced the seven GBM members used in the submission:
  v11 family (v11, v12, v13, v14): task2_v10 features with a full weekday profile at test time
  v15 family (v15, v16):           long-context features (3 h of prior state, queue geometry, ramps)
  v18 family (v20):                v10 features through v18's wrapper (history blanking is training-only)
Each member is a directory holding task2_general.txt, task2_general_w.txt and task2_onsetT30.txt.
dump() rebuilds the exact test-time features of the original training run (once per family) and returns the same
window_probs layout (split, panel, window_id, condition, h, link_id, p_general, p_general_w, p_onsetT30).
"""
import os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "v2"))
import task2_v10 as v10, task2_v11 as v11
from common import DATA, network, day_files, date_of, weekday_of, slots_of, day_grid

H = v10.H
Q2 = v10.Q2
_STATS = {}


def stats(panel):
    """Weekday profile sums per panel (shared by all three families; v18 and v15 use the same definition)."""
    if panel not in _STATS:
        net = network(panel); st, days = v11.profile_stats(panel, net)
        _STATS[panel] = (net, st, days, v11.profile_from(st))
    return _STATS[panel]


def _boosters(model_dir, threads):
    import lightgbm as lgb
    return {k: lgb.Booster(model_file=os.path.join(model_dir, f"task2_{k}.txt"), params={"num_threads": threads})
            for k in ("general", "general_w", "onsetT30")}


def dump(family, model_dirs, panels=Q2, splits=("validation", "private"), threads=4):
    """Members of one family share their features, so each window's features are built once and every member predicts
    from them. model_dirs: one directory or a list. Returns {model_dir: window_probs DataFrame}."""
    if isinstance(model_dirs, str): model_dirs = [model_dirs]
    ms = {d: _boosters(d, threads) for d in model_dirs}
    if family == "v15":
        import task2_v15 as v15
    if family == "v18":
        import task2_v18 as v18
        v18.TRAIN_BLANK["on"] = False
    rows = {d: [] for d in model_dirs}
    for split in splits:
        for p in panels:
            net, st, days, full = stats(p); li = {l: i for i, l in enumerate(net.link_id)}; M = len(net)
            wi = pd.read_csv(DATA / "task2" / p / split / "window_index.csv")
            hist = pd.read_parquet(DATA / "task2" / p / split / "window_history.parquet")
            mfiles = {date_of(f): f for f in day_files(p, split, masked=True)}
            for w in wi.itertuples(index=False):
                t0 = int(slots_of(pd.Series([w.forecast_origin]))[0]); onset = int(w.condition == "queue_onset"); wd = weekday_of(w.date)
                if family == "v15":
                    mg = day_grid(mfiles[w.date], li, M, with_occ=True); on, off = v15.ramp_day(p, split, w.date, net)
                    Hv, Hq, Ho = v10.load_window(hist, w, li, M, mg, t0)
                    X, _ = v15.feats15(Hv, Hq, Ho, v15.ext_window(mg, t0, M), on, off, net, full, t0, wd, onset)
                else:
                    mg = day_grid(mfiles[w.date], li, M, with_occ=True) if w.date in mfiles else None
                    Hv, Hq, Ho = v10.load_window(hist, w, li, M, mg, t0)
                    X, _ = v10.feats(Hv, Hq, Ho, net, full, t0, wd, onset)
                for d, m in ms.items():
                    pg = m["general"].predict(X.reshape(M * H, -1)).reshape(M, H)
                    pw = m["general_w"].predict(X.reshape(M * H, -1)).reshape(M, H)
                    p6 = m["onsetT30"].predict(X[:, H - 1]) if onset else np.full(M, np.nan)
                    for h in range(H):
                        rows[d].append(pd.DataFrame({"split": split, "panel": p, "window_id": w.window_id, "condition": w.condition, "h": h + 1,
                                                     "link_id": net.link_id.to_numpy(), "p_general": pg[:, h], "p_general_w": pw[:, h],
                                                     "p_onsetT30": p6 if h == H - 1 else np.nan}))
    return {d: pd.concat(r, ignore_index=True) for d, r in rows.items()}
