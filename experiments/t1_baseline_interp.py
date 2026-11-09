"""Sanity baseline for the Task 1 harness: time interpolation, spatial fallback, February score."""
import numpy as np
import pandas as pd

from common import load, panels
from t1_common import family_mean, score_panel, targets, train_truth


def interp_time(a: np.ndarray) -> np.ndarray:
    """Linear interpolation along the slot axis of a (288, L) array, edge-filled."""
    df = pd.DataFrame(a)
    return df.interpolate(limit_direction="both", axis=0).to_numpy(np.float32)


def interp_space(a: np.ndarray) -> np.ndarray:
    return pd.DataFrame(a.T).interpolate(limit_direction="both", axis=0).to_numpy(np.float32).T


per = {}
for p in panels():
    tr = train_truth(p); mk = load(p, "train", masked=True); tgt = targets(p, "train")
    days = np.where(np.char.startswith(tr["dates"].astype(str), "2031-02"))[0]
    pv = np.full_like(mk["speed"], np.nan); pq = np.full_like(mk["flow"], np.nan)
    for d in days:
        for src, dst in ((mk["speed"], pv), (mk["flow"], pq)):
            t_ = interp_time(src[d]); s_ = interp_space(src[d])
            dst[d] = np.where(np.isfinite(t_), 0.85 * t_ + 0.15 * np.nan_to_num(s_, nan=0) + 0.15 * np.isnan(s_) * np.nan_to_num(t_), s_)
    r = score_panel(p, pv, pq, tr, tgt, days)
    per[p] = r["S_state"]
    print(p, round(r["S_state"], 4), {k: (round(v["rmse_v"], 2), round(v["rmse_q"], 1)) for k, v in r.items() if k != "S_state"}, flush=True)
print("family mean S_state (Feb):", round(family_mean(per), 4))
