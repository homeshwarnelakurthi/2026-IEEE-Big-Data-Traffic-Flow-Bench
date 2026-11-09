"""Task 1: exact target masks from the official templates and the official S_state scoring rule."""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import CACHE, RELEASE, family, load, meta, panels


def targets(panel: str, split: str) -> np.ndarray:
    """(D, 288, L) bool, True at template cells, aligned with the masked cache (milepost order)."""
    out = CACHE / panel / f"{split}_targets.npy"
    if out.exists():
        return np.load(out)
    z = load(panel, split, masked=True)
    dates = list(z["dates"])
    col = {l: i for i, l in enumerate(meta(panel).link_id)}
    t = pd.read_csv(RELEASE / "task1" / panel / split / "sample_submission_state.csv",
                    usecols=["timestamp", "link_id"], dtype=str)
    d = t.timestamp.str[:10].map({x: i for i, x in enumerate(dates)}).to_numpy()
    s = ((t.timestamp.str[11:13].astype(int) * 60 + t.timestamp.str[14:16].astype(int)) // 5).to_numpy()
    j = t.link_id.map(col).to_numpy()
    m = np.zeros(z["speed"].shape, bool)
    m[d, s, j] = True
    np.save(out, m)
    return m


def score_panel(panel: str, pv: np.ndarray, pq: np.ndarray, truth: dict, tgt: np.ndarray, days: np.ndarray) -> dict:
    """S_state for one panel on the given day indices. pv/pq/truth arrays are (D, 288, L)."""
    lanes = meta(panel).lanes.to_numpy(np.float32)
    reg = truth["regimes"]
    res = {}
    for r in ("R1", "R2", "R3"):
        dd = days[reg[days] == r]
        if len(dd) == 0:
            continue
        m = tgt[dd]
        ev = (pv[dd] - truth["speed"][dd])[m]
        eq = ((pq[dd] - truth["flow"][dd]) / lanes)[m]
        rv, rq = np.sqrt(np.mean(ev ** 2)), np.sqrt(np.mean(eq ** 2))
        res[r] = dict(rmse_v=rv, rmse_q=rq, S=0.54 * max(0, 1 - rv / 25) + 0.46 * max(0, 1 - rq / 600), n=int(m.sum()))
    res["S_state"] = float(np.mean([v["S"] for k, v in res.items()]))
    return res


def family_mean(per_panel: dict) -> float:
    fam = {}
    for p, s in per_panel.items():
        fam.setdefault(family(p), []).append(s)
    return float(np.mean([np.mean(v) for v in fam.values()]))


def train_truth(panel: str) -> dict:
    """Unmasked train observations with the per-day regime taken from the masked view."""
    full = load(panel, "train", masked=False)
    masked = load(panel, "train", masked=True)
    assert (full["dates"] == masked["dates"]).all()
    return dict(speed=full["speed"], flow=full["flow"], regimes=masked["regimes"], dates=full["dates"])
