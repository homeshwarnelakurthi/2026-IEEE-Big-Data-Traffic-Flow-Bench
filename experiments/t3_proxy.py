# LOCAL ANALYSIS ONLY: not used by run_all.py. Reads intermediate outputs of our Kaggle runs under $TFB_WORKSPACE (not in this repo).
"""Local Task 3 proxy on February.

The organizer boundary flows are projected onto the released observations, so the observed state satisfies
conservation exactly: rhs = N_obs(t+dt) - N_obs(t). A submission keeps the observations everywhere except
at the Task 1 targets, so the residual is |dN_sub - dN_obs| and
    S_LWR = 1 - sum|dN_sub - dN_obs| / sum|dN_obs|
over 5-minute transitions between eligible cells. (Ramp-validity and observability filters are ignored.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import family, meta, panels
from t1_common import score_panel, targets, train_truth

WS = __import__("os").environ.get("TFB_WORKSPACE", ".")  # folder holding our Kaggle run outputs

FEB = "2031-02"


def lwr_panel(panel, pv, pq, tr, tgt, days):
    """S_LWR proxy per regime and its mean. pv/pq hold predictions at target cells (D, 288, L)."""
    m = meta(panel)
    length = m.length_km.to_numpy(np.float32)
    elig = np.isfinite(tr["speed"]) & np.isfinite(tr["flow"])
    out = {}
    for r in ("R1", "R2", "R3"):
        dd = days[tr["regimes"][days] == r]
        if len(dd) == 0:
            continue
        v_obs, q_obs = tr["speed"][dd], tr["flow"][dd]
        t_ = tgt[dd]
        v_sub = np.where(t_, pv[dd], v_obs); q_sub = np.where(t_, pq[dd], q_obs)
        ok = elig[dd]
        N_obs = q_obs / np.maximum(v_obs, 1) * length
        N_sub = q_sub / np.maximum(v_sub, 1) * length
        both = ok[:, 1:] & ok[:, :-1]
        d_obs = np.diff(N_obs, axis=1)[both]; d_sub = np.diff(N_sub, axis=1)[both]
        out[r] = 1 - np.abs(d_sub - d_obs).sum() / np.abs(d_obs).sum()
    return float(np.mean(list(out.values()))), out


def evaluate(pred_frame: pd.DataFrame | None, label: str, transform=None):
    """pred_frame: panel,timestamp,link_id,speed_kmh,flow_vph on February targets (None -> truth itself)."""
    st, lw = {}, {}
    for p in panels():
        tr = train_truth(p); tgt = targets(p, "train")
        days = np.where(np.char.startswith(tr["dates"].astype(str), FEB))[0]
        pv = np.full(tr["speed"].shape, np.nan, np.float32); pq = pv.copy()
        if pred_frame is None:
            pv, pq = tr["speed"], tr["flow"]
        else:
            f = pred_frame[pred_frame.panel == p]
            dates = {d: i for i, d in enumerate(tr["dates"])}
            col = {l: i for i, l in enumerate(meta(p).link_id)}
            d = f.timestamp.str[:10].map(dates).to_numpy()
            s = ((f.timestamp.str[11:13].astype(int) * 60 + f.timestamp.str[14:16].astype(int)) // 5).to_numpy()
            j = f.link_id.map(col).to_numpy()
            pv[d, s, j] = f.speed_kmh.to_numpy(); pq[d, s, j] = f.flow_vph.to_numpy()
        if transform is not None:
            pv, pq = transform(p, pv, pq, tr, tgt, days)
        st[p] = score_panel(p, pv, pq, tr, tgt, days)["S_state"]
        lw[p], per = lwr_panel(p, pv, pq, tr, tgt, days)
    fam = lambda dct: np.mean([np.mean([v for k, v in dct.items() if family(k) == f]) for f in {family(k) for k in dct}])
    print(f"{label}: S_state {fam(st):.5f}  S_LWR_proxy {fam(lw):.5f}  " +
          " ".join(f"{k}:{v:.3f}" for k, v in lw.items()), flush=True)
    return st, lw


if __name__ == "__main__":
    import sys
    evaluate(None, "truth (sanity, expect 1.0)")
    cb = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else WS + r"/kaggle_kernels/t1_cat/out_v1/feb_state_catboost.csv",
                     usecols=["panel", "timestamp", "link_id", "speed_kmh", "flow_vph"], dtype={"link_id": str})
    evaluate(cb, "catboost v1")
