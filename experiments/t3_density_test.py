# LOCAL ANALYSIS ONLY: not used by run_all.py. Reads intermediate outputs of our Kaggle runs under $TFB_WORKSPACE (not in this repo).
"""Does re-deriving flow from a blended density estimate trade S_state for S_LWR profitably?
k_final = a*k_model + (1-a)*k_interp (time-interpolated observed density); flow := k_final * v_model."""
import numpy as np
import pandas as pd

from common import load
from t3_proxy import evaluate

WS = __import__("os").environ.get("TFB_WORKSPACE", ".")  # folder holding our Kaggle run outputs

cb = pd.read_csv(WS + r"/kaggle_kernels/t1_cat/out_v1/feb_state_catboost.csv",
                 usecols=["panel", "timestamp", "link_id", "speed_kmh", "flow_vph"], dtype={"link_id": str})
MK = {}


def make(a):
    def tf(p, pv, pq, tr, tgt, days):
        if p not in MK:
            MK[p] = load(p, "train", masked=True)
        mk = MK[p]
        pv2, pq2 = pv.copy(), pq.copy()
        for d in days:
            k_obs = mk["flow"][d] / np.maximum(mk["speed"][d], 1)
            k_int = pd.DataFrame(k_obs).interpolate(limit_direction="both", axis=0).to_numpy(np.float32)
            t = tgt[d]
            k_mod = pq[d] / np.maximum(pv[d], 1)
            k_fin = np.where(np.isfinite(k_int), a * k_mod + (1 - a) * k_int, k_mod)
            pq2[d] = np.where(t, k_fin * np.maximum(pv[d], 1), pq[d])
        return pv2, pq2
    return tf


for a in (0.8, 0.6, 0.4):
    st, lw = evaluate(cb, f"a={a}", make(a))
