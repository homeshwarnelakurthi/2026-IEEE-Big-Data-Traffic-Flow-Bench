# LOCAL ANALYSIS ONLY: not used by run_all.py. Reads intermediate outputs of our Kaggle runs under $TFB_WORKSPACE (not in this repo).
"""February: blend in density space for Task 3.

S_LWR scores the accumulation N = (q/v) L at the target cells, so a blend that is good for speed and flow separately
need not give the best density. Variants on the normal-day recipe (0.9 x NN 6 seeds + 0.1 x CatBoost):
  base      v, q blended separately (current)
  kq        k blended (arithmetic over models), q as base, v = q/k
  kv        k blended, v as base, q = k v
  hv        v from the harmonic mean of member speeds (= density-consistent for equal flows)
"""
import numpy as np
import pandas as pd

from common import load, meta, panels
from t2_data import downstream_order
from t3_proxy import evaluate

WS = __import__("os").environ.get("TFB_WORKSPACE", ".")  # folder holding our Kaggle run outputs

K = WS + r"/kaggle_kernels"
KEY = ["panel", "timestamp", "link_id"]
cb = pd.read_csv(K + r"/t1_cat/out_v2/feb_state_catboost.csv", usecols=KEY + ["speed_kmh", "flow_vph"], dtype={"link_id": str})


def member(path):
    z = np.load(path); parts = []
    for p in panels():
        dates = load(p, "train", masked=True)["dates"]; links = meta(p).link_id.to_numpy()[downstream_order(p)]
        d, s, j = z[f"{p}_d"], z[f"{p}_s"], z[f"{p}_j"]
        ts = pd.Series(dates[d]).str.cat(pd.Series([f"T{x//12:02d}:{(x%12)*5:02d}:00Z" for x in s]))
        parts.append(pd.DataFrame({"panel": p, "timestamp": ts, "link_id": links[j], "v": z[f"{p}_v"], "q": z[f"{p}_q"]}))
    n = cb[KEY].merge(pd.concat(parts, ignore_index=True), on=KEY, how="left")
    assert n.v.notna().all()
    return n.v.to_numpy(np.float64), n.q.to_numpy(np.float64)


files = [r"/t1_nn/out_v1/feb_pred_seed1.npz", r"/t1_nn/out_v1/feb_pred_seed2.npz", r"/t1_nn/out/feb_pred_seed1.npz",
         r"/t1_nn/out/feb_pred_seed2.npz", r"/t1_nn_b/out/feb_pred_seed3.npz", r"/t1_nn_b/out/feb_pred_seed4.npz"]
M = [member(K + f) for f in files]
vs = np.stack([m[0] for m in M]); qs = np.stack([m[1] for m in M])
vcb, qcb = cb.speed_kmh.to_numpy(np.float64), cb.flow_vph.to_numpy(np.float64)
v_base = 0.9 * vs.mean(0) + 0.1 * vcb; q_base = 0.9 * qs.mean(0) + 0.1 * qcb
k_nn = (qs / np.maximum(vs, 1)).mean(0); k_blend = 0.9 * k_nn + 0.1 * qcb / np.maximum(vcb, 1)


def frame(v, q):
    f = cb[KEY].copy(); f["speed_kmh"] = v; f["flow_vph"] = q; return f


evaluate(frame(v_base, q_base), "base (v, q separately)")
evaluate(frame(q_base / np.maximum(k_blend, 1e-6), q_base), "kq: k blended, q base, v=q/k")
evaluate(frame(v_base, k_blend * v_base), "kv: k blended, v base, q=k v")
vh = 0.9 / (1 / np.maximum(vs, 1)).mean(0) + 0.1 * vcb
evaluate(frame(vh, q_base), "hv: harmonic NN speed")
for i in (0, 2):
    evaluate(frame(0.9 * vs[i] + 0.1 * vcb, 0.9 * qs[i] + 0.1 * qcb), f"single member {i}")
