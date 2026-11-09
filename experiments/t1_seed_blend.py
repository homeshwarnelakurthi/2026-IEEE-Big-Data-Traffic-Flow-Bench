# LOCAL ANALYSIS ONLY: not used by run_all.py. Reads intermediate outputs of our Kaggle runs under $TFB_WORKSPACE (not in this repo).
"""February S_state of the normal-day Task 1 blend for different NN seed sets and NN/CatBoost weights.
Usage: python t1_seed_blend.py <extra v2 feb_pred_seed*.npz ...>
Baseline = run19 normal days: 0.9 * mean(v1 seeds 1-2, v2 seeds 1-4, equal per seed) + 0.1 * CatBoost.
"""
import sys

import numpy as np
import pandas as pd

from common import load, meta, panels
from t2_data import downstream_order
from t3_proxy import evaluate

WS = __import__("os").environ.get("TFB_WORKSPACE", ".")  # folder holding our Kaggle run outputs

K = WS + r"/kaggle_kernels"
key = ["panel", "timestamp", "link_id"]
cb = pd.read_csv(K + r"/t1_cat/out_v2/feb_state_catboost.csv", usecols=key + ["speed_kmh", "flow_vph"], dtype={"link_id": str})


def nn_frame(path):  # same as t1_blend.nn_frame (that module runs its analysis on import)
    z = np.load(path)
    parts = []
    for p in panels():
        dates = load(p, "train", masked=True)["dates"]
        links = meta(p).link_id.to_numpy()[downstream_order(p)]
        d, s, j = z[f"{p}_d"], z[f"{p}_s"], z[f"{p}_j"]
        ts = pd.Series(dates[d]).str.cat(pd.Series([f"T{x//12:02d}:{(x%12)*5:02d}:00Z" for x in s]))
        parts.append(pd.DataFrame({"panel": p, "timestamp": ts, "link_id": links[j], "speed_kmh": z[f"{p}_v"], "flow_vph": z[f"{p}_q"]}))
    return pd.concat(parts, ignore_index=True)

v1 = [K + r"/t1_nn/out_v1/feb_pred_seed1.npz", K + r"/t1_nn/out_v1/feb_pred_seed2.npz"]
v2 = [K + r"/t1_nn/out/feb_pred_seed1.npz", K + r"/t1_nn/out/feb_pred_seed2.npz",
      K + r"/t1_nn_b/out/feb_pred_seed3.npz", K + r"/t1_nn_b/out/feb_pred_seed4.npz"]
extra = sys.argv[1:]


def stack(paths):
    fr = [nn_frame(p).sort_values(key).reset_index(drop=True) for p in paths]
    return np.stack([f.speed_kmh.to_numpy() for f in fr]), np.stack([f.flow_vph.to_numpy() for f in fr]), fr[0][key]


V1v, V1q, K0 = stack(v1); V2v, V2q, _ = stack(v2)
EXv, EXq = stack(extra)[:2] if extra else (np.zeros((0, len(K0))),) * 2
c = K0.merge(cb, on=key, how="left", validate="one_to_one")
assert c.speed_kmh.notna().all()
cbv, cbq = c.speed_kmh.to_numpy(), c.flow_vph.to_numpy()


def run(name, vs, qs, w_nn=0.9):
    nv, nq = np.concatenate(vs).mean(0), np.concatenate(qs).mean(0)
    evaluate(pd.DataFrame({**{k: K0[k] for k in key}, "speed_kmh": w_nn * nv + (1 - w_nn) * cbv, "flow_vph": w_nn * nq + (1 - w_nn) * cbq}), name)


run("base: v1x2 + v2x4, 0.9 nn", [V1v, V2v], [V1q, V2q])
if extra:
    for i in range(len(extra)):
        evaluate(pd.DataFrame({**{k: K0[k] for k in key}, "speed_kmh": EXv[i], "flow_vph": EXq[i]}), "new seed alone " + extra[i][-10:])
    run(f"v1x2 + v2x{4 + len(extra)}, 0.9 nn", [V1v, V2v, EXv], [V1q, V2q, EXq])
    run(f"v2x{4 + len(extra)} only, 0.9 nn", [V2v, EXv], [V2q, EXq])
    for w in (0.85, 0.95, 1.0):
        run(f"v1x2 + v2x{4 + len(extra)}, {w} nn", [V1v, V2v, EXv], [V1q, V2q, EXq], w)
