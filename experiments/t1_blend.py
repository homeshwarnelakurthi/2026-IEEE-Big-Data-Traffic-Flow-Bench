# LOCAL ANALYSIS ONLY: not used by run_all.py. Reads intermediate outputs of our Kaggle runs under $TFB_WORKSPACE (not in this repo).
"""Task 1 blend on February: CatBoost v2, NN seeds, and weighted blends, scored for S_state and the S_LWR proxy."""
import numpy as np
import pandas as pd

from common import meta, panels
from t2_data import downstream_order
from t3_proxy import evaluate

WS = __import__("os").environ.get("TFB_WORKSPACE", ".")  # folder holding our Kaggle run outputs

K = WS + r"/kaggle_kernels"


def nn_frame(path):
    z = np.load(path)
    parts = []
    for p in panels():
        from common import load
        dates = load(p, "train", masked=True)["dates"]
        links = meta(p).link_id.to_numpy()[downstream_order(p)]
        d, s, j = z[f"{p}_d"], z[f"{p}_s"], z[f"{p}_j"]
        ts = pd.Series(dates[d]).str.cat(pd.Series([f"T{x//12:02d}:{(x%12)*5:02d}:00Z" for x in s]))
        parts.append(pd.DataFrame({"panel": p, "timestamp": ts, "link_id": links[j], "speed_kmh": z[f"{p}_v"], "flow_vph": z[f"{p}_q"]}))
    return pd.concat(parts, ignore_index=True)


key = ["panel", "timestamp", "link_id"]
cb = pd.read_csv(K + r"/t1_cat/out_v2/feb_state_catboost.csv", usecols=key + ["speed_kmh", "flow_vph"], dtype={"link_id": str})
n1, n2 = nn_frame(K + r"/t1_nn/out_v1/feb_pred_seed1.npz"), nn_frame(K + r"/t1_nn/out_v1/feb_pred_seed2.npz")
m = cb.merge(n1, on=key, suffixes=("_cb", "_n1")).merge(n2.rename(columns={"speed_kmh": "speed_kmh_n2", "flow_vph": "flow_vph_n2"}), on=key)
assert len(m) == len(cb) == len(n1), (len(m), len(cb), len(n1))
m["v_nn"] = (m.speed_kmh_n1 + m.speed_kmh_n2) / 2; m["q_nn"] = (m.flow_vph_n1 + m.flow_vph_n2) / 2


def frame(wv, wq):
    return pd.DataFrame({**{k: m[k] for k in key}, "speed_kmh": wv * m.v_nn + (1 - wv) * m.speed_kmh_cb,
                         "flow_vph": wq * m.q_nn + (1 - wq) * m.flow_vph_cb})


evaluate(cb.copy(), "catboost v2")
evaluate(n1, "nn seed1")
evaluate(frame(1, 1), "nn 2-seed avg")
for w in (0.9, 0.8, 0.7, 0.6):
    evaluate(frame(w, w), f"blend nn={w}")
