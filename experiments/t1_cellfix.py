# LOCAL ANALYSIS ONLY: not used by run_all.py. Reads intermediate outputs of our Kaggle runs under $TFB_WORKSPACE (not in this repo).
"""Task 1 per cell: CatBoost only on target cells inside Task 2 blank blocks (T+5..T+90 on window days, where no link
reports anything); elsewhere 0.9 * NN seed average + 0.1 * CatBoost (6 seeds on normal days, NN v2's 4 seeds on
Task-2 days, since v1 never saw the blanks). Writes cache/state_{split}_cellfix.csv."""
import glob

import numpy as np
import pandas as pd

from common import CACHE, RELEASE

WS = __import__("os").environ.get("TFB_WORKSPACE", ".")  # folder holding our Kaggle run outputs

K = WS + r"/kaggle_kernels"
V1 = [rf"{K}\t1_nn\out_v1\state_{{s}}_seed1.csv", rf"{K}\t1_nn\out_v1\state_{{s}}_seed2.csv"]
V2 = [rf"{K}\t1_nn\out\state_{{s}}_seed1.csv", rf"{K}\t1_nn\out\state_{{s}}_seed2.csv",
      rf"{K}\t1_nn_b\out\state_{{s}}_seed3.csv", rf"{K}\t1_nn_b\out\state_{{s}}_seed4.csv"]
key = ["panel", "timestamp", "station_id", "link_id", "mask_regime"]
rd = lambda f: pd.read_csv(f, usecols=key + ["speed_kmh", "flow_vph"], dtype={"station_id": str, "link_id": str}).set_index(key)
wi = pd.concat([pd.read_csv(f) for f in glob.glob(str(RELEASE / "task2" / "*" / "*" / "window_index.csv"))])
for split in ("validation", "private"):
    cb = rd(rf"{K}\t1_cat\out_v2\state_{split}_catboost.csv")
    v1 = sum(rd(f.format(s=split)).reindex(cb.index) for f in V1) / len(V1)
    v2 = sum(rd(f.format(s=split)).reindex(cb.index) for f in V2) / len(V2)
    panel = cb.index.get_level_values("panel"); ts = pd.Series(cb.index.get_level_values("timestamp"))
    day = ts.str[:10].to_numpy(); slot = (ts.str[11:13].astype(int) * 12 + ts.str[14:16].astype(int) // 5).to_numpy()
    t2day = pd.Series(list(zip(panel, day))).isin(set(zip(wi.panel, wi.date))).to_numpy()
    inblk = np.zeros(len(cb), bool)
    for w in wi[wi.split == split].itertuples():
        o = int(w.forecast_origin[11:13]) * 12 + int(w.forecast_origin[14:16]) // 5
        inblk |= (panel == w.panel) & (day == w.date) & (slot > o) & (slot <= o + 18)
    nn = (2 * v1 + 4 * v2) / 6
    nn[t2day] = v2[t2day]
    out = 0.9 * nn + 0.1 * cb
    out[inblk] = cb[inblk]
    out.reset_index().to_csv(CACHE / f"state_{split}_cellfix.csv", index=False)
    print(split, f"in-block targets (CatBoost): {inblk.mean():.2%}; Task-2-day targets now on NN v2: {(t2day & ~inblk).mean():.2%}")
