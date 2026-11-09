# LOCAL ANALYSIS ONLY: not used by run_all.py. Reads intermediate outputs of our Kaggle runs under $TFB_WORKSPACE (not in this repo).
"""Combine Task 1 predictions per day: NN seed average (90%) + CatBoost (10%) on normal days, CatBoost alone on days
that carry Task 2 windows (their 90-minute all-link blanks still degrade the NN). Writes cache/state_{split}_nnfix.csv."""
import glob

import pandas as pd

from common import CACHE, RELEASE

WS = __import__("os").environ.get("TFB_WORKSPACE", ".")  # folder holding our Kaggle run outputs

K = WS + r"/kaggle_kernels"
NN = [rf"{K}\t1_nn\out_v1\state_{{split}}_seed1.csv", rf"{K}\t1_nn\out_v1\state_{{split}}_seed2.csv",
      rf"{K}\t1_nn\out\state_{{split}}_seed1.csv", rf"{K}\t1_nn\out\state_{{split}}_seed2.csv",
      rf"{K}\t1_nn_b\out\state_{{split}}_seed3.csv", rf"{K}\t1_nn_b\out\state_{{split}}_seed4.csv"]
key = ["panel", "timestamp", "station_id", "link_id", "mask_regime"]
wi = pd.concat([pd.read_csv(f) for f in glob.glob(str(RELEASE / "task2" / "*" / "*" / "window_index.csv"))])
t2days = set(zip(wi.panel, wi.date))
rd = lambda f: pd.read_csv(f, usecols=key + ["speed_kmh", "flow_vph"], dtype={"station_id": str, "link_id": str}).set_index(key)
for split in ("validation", "private"):
    cb = rd(rf"{K}\t1_cat\out_v2\state_{split}_catboost.csv")
    nn = sum(rd(f.format(split=split)).reindex(cb.index) for f in NN) / len(NN)
    assert nn.notna().all().all()
    out = 0.9 * nn + 0.1 * cb
    bad = pd.Series(list(zip(cb.index.get_level_values("panel"), cb.index.get_level_values("timestamp").str[:10]))).isin(t2days).to_numpy()
    out[bad] = cb[bad]
    out.reset_index().to_csv(CACHE / f"state_{split}_nnfix.csv", index=False)
    print(split, "rows", len(out), "NN seeds", len(NN), "CatBoost-only share (Task-2 days)", round(bad.mean(), 3))
