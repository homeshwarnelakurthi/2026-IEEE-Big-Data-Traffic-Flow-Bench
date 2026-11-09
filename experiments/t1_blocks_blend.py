"""Score w * NN + (1 - w) * gap-filler on February target cells inside simulated Task 2 blank blocks.

Inputs: the per-cell dumps of t1_gapfill.py eval (TFB_GF_DUMP) and t1_blocks_nn_dump.py, joined on
(panel, day, slot, link_id). Pooled RMSE in score units, as in the IN-BLOCK lines of the eval scripts.
Usage: python t1_blocks_blend.py <gapfill_dump.csv> <nn_dump.csv>
"""
import sys

import numpy as np
import pandas as pd

from common import meta

gf = pd.read_csv(sys.argv[1])
gf["link_id"] = [str(x) for x in np.concatenate([meta(p).link_id.to_numpy()[g.j.to_numpy()] for p, g in gf.groupby("panel", sort=False)])]
nn = pd.read_csv(sys.argv[2], dtype={"link_id": str})
m = gf.merge(nn, on=["panel", "day", "slot", "link_id"], how="inner")
print(f"cells: gap-filler {len(gf)}, nn {len(nn)}, joined {len(m)}")


def score(v, q):
    rv = np.sqrt(((v - m.v_true) ** 2).mean()); rq = np.sqrt((((q - m.q_true) / m.lanes) ** 2).mean())
    return rv, rq, 0.54 * (1 - rv / 25) + 0.46 * (1 - rq / 600)


print("   w_nn  speed RMSE  flow/lane RMSE       S")
for w in np.round(np.arange(0, 1.01, 0.1), 1):
    rv, rq, s = score(w * m.v_nn + (1 - w) * m.v_gf, w * m.q_nn + (1 - w) * m.q_gf)
    print(f"{w:7.1f} {rv:11.3f} {rq:15.2f} {s:8.4f}")
best = max(((wv, wq) for wv in np.arange(0, 1.01, 0.05) for wq in np.arange(0, 1.01, 0.05)),
           key=lambda t: score(t[0] * m.v_nn + (1 - t[0]) * m.v_gf, t[1] * m.q_nn + (1 - t[1]) * m.q_gf)[2])
rv, rq, s = score(best[0] * m.v_nn + (1 - best[0]) * m.v_gf, best[1] * m.q_nn + (1 - best[1]) * m.q_gf)
print(f"separate weights: w_speed {best[0]:.2f}  w_flow {best[1]:.2f} -> {rv:.3f} / {rq:.2f} / S {s:.4f}")
