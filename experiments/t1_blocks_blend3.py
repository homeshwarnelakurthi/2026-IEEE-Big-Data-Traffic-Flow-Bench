"""Score the CTM fill against NN v3 and the gap-filler on February in-block cells, and three-way blends of them.

Inputs: t1_gapfill.py eval dump (TFB_GF_DUMP), t1_blocks_nn_dump.py dump, t1_ctm_gap.py eval dump; joined on
(panel, day, slot, link). Blend weights are searched on a 0.1 grid, separately for speed and flow.
Usage: python t1_blocks_blend3.py <gapfill.csv> <nn_v3.csv> <ctm.csv>
"""
import itertools
import sys

import numpy as np
import pandas as pd

from common import meta

gf = pd.read_csv(sys.argv[1]); ctm = pd.read_csv(sys.argv[3]); nn = pd.read_csv(sys.argv[2], dtype={"link_id": str})
jmap = pd.concat([pd.DataFrame({"panel": p, "link_id": meta(p).link_id.astype(str), "j": np.arange(len(meta(p)))}) for p in nn.panel.unique()])
nn = nn.merge(jmap, on=["panel", "link_id"])
k = ["panel", "day", "slot", "j"]
m = gf.merge(nn[k + ["v_nn", "q_nn"]], on=k).merge(ctm[k + ["v_ctm", "q_ctm"]], on=k)
print(f"cells: gap-filler {len(gf)}, nn {len(nn)}, ctm {len(ctm)}, joined {len(m)}")
P = {"gf": ("v_gf", "q_gf"), "v3": ("v_nn", "q_nn"), "ctm": ("v_ctm", "q_ctm")}
rmse_v = lambda v: np.sqrt(((v - m.v_true) ** 2).mean())
rmse_q = lambda q: np.sqrt((((q - m.q_true) / m.lanes) ** 2).mean())
S = lambda rv, rq: 0.54 * (1 - rv / 25) + 0.46 * (1 - rq / 600)
for name, (cv, cq) in P.items():
    rv, rq = rmse_v(m[cv]), rmse_q(m[cq]); print(f"{name:4s} speed {rv:6.3f}  flow/lane {rq:6.2f}  S {S(rv, rq):.4f}")
print("run14 (0.5 v3 + 0.5 gf):", f"S {S(rmse_v(0.5 * m.v_nn + 0.5 * m.v_gf), rmse_q(0.5 * m.q_nn + 0.5 * m.q_gf)):.4f}")
grid = [(a, b, round(1 - a - b, 1)) for a, b in itertools.product(np.round(np.arange(0, 1.01, 0.1), 1), repeat=2) if a + b <= 1.0001]
bv = min(grid, key=lambda g: rmse_v(g[0] * m.v_gf + g[1] * m.v_nn + g[2] * m.v_ctm))
bq = min(grid, key=lambda g: rmse_q(g[0] * m.q_gf + g[1] * m.q_nn + g[2] * m.q_ctm))
rv = rmse_v(bv[0] * m.v_gf + bv[1] * m.v_nn + bv[2] * m.v_ctm); rq = rmse_q(bq[0] * m.q_gf + bq[1] * m.q_nn + bq[2] * m.q_ctm)
print(f"best blend  speed (gf, v3, ctm) {bv} -> {rv:.3f}   flow {bq} -> {rq:.2f}   S {S(rv, rq):.4f}")
# where does the CTM help: congested vs free-flow truth (speed below 0.6 free speed counts as congested here)
vf = pd.concat([pd.DataFrame({"panel": p, "j": np.arange(len(meta(p))), "vf": meta(p).free_speed_kmh.to_numpy()}) for p in m.panel.unique()])
m = m.merge(vf, on=["panel", "j"]); cong = m.v_true <= 0.6 * m.vf
for lab, sel in (("congested", cong), ("free", ~cong)):
    mm = m[sel]
    print(lab, f"share {sel.mean():.3f}", " ".join(f"{n}:{np.sqrt(((mm[c[0]] - mm.v_true) ** 2).mean()):.2f}" for n, c in P.items()))
