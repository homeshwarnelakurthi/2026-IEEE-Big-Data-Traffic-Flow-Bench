"""Per-cell February predictions of NN checkpoints on target cells inside simulated Task 2 blank blocks.

Same simulation as t1_blocks_eval3.py ("realistic"): March's validation window schedule copied onto February days,
T+5..T+90 blanked on every link, the hour T-55..T republished unmasked. Writes one row per in-block official target
cell (panel, day, slot, link_id, v_nn, q_nn); scored and blended against the gap-filler in t1_blocks_blend.py.
Usage: python t1_blocks_nn_dump.py <t1_nn.py> <out.csv> <ckpt.pt> [<ckpt.pt> ...]   (env TFB_DATA_ROOT)
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("t1nn", sys.argv[1]); nnm = importlib.util.module_from_spec(spec); spec.loader.exec_module(nnm)
import torch  # noqa: E402

torch.set_num_threads(os.cpu_count() or 4)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nets = []
for w in sys.argv[3:]:
    n = nnm.build_model().to(dev); n.load_state_dict(torch.load(w, map_location=dev)); n.eval(); nets.append(n)
out = []
for panel in nnm.panels():
    wpath = nnm.DATA / "task2" / panel / "validation" / "window_index.csv"
    if not wpath.exists():
        continue
    wi = pd.read_csv(wpath)
    pd_ = nnm.PanelData(panel); dates = list(pd_.trm["dates"]); links = pd_.m.link_id.to_numpy()
    feb = [i for i, d in enumerate(dates) if d.startswith("2031-02")]
    sched = {}
    for w in wi.itertuples():
        dom = int(w.date[8:10])
        if dom <= len(feb):
            sched.setdefault(feb[dom - 1], []).append((int(w.forecast_origin[11:13]) * 60 + int(w.forecast_origin[14:16])) // 5)
    t, d, s, j = nnm.template_mask(panel, "train", dates, pd_.m)
    for day, origins in sched.items():
        blk = np.zeros(288, bool); hist = np.zeros(288, bool)
        for o in origins:
            blk[o + 1:min(o + 19, 288)] = True; hist[max(o - 11, 0):o + 1] = True
        hist &= ~blk
        full_v, full_q, full_o = pd_.tr["speed"][day], pd_.tr["flow"][day], pd_.tr["occ"][day]
        v, q, o_ = pd_.trm["speed"][day].copy(), pd_.trm["flow"][day].copy(), pd_.trm["occ"][day].copy()
        v[blk] = q[blk] = o_[blk] = np.nan
        v[hist], q[hist], o_[hist] = full_v[hist], full_q[hist], full_o[hist]
        kin = (d == day) & blk[s]
        si, ji = s[kin], j[kin]
        outs = [nnm.run(n, pd_, v, q, o_, pd_.dows[day], pd_.prof["2031-02"], dev) for n in nets]
        nv, nq = np.mean([x[0] for x in outs], 0), np.mean([x[1] for x in outs], 0)
        out.append(pd.DataFrame({"panel": panel, "day": dates[day], "slot": si, "link_id": links[ji], "v_nn": nv[si, ji], "q_nn": nq[si, ji]}))
    print(panel, "days", len(sched), flush=True)
pd.concat(out).to_csv(sys.argv[2], index=False)
print("written", sum(len(x) for x in out))
