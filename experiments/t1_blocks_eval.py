# LOCAL ANALYSIS ONLY: not used by run_all.py. Reads intermediate outputs of our Kaggle runs under $TFB_WORKSPACE (not in this repo).
"""Which Task 1 model to trust on days that carry Task 2 windows?

Validation/private days with Task 2 windows blank T+5 .. T+90 (horizon + one hour) on every link. February has none,
so we copy the March window schedule onto February (day-of-month i -> Feb day i), blank those 18-slot blocks in the
masked input, and score NN v1, NN v2 and CatBoost against the truth on the official targets outside the blocks.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pandas as pd

WS = __import__("os").environ.get("TFB_WORKSPACE", ".")  # folder holding our Kaggle run outputs

K = WS + r"/kaggle_kernels"
os.environ.update(TFB_DATA_ROOT=WS + r"/data/kaggle_public", TFB_OUT=WS + r"/cache/t1nn_local",
                  TFB_OUTPUT_ROOT=WS + r"/cache/cat_local", TFB_THREADS="12")
spec = importlib.util.spec_from_file_location("t1nn", K + r"/t1_nn/t1_nn.py"); nnm = importlib.util.module_from_spec(spec); spec.loader.exec_module(nnm)
import torch  # noqa: E402

torch.set_num_threads(12)
sys.path.insert(0, WS + r"/public_outputs/catboost/tfb_source")
import state_model as base  # noqa: E402
from prepare_state import features  # noqa: E402
from state_inference import StatePredictor  # noqa: E402

DATA = nnm.DATA
NETS = {"nn_v1": [K + r"/t1_nn/out_v1/t1nn_seed1.pt", K + r"/t1_nn/out_v1/t1nn_seed2.pt"],
        "nn_v2": [K + r"/t1_nn/out/t1nn_seed1.pt", K + r"/t1_nn/out/t1nn_seed2.pt",
                  K + r"/t1_nn_b/out/t1nn_seed3.pt", K + r"/t1_nn_b/out/t1nn_seed4.pt"]}
dev = torch.device("cpu")
nets = {}
for k, paths in NETS.items():
    nets[k] = []
    for p in paths:
        n = nnm.build_model(); n.load_state_dict(torch.load(p, map_location=dev)); n.eval(); nets[k].append(n)
cat = StatePredictor(K + r"/t1_cat/out_v2")
os.environ["TFB_PREDICT_DEVICE"] = "CPU"; cat.task_type = "CPU"

rows = []
for pi, panel in enumerate(nnm.panels()):
    wpath = DATA / "task2" / panel / "validation" / "window_index.csv"
    if not wpath.exists():  # D12_I405_N/S have no Task 2 windows
        continue
    wi = pd.read_csv(wpath)
    pd_ = nnm.PanelData(panel)
    dates = list(pd_.trm["dates"])
    feb = [i for i, d in enumerate(dates) if d.startswith("2031-02")]
    blocks = {}  # Feb day index -> list of blocked slot ranges
    for w in wi.itertuples():
        dom = int(w.date[8:10])
        if dom <= len(feb):
            t = (int(w.forecast_origin[11:13]) * 60 + int(w.forecast_origin[14:16])) // 5
            blocks.setdefault(feb[dom - 1], []).append((t + 1, min(t + 19, 288)))
    t, d, s, j = nnm.template_mask(panel, "train", dates, pd_.m)
    lanes = pd_.lanes
    for day, rngs in blocks.items():
        blk = np.zeros(288, bool)
        for a, b in rngs:
            blk[a:b] = True
        v = pd_.trm["speed"][day].copy(); q = pd_.trm["flow"][day].copy(); o = pd_.trm["occ"][day].copy()
        v[blk] = np.nan; q[blk] = np.nan; o[blk] = np.nan
        k = (d == day) & ~blk[s]
        ss, jj = s[k], j[k]
        tv, tq = pd_.tr["speed"][day][ss, jj], pd_.tr["flow"][day][ss, jj]
        near = np.zeros(288, bool)
        for a, b in rngs:
            near[max(a - 12, 0):a] = True; near[b:min(b + 12, 288)] = True
        preds = {}
        for name, ns in nets.items():
            outs = [nnm.run(n, pd_, v, q, o, pd_.dows[day], pd_.prof["2031-02"], dev) for n in ns]
            preds[name] = (np.mean([x[0] for x in outs], 0)[ss, jj], np.mean([x[1] for x in outs], 0)[ss, jj])
        # CatBoost through the public feature pipeline (links in fd order, timestamps as text)
        cp = base.Panel(panel, pi)
        frame_links = pd_.m.link_id.to_numpy()
        ts = [f"{dates[day]}T{x // 12:02d}:{(x % 12) * 5:02d}:00Z" for x in range(288)]
        fr = pd.DataFrame({"timestamp": np.repeat(ts, len(frame_links)), "link_id": np.tile(frame_links, 288),
                           "speed_kmh": v.ravel(), "flow_vph": q.ravel(),
                           "is_score_eligible": pd_.trm["elig"][day].ravel().astype(int)})
        times, V, Q, E = base.matrices(fr, cp)
        rows_ = times.get_indexer([ts[x] for x in ss]); cols_ = pd.Index(cp.links).get_indexer(frame_links[jj])
        reg = {"R1": 0, "R2": 1, "R3": 2}[pd_.trm["regimes"][day]]
        X, bv, bq = features(cp, pi, times, V, Q, E, rows_, cols_, np.full(len(ss), reg))
        preds["catboost"] = (np.clip(bv + cat.residual(X, "speed"), 1, 180), np.clip((bq + cat.residual(X, "flow")) * cp.lanes[cols_], 0, None))
        for name, (pv, pq) in preds.items():
            for part, m in (("all", np.ones(len(ss), bool)), ("near_block", near[ss])):
                rows.append(dict(panel=panel, day=dates[day], model=name, part=part, n=int(m.sum()),
                                 se_v=float(((pv - tv)[m] ** 2).sum()), se_q=float((((pq - tq) / lanes[jj])[m] ** 2).sum())))
    print(panel, "blocked Feb days:", len(blocks), flush=True)

r = pd.DataFrame(rows)
g = r.groupby(["model", "part"])[["n", "se_v", "se_q"]].sum()
g["rmse_v"] = np.sqrt(g.se_v / g.n); g["rmse_q_lane"] = np.sqrt(g.se_q / g.n)
g["S_approx"] = 0.54 * (1 - g.rmse_v / 25) + 0.46 * (1 - g.rmse_q_lane / 600)
print(g[["n", "rmse_v", "rmse_q_lane", "S_approx"]].round(4).to_string())
r.to_csv(WS + r"/cache/t1_blocks_eval.csv", index=False)
