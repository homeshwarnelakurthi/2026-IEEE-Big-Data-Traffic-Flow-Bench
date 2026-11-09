"""Realistic February simulation of validation/private days that carry Task 2 windows, scored against the truth.

On those days the release (a) blanks T+5 .. T+90 on every link and (b) leaves the hour T-55 .. T free of the Task 1
mask, because it is republished as the window history. Variants (NN v2 4-seed average vs CatBoost):
  blocks      : (a) only
  realistic   : (a) + (b)
  realistic_fix (NN only): (a) + (b), then the protected hour is re-masked at the day's regime rate before the NN sees it
Targets are the official February template cells outside the blocks and outside the protected hour.
Env: TFB_DATA_ROOT, TFB_MODELS (models/), TFB_NN_CODE (path to t1_nn.py), TFB_OUT, TFB_CAT_SRC (public CatBoost source).
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pandas as pd

MODELS = os.environ["TFB_MODELS"]
spec = importlib.util.spec_from_file_location("t1nn", os.environ["TFB_NN_CODE"]); nnm = importlib.util.module_from_spec(spec); spec.loader.exec_module(nnm)
import torch  # noqa: E402

sys.path.insert(0, os.environ["TFB_CAT_SRC"])
import state_model as base  # noqa: E402
from prepare_state import features  # noqa: E402
from state_inference import StatePredictor  # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nets = []
# TFB_NN_LIST (comma-separated .pt paths) overrides the default NN v2 seeds
for w in (os.environ["TFB_NN_LIST"].split(",") if os.environ.get("TFB_NN_LIST") else [f"{MODELS}/nn/v2_seed{i}.pt" for i in (1, 2, 3, 4)]):
    n = nnm.build_model().to(dev); n.load_state_dict(torch.load(w, map_location=dev)); n.eval(); nets.append(n)
cat = StatePredictor(f"{MODELS}/catboost")
rng = np.random.default_rng(0)
RATE = {"R1": 0.2, "R2": 0.3, "R3": 0.5}
rows = []
for pi, panel in enumerate(nnm.panels()):
    wpath = nnm.DATA / "task2" / panel / "validation" / "window_index.csv"
    if not wpath.exists():
        continue
    wi = pd.read_csv(wpath)
    pd_ = nnm.PanelData(panel); dates = list(pd_.trm["dates"])
    feb = [i for i, d in enumerate(dates) if d.startswith("2031-02")]
    sched = {}
    for w in wi.itertuples():
        dom = int(w.date[8:10])
        if dom <= len(feb):
            sched.setdefault(feb[dom - 1], []).append((int(w.forecast_origin[11:13]) * 60 + int(w.forecast_origin[14:16])) // 5)
    t, d, s, j = nnm.template_mask(panel, "train", dates, pd_.m)
    cp = base.Panel(panel, pi); links = pd_.m.link_id.to_numpy()
    for day, origins in sched.items():
        blk = np.zeros(288, bool); hist = np.zeros(288, bool)
        for o in origins:
            blk[o + 1:min(o + 19, 288)] = True; hist[max(o - 11, 0):o + 1] = True
        hist &= ~blk
        full_v, full_q, full_o = pd_.tr["speed"][day], pd_.tr["flow"][day], pd_.tr["occ"][day]
        mv, mq, mo = pd_.trm["speed"][day].copy(), pd_.trm["flow"][day].copy(), pd_.trm["occ"][day].copy()
        kin = (d == day) & blk[s]
        si, ji = s[kin], j[kin]
        k = (d == day) & ~blk[s] & ~hist[s]
        ss, jj = s[k], j[k]; tv, tq = full_v[ss, jj], full_q[ss, jj]
        variants = {}
        b_v, b_q, b_o = mv.copy(), mq.copy(), mo.copy(); b_v[blk] = b_q[blk] = b_o[blk] = np.nan
        variants["blocks"] = (b_v, b_q, b_o)
        r_v, r_q, r_o = b_v.copy(), b_q.copy(), b_o.copy()
        r_v[hist], r_q[hist], r_o[hist] = full_v[hist], full_q[hist], full_o[hist]  # protected history: no Task 1 mask
        variants["realistic"] = (r_v, r_q, r_o)
        f_v, f_q, f_o = r_v.copy(), r_q.copy(), r_o.copy()
        remask = hist[:, None] & (rng.random(f_v.shape) < RATE[pd_.trm["regimes"][day]])
        f_v[remask] = f_q[remask] = f_o[remask] = np.nan
        variants["realistic_fix"] = (f_v, f_q, f_o)
        variants = {"realistic": variants["realistic"]}
        for vname, (v, q, o) in variants.items():
            outs = [nnm.run(n, pd_, v, q, o, pd_.dows[day], pd_.prof["2031-02"], dev) for n in nets]
            nv, nq = np.mean([x[0] for x in outs], 0), np.mean([x[1] for x in outs], 0)
            iv = pd.DataFrame(v).interpolate(limit_direction="both").to_numpy(); iq = pd.DataFrame(q).interpolate(limit_direction="both").to_numpy()
            pv_, pq_ = pd_.prof["2031-02"][int(pd_.dows[day] >= 5)]
            ins = {"nn_v2": (nv[si, ji], nq[si, ji]), "interp": (iv[si, ji], iq[si, ji]), "profile": (pv_[si, ji], pq_[si, ji]),
                   "nn_interp_avg": ((nv[si, ji] + iv[si, ji]) / 2, (nq[si, ji] + iq[si, ji]) / 2)}
            tvi, tqi = full_v[si, ji], full_q[si, ji]
            preds = {"nn_v2": (nv[ss, jj], nq[ss, jj])}
            if vname != "realistic_fix":
                ts = [f"{dates[day]}T{x // 12:02d}:{(x % 12) * 5:02d}:00Z" for x in range(288)]
                fr = pd.DataFrame({"timestamp": np.repeat(ts, len(links)), "link_id": np.tile(links, 288), "speed_kmh": v.ravel(),
                                   "flow_vph": q.ravel(), "is_score_eligible": pd_.trm["elig"][day].ravel().astype(int)})
                times, V, Q, E = base.matrices(fr, cp)
                r_ = times.get_indexer([ts[x] for x in ss]); c_ = pd.Index(cp.links).get_indexer(links[jj])
                reg = {"R1": 0, "R2": 1, "R3": 2}[pd_.trm["regimes"][day]]
                X, bv, bq = features(cp, pi, times, V, Q, E, r_, c_, np.full(len(ss), reg))
                preds["catboost"] = (np.clip(bv + cat.residual(X, "speed"), 1, 180), np.clip((bq + cat.residual(X, "flow")) * cp.lanes[c_], 0, None))
                r2 = times.get_indexer([ts[x] for x in si]); c2 = pd.Index(cp.links).get_indexer(links[ji])
                X2, bv2, bq2 = features(cp, pi, times, V, Q, E, r2, c2, np.full(len(si), reg))
                ins["catboost"] = (np.clip(bv2 + cat.residual(X2, "speed"), 1, 180), np.clip((bq2 + cat.residual(X2, "flow")) * cp.lanes[c2], 0, None))
                for mname, (pv, pq) in ins.items():
                    rows.append(dict(panel=panel, day=dates[day], variant="IN_BLOCK", model=mname, n=len(si),
                                     se_v=float(((pv - tvi) ** 2).sum()), se_q=float((((pq - tqi) / pd_.lanes[ji]) ** 2).sum())))
            if vname == "realistic":
                fr2 = fr if False else None
            for mname, (pv, pq) in preds.items():
                rows.append(dict(panel=panel, day=dates[day], variant=vname, model=mname, n=len(ss),
                                 se_v=float(((pv - tv) ** 2).sum()), se_q=float((((pq - tq) / pd_.lanes[jj]) ** 2).sum())))
    print(panel, "days", len(sched), flush=True)
    pd.DataFrame(rows).to_csv(f"{os.environ['TFB_OUT']}/t1_blocks_eval2_rows.csv", index=False)

r = pd.DataFrame(rows)
g = r.groupby(["variant", "model"])[["n", "se_v", "se_q"]].sum()
g["rmse_v"] = np.sqrt(g.se_v / g.n); g["rmse_q_lane"] = np.sqrt(g.se_q / g.n)
g["S_approx"] = 0.54 * (1 - g.rmse_v / 25) + 0.46 * (1 - g.rmse_q_lane / 600)
print(g[["n", "rmse_v", "rmse_q_lane", "S_approx"]].round(4).to_string())
