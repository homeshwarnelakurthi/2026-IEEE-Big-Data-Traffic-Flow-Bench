# Source: yeashusemwal, "TrafficFlowBench Trees + U-Net + CNN" (Kaggle notebook, 2026),
# kaggle.com/code/yeashusemwal/trafficflowbench-trees-u-net-cnn-lb-0-87. Redistributed for reproduction with credit; see NOTICE.
"""Inference-only replay of the Task 1 members used in candidate W (no training).

trees(model_dirs): LightGBM speed and per-lane flow models (task1_v4.py family) on task1_lgbm features with the
                  full-train weekday profile, features built once for all models; outputs clipped to [5, 1.08 vf] km/h
                  and >= 100 vph as in training.
unet(weights):    2-D U-Net over each day's (links x 288) grid (task1_unet2.py), predicting a residual on top of the
                  weekday profile; same clipping. Candidate W averages two seeds (51, 52) at width 48.
Both return {model: {split: DataFrame in sample_submission_state order with speed_kmh, flow_vph}}.
"""
import os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "v2"))
import task2_v11 as v11, task1_lgbm as t1, task1_unet2 as un
from common import DATA, T, panels, network, day_files, date_of, regime_of, weekday_of, slots_of, day_grid

_PROF = {}
SPLITS = ("validation", "private")


def full_profile(p):
    if p not in _PROF:
        net = network(p); st, _ = v11.profile_stats(p, net); _PROF[p] = (net, v11.profile_from(st))
    return _PROF[p]


def _template(p, split, li):
    tpl = pd.read_csv(DATA / "task1" / p / split / "sample_submission_state.csv", dtype=str)
    return tpl, tpl.timestamp.str[:10].to_numpy(), slots_of(tpl.timestamp), tpl.link_id.map(li).to_numpy(dtype=int)


def trees(model_dirs, threads=4, splits=SPLITS, panel_list=None):
    """Predict with several tree models at once. Features are identical across the models, so each day's features are
    built once and every model predicts from them. Returns {model_dir: {split: DataFrame}}."""
    import lightgbm as lgb
    if isinstance(model_dirs, str): model_dirs = [model_dirs]
    ms = {d: {k: lgb.Booster(model_file=os.path.join(d, f"task1_{k}.txt"), params={"num_threads": threads}) for k in ("speed", "flow")} for d in model_dirs}
    res = {d: {} for d in model_dirs}
    for split in splits:
        outs = {d: [] for d in model_dirs}
        for p in (panel_list or panels()):
            net, full = full_profile(p); li = {l: i for i, l in enumerate(net.link_id)}; M = len(net); lanes = net.lanes.to_numpy()
            tpl, d_arr, slot, lidx = _template(p, split, li)
            pv = {d: np.zeros(len(tpl), np.float32) for d in model_dirs}; pq = {d: np.zeros(len(tpl), np.float32) for d in model_dirs}
            for mp in day_files(p, split, masked=True):
                day = date_of(mp); rows = np.nonzero(d_arr == day)[0]
                if len(rows) == 0: continue
                g = day_grid(mp, li, M); X, _ = t1.features(g, net, full, weekday_of(day), regime_of(mp)); Xr = X[lidx[rows], slot[rows]]
                for d in model_dirs:
                    pv[d][rows] = ms[d]["speed"].predict(Xr); pq[d][rows] = ms[d]["flow"].predict(Xr) * lanes[lidx[rows]]
            vf = net.free_speed_kmh.to_numpy()[lidx]
            for d in model_dirs:
                o = tpl.copy(); o["speed_kmh"] = np.clip(pv[d], 5.0, vf * 1.08); o["flow_vph"] = np.clip(pq[d], 100.0, None); outs[d].append(o)
        for d in model_dirs: res[d][split] = pd.concat(outs[d], ignore_index=True)
    return res


def build_unet(width=48):
    import torch, torch.nn as nn, torch.nn.functional as F
    def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.GELU(), nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.GELU())

    class UNet(nn.Module):
        def __init__(s, w=width):
            super().__init__(); s.e1 = blk(13, w); s.e2 = blk(w, 2 * w); s.e3 = blk(2 * w, 4 * w); s.b = blk(4 * w, 8 * w)
            s.u3 = nn.ConvTranspose2d(8 * w, 4 * w, 2, 2); s.d3 = blk(8 * w, 4 * w); s.u2 = nn.ConvTranspose2d(4 * w, 2 * w, 2, 2); s.d2 = blk(4 * w, 2 * w)
            s.u1 = nn.ConvTranspose2d(2 * w, w, 2, 2); s.d1 = blk(2 * w, w); s.o = nn.Conv2d(w, 2, 1)
        def forward(s, x):
            e1 = s.e1(x); e2 = s.e2(F.max_pool2d(e1, 2)); e3 = s.e3(F.max_pool2d(e2, 2)); b = s.b(F.max_pool2d(e3, 2))
            d3 = s.d3(torch.cat([s.u3(b), e3], 1)); d2 = s.d2(torch.cat([s.u2(d3), e2], 1)); d1 = s.d1(torch.cat([s.u1(d2), e1], 1))
            return torch.stack([x[:, 4], x[:, 5]], 1) + s.o(d1)          # residual on top of the profile
    return UNet()


def unet(weights, width=48, device="cpu", splits=SPLITS, batch=4, threads=4, panel_list=None):
    """Predict with one or more U-Net weight files. Each day's input grid is built once and fed to every network,
    several days per forward pass. Returns {weights: {split: DataFrame}}."""
    import torch
    torch.set_num_threads(threads)
    if isinstance(weights, str): weights = [weights]
    nets = {}
    for w in weights:
        n = build_unet(width); n.load_state_dict(torch.load(w, map_location=device)); nets[w] = n.to(device).eval()
    res = {w: {} for w in weights}
    for split in splits:
        outs = {w: [] for w in weights}
        for p in (panel_list or panels()):
            netw, full = full_profile(p); li = {l: i for i, l in enumerate(netw.link_id)}; M = len(netw); st = un.static(netw)
            tpl, d_arr, slot, lidx = _template(p, split, li)
            pv = {w: np.zeros(len(tpl), np.float32) for w in weights}; pq = {w: np.zeros(len(tpl), np.float32) for w in weights}
            jobs = []
            for f in day_files(p, split, masked=True):
                day = date_of(f); rows = np.nonzero(d_arr == day)[0]
                if len(rows) == 0: continue
                g = day_grid(f, li, M, with_occ=True); obs = np.isfinite(g["v"]) & np.isfinite(g["q"])
                X = np.zeros((13, un.MMAX, T), np.float32); X[:, :M] = un.make_x(g["v"], g["q"], g["occ"], obs, full, weekday_of(day), un.RATE[regime_of(f)], st)
                jobs.append((rows, X))
            for i in range(0, len(jobs), batch):
                chunk = jobs[i:i + batch]; xb = torch.tensor(np.stack([x for _, x in chunk]), device=device)
                for w in weights:
                    with torch.no_grad(): out = nets[w](xb).float().cpu().numpy()[:, :, :M]
                    for (rows, _), o in zip(chunk, out):
                        pv[w][rows] = (o[0] * st["vf"][:, None])[lidx[rows], slot[rows]]; pq[w][rows] = (o[1] * st["cap"][:, None])[lidx[rows], slot[rows]]
            for w in weights:
                o = tpl.copy(); o["speed_kmh"] = np.clip(pv[w], 5.0, st["vf"][lidx] * 1.08); o["flow_vph"] = np.clip(pq[w], 100.0, None); outs[w].append(o)
        for w in weights: res[w][split] = pd.concat(outs[w], ignore_index=True)
    return res
