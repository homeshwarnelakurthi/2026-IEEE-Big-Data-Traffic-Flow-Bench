"""Task 1 masked-state imputer: axial attention over the (time x link) grid of one panel-day.

Trains on every train day with fresh random masks at the three regime rates, loss shaped like S_state
(speed /25 km/h, flow per lane /600), early-scores February on the official masks, and predicts the
validation/private template cells. Runs one seed per GPU. Self-contained so it can run as a Kaggle script.

Env: TFB_DATA_ROOT (release root), TFB_OUT (output dir), TFB_STEPS, TFB_SEEDS (comma list), TFB_SMOKE=1 for a CPU test.
"""
from __future__ import annotations

import glob
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(os.environ.get("TFB_DATA_ROOT") or glob.glob("/kaggle/input/**/kaggle_public", recursive=True)[0])
OUT = Path(os.environ.get("TFB_OUT", "/kaggle/working"))
CACHE = OUT / "cache"
STEPS = int(os.environ.get("TFB_STEPS", "40000"))
SMOKE = os.environ.get("TFB_SMOKE") == "1"
MAX_HOURS = float(os.environ.get("TFB_MAX_HOURS", "7.0"))  # training budget; inference follows
BLOCK_P = float(os.environ.get("TFB_BLOCK_P", "0.4"))  # share of training days with simulated Task 2 blocks (v1: 0, v2: 0.4)
SLOTS = 288
CROP = 64
HOLD = "2031-02"


# ----------------------------------------------------------------- data
def panels():
    return [p["corridor_id"] for p in json.loads((DATA / "config" / "corridors.json").read_text())["panels"]]


def meta(panel):
    net = DATA / "corridors" / panel / "network"
    fd = pd.read_csv(net / "fd_parameters.csv", dtype={"link_id": str}).drop_duplicates("link_id")
    first = sorted((DATA / "corridors" / panel / "train" / "mainline_states").glob("*/*.parquet"))[0]
    loc = pd.read_parquet(first, columns=["link_id", "station_id", "milepost"]).drop_duplicates("link_id")
    loc["link_id"] = loc.link_id.astype(str)
    m = loc.merge(fd, on="link_id").sort_values(["milepost", "link_id"]).reset_index(drop=True)
    topo = pd.read_csv(net / "lwr_mainline_topology.csv", dtype=str)
    mp = dict(zip(m.link_id, m.milepost)); votes = 0
    for r in topo.itertuples(index=False):
        for o in str(r.outgoing_link_ids).split(";"):
            if r.link_id in mp and o in mp:
                votes += np.sign(mp[o] - mp[r.link_id])
    return m if votes >= 0 else m.iloc[::-1].reset_index(drop=True)  # travel order


def dense(panel, split, masked, m):
    out = CACHE / f"{panel}_{split}_{int(masked)}.npz"
    if out.exists():
        z = np.load(out); return {k: z[k] for k in z.files}
    sub = "mainline_states_masked" if masked else "mainline_states"
    files = sorted((DATA / "corridors" / panel / split / sub).glob("*/*.parquet"), key=lambda p: p.name)
    col = {l: i for i, l in enumerate(m.link_id)}
    D, L = len(files), len(m)
    v = np.full((D, SLOTS, L), np.nan, np.float32); q = v.copy(); o = v.copy(); e = np.zeros((D, SLOTS, L), bool)
    dates, regs = [], []
    for d, f in enumerate(files):
        x = pd.read_parquet(f, columns=["date", "timestamp", "link_id", "speed_kmh", "flow_vph", "occupancy", "is_score_eligible"])
        ts = x.timestamp.astype(str)
        s = ((ts.str[11:13].astype(int) * 60 + ts.str[14:16].astype(int)) // 5).to_numpy()
        j = x.link_id.astype(str).map(col).to_numpy()
        ok = ~pd.isna(j); s, j = s[ok], j[ok].astype(int)
        v[d, s, j] = x.speed_kmh.to_numpy()[ok]; q[d, s, j] = x.flow_vph.to_numpy()[ok]
        o[d, s, j] = x.occupancy.to_numpy()[ok]; e[d, s, j] = x.is_score_eligible.to_numpy()[ok].astype(bool)
        dates.append(str(x.date.iloc[0])); regs.append(f.parent.name.split("=")[-1] if masked else "")
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(out, speed=v, flow=q, occ=o, elig=e, dates=np.array(dates), regimes=np.array(regs))
    return dense(panel, split, masked, m)


def template_mask(panel, split, dates, m):
    t = pd.read_csv(DATA / "task1" / panel / split / "sample_submission_state.csv", dtype=str)
    di = {x: i for i, x in enumerate(dates)}; col = {l: i for i, l in enumerate(m.link_id)}
    d = t.timestamp.str[:10].map(di).to_numpy()
    s = ((t.timestamp.str[11:13].astype(int) * 60 + t.timestamp.str[14:16].astype(int)) // 5).to_numpy()
    j = t.link_id.map(col).to_numpy()
    return t, d, s, j


def profiles(v, q, dows, months, exclude=None):
    """Weekday/weekend x slot mean per link, optionally excluding one month (leave-own-month-out)."""
    keep = np.ones(len(months), bool) if exclude is None else months != exclude
    out = []
    for we in (False, True):
        sel = keep & ((dows >= 5) == we)
        with np.errstate(all="ignore"):
            out.append((np.nanmean(v[sel], 0), np.nanmean(q[sel], 0)))
    return out


class PanelData:
    def __init__(self, panel):
        self.panel = panel
        self.m = meta(panel)
        self.L = len(self.m)
        self.vf = self.m.free_speed_kmh.to_numpy(np.float32)
        self.cap = self.m.capacity_vph.to_numpy(np.float32)
        self.lanes = self.m.lanes.to_numpy(np.float32)
        self.static = np.stack([self.vf / 120, self.cap / self.lanes / 2400, self.lanes / 6,
                                np.log1p(self.m.length_km.to_numpy(np.float32))]).astype(np.float32)
        self.tr = dense(panel, "train", False, self.m)
        self.trm = dense(panel, "train", True, self.m)
        self.months = np.array([d[:7] for d in self.tr["dates"]])
        self.dows = pd.to_datetime(self.tr["dates"]).day_of_week.to_numpy()
        self.prof = {mo: profiles(self.tr["speed"], self.tr["flow"], self.dows, self.months, mo) for mo in np.unique(self.months)}
        self.prof_all = profiles(self.tr["speed"], self.tr["flow"], self.dows, self.months)

    def inputs(self, v, q, o, dow, prof):
        """Channel stack (C, 288, L) from visible observations (NaN = hidden)."""
        ps, pq = prof[int(dow >= 5)]
        ps = np.where(np.isfinite(ps), ps, self.vf); pq = np.where(np.isfinite(pq), pq, 0.3 * self.cap)
        obs = np.isfinite(v) & np.isfinite(q)
        vn = np.where(obs, v / self.vf, 0); qn = np.where(obs, q / self.cap, 0); on = np.where(np.isfinite(o), o * 5, 0)
        tod = np.arange(SLOTS) / SLOTS * 2 * np.pi
        L = self.L
        ch = [vn, qn, on, obs.astype(np.float32), ps / self.vf, pq / self.cap,
              np.repeat(np.sin(tod)[:, None], L, 1), np.repeat(np.cos(tod)[:, None], L, 1),
              np.full((SLOTS, L), float(dow >= 5)), np.full((SLOTS, L), obs.mean())]
        ch += [np.repeat(s[None], SLOTS, 0) for s in self.static]
        return np.stack(ch).astype(np.float32)


# ----------------------------------------------------------------- model
def build_model(d=96, heads=4, blocks=6, cin=14):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Axial(nn.Module):
        def __init__(self, axis):
            super().__init__(); self.axis = axis
            self.ln = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d)

        def forward(self, x):  # x: (B, T, L, d)
            B, T, L, _ = x.shape
            y = self.ln(x)
            if self.axis == "t":
                y = y.permute(0, 2, 1, 3).reshape(B * L, T, d)
            else:
                y = y.reshape(B * T, L, d)
            qkv = self.qkv(y).reshape(y.shape[0], y.shape[1], 3, heads, d // heads).permute(2, 0, 3, 1, 4)
            a = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
            a = a.transpose(1, 2).reshape(y.shape[0], y.shape[1], d)
            a = self.proj(a)
            if self.axis == "t":
                a = a.reshape(B, L, T, d).permute(0, 2, 1, 3)
            else:
                a = a.reshape(B, T, L, d)
            return x + a

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.dw = nn.Conv2d(d, d, 5, padding=2, groups=d)
            self.t = Axial("t"); self.l = Axial("l")
            self.ln = nn.LayerNorm(d); self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

        def forward(self, x):  # (B, T, L, d)
            x = x + self.dw(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            x = self.l(self.t(x))
            return x + self.mlp(self.ln(x))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv2d(cin, d, 3, padding=1), nn.GELU(), nn.Conv2d(d, d, 3, padding=1))
            self.blocks = nn.ModuleList([Block() for _ in range(blocks)])
            self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2))

        def forward(self, x):  # (B, C, T, L) -> (B, 2, T, L) normalized speed/vf, flow/cap
            h = self.stem(x).permute(0, 2, 3, 1)
            for b in self.blocks:
                h = b(h)
            return self.head(h).permute(0, 3, 1, 2) + x[:, 4:6]  # residual on the profile channels

    return Net()


# ----------------------------------------------------------------- training
def train(seed, gpu):
    import torch
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    dev = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    P = {p: PanelData(p) for p in (panels()[:2] if SMOKE else panels())}
    net = build_model().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.02)
    warm = max(1, STEPS // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, STEPS) / STEPS)))
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    pool = [(p, d) for p, pd_ in P.items() for d in np.where(pd_.months != HOLD)[0]]
    B = 2 if SMOKE else 8
    log = open(OUT / f"train_seed{seed}.log", "a")
    best, t0 = -1, time.time()
    for step in range(1, STEPS + 1):
        X, Y, M, W = [], [], [], []
        for _ in range(B):
            p, d = pool[rng.integers(len(pool))]; pd_ = P[p]
            v, q, o, e = pd_.tr["speed"][d], pd_.tr["flow"][d], pd_.tr["occ"][d], pd_.tr["elig"][d]
            rate = rng.choice([0.2, 0.3, 0.5])
            hide = e & np.isfinite(v) & (rng.random(v.shape) < rate)
            # validation/private days with Task 2 windows blank the 30-min horizon plus the following hour
            # on every link; mimic that so those days stay in distribution (block cells are not scored)
            blk = np.zeros(v.shape, bool)
            if rng.random() < BLOCK_P:
                for _ in range(rng.integers(1, 3)):
                    s0 = rng.integers(84, 252); blk[s0:s0 + 18] = True
            hide &= ~blk
            gone = hide | blk
            x = pd_.inputs(np.where(gone, np.nan, v), np.where(gone, np.nan, q), np.where(gone, np.nan, o),
                           pd_.dows[d], pd_.prof[pd_.months[d]])
            L = pd_.L; c0 = rng.integers(0, max(L - CROP, 0) + 1); sl = slice(c0, c0 + CROP)
            y = np.stack([np.nan_to_num(v / pd_.vf), np.nan_to_num(q / pd_.cap)])
            # weights turn normalized errors into the score's units: speed km/h /25, flow per lane /600
            w = np.stack([np.broadcast_to(pd_.vf / 25, v.shape), np.broadcast_to(pd_.cap / pd_.lanes / 600, v.shape)])
            def pad(a):
                a = a[..., sl]
                return np.pad(a, [(0, 0)] * (a.ndim - 1) + [(0, CROP - a.shape[-1])])
            X.append(pad(x)); Y.append(pad(y)); M.append(pad(hide[None].astype(np.float32))); W.append(pad(w))
        X, Y, M, W = (torch.from_numpy(np.stack(a)).to(dev) for a in (X, Y, M, W))
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            out = net(X)
        err = ((out.float() - Y) * W) ** 2
        loss = (0.54 * err[:, 0] * M[:, 0] + 0.46 * err[:, 1] * M[:, 0]).sum() / M.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        if step % 200 == 0:
            print(f"seed{seed} step {step} loss {loss.item():.5f} {time.time()-t0:.0f}s", file=log, flush=True)
        out_of_time = time.time() - t0 > MAX_HOURS * 3600
        if step % (STEPS // 4 if not SMOKE else STEPS) == 0 or step == STEPS or out_of_time:
            s = feb_score(net, P, dev, save=None)
            print(f"seed{seed} step {step} FEB S_state {s['family_mean']:.5f} {json.dumps(s['panels'])}", file=log, flush=True)
            if s["family_mean"] > best:
                best = s["family_mean"]; torch.save(net.state_dict(), OUT / f"t1nn_seed{seed}.pt")
        if out_of_time:
            print(f"seed{seed} stopping at step {step}: time budget", file=log, flush=True)
            break
    net.load_state_dict(torch.load(OUT / f"t1nn_seed{seed}.pt", map_location=dev))
    feb_score(net, P, dev, save=OUT / f"feb_pred_seed{seed}.npz")
    for split in ("validation", "private"):
        predict_split(net, P, dev, split, OUT / f"state_{split}_seed{seed}.csv")
    print(f"seed{seed} done best {best:.5f}", file=log, flush=True)


def run(net, pd_, v, q, o, dow, prof, dev):
    import torch
    x = pd_.inputs(v, q, o, dow, prof)
    with torch.no_grad(), torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
        y = net(torch.from_numpy(x[None]).to(dev)).float().cpu().numpy()[0]
    return y[0] * pd_.vf, y[1] * pd_.cap


def feb_score(net, P, dev, save):
    net.eval(); per, saved = {}, {}
    for p, pd_ in P.items():
        dates = list(pd_.trm["dates"]); t, d, s, j = template_mask(p, "train", dates, pd_.m)
        sel = np.array([dates[i].startswith(HOLD) for i in d])
        d, s, j = d[sel], s[sel], j[sel]
        pv = np.zeros(len(d), np.float32); pq = np.zeros(len(d), np.float32)
        for day in np.unique(d):
            v_, q_ = run(net, pd_, pd_.trm["speed"][day], pd_.trm["flow"][day], pd_.trm["occ"][day],
                         pd_.dows[day], pd_.prof[HOLD], dev)
            k = d == day; pv[k] = v_[s[k], j[k]]; pq[k] = q_[s[k], j[k]]
        tv, tq = pd_.tr["speed"][d, s, j], pd_.tr["flow"][d, s, j]
        reg = pd_.trm["regimes"][d]; lanes = pd_.lanes[j]; sc = []
        for r in ("R1", "R2", "R3"):
            k = reg == r
            if k.any():
                rv = np.sqrt(np.mean((pv[k] - tv[k]) ** 2)); rq = np.sqrt(np.mean(((pq[k] - tq[k]) / lanes[k]) ** 2))
                sc.append(0.54 * max(0, 1 - rv / 25) + 0.46 * max(0, 1 - rq / 600))
        per[p] = float(np.mean(sc))
        saved[p] = (d, s, j, pv, pq)
    fam = {}
    for p, v in per.items():
        fam.setdefault(p.rsplit("_", 1)[0], []).append(v)
    if save is not None:
        np.savez(save, **{f"{p}_{k}": a for p, arrs in saved.items() for k, a in zip("dsjvq", arrs)})
    net.train()
    return dict(family_mean=float(np.mean([np.mean(v) for v in fam.values()])), panels={k: round(v, 5) for k, v in per.items()})


def predict_split(net, P, dev, split, path):
    net.eval(); parts = []
    for p, pd_ in P.items():
        z = dense(p, split, True, pd_.m); dates = list(z["dates"])
        t, d, s, j = template_mask(p, split, dates, pd_.m)
        dows = pd.to_datetime(z["dates"]).day_of_week.to_numpy()
        pv = np.zeros(len(t), np.float32); pq = np.zeros(len(t), np.float32)
        for day in np.unique(d):
            v_, q_ = run(net, pd_, z["speed"][day], z["flow"][day], z["occ"][day], dows[day], pd_.prof_all, dev)
            k = d == day; pv[k] = v_[s[k], j[k]]; pq[k] = q_[s[k], j[k]]
        t["speed_kmh"] = np.clip(pv, 3, 140); t["flow_vph"] = np.clip(pq, 0, None)
        parts.append(t)
    pd.concat(parts, ignore_index=True).to_csv(path, index=False)
    net.train()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        train(int(sys.argv[2]), int(sys.argv[3]))
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        for p in (panels()[:2] if SMOKE else panels()):  # build caches once before the workers fork
            m = meta(p)
            for sp, mk in (("train", False), ("train", True), ("validation", True), ("private", True)):
                dense(p, sp, mk, m)
            print("cached", p, flush=True)
        seeds = [int(s) for s in os.environ.get("TFB_SEEDS", "1,2").split(",")]
        procs = [subprocess.Popen([sys.executable, __file__, "worker", str(sd), str(i)]) for i, sd in enumerate(seeds)]
        for pr in procs:
            pr.wait()
        for f in sorted(OUT.glob("train_seed*.log")):
            print(f.read_text()[-3000:])
