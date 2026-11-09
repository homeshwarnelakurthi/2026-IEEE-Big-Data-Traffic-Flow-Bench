"""Task 2: window generation that mimics the official selector, causal features, and IoU scoring.

Arrays are (slot, link) per day with links in milepost order. Origin index t is the slot of T;
history is t-12..t-1 plus the row at t itself (allowed by the rules), horizon is t+1..t+6.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import RELEASE, meta, slot_of

HIST, HOR = 12, 6
PATCH = 6  # links each side in the space-time patch


def downstream_order(panel: str) -> np.ndarray:
    """Permutation of milepost-ordered links so that index increases in the travel direction."""
    m = meta(panel)
    topo = pd.read_csv(RELEASE / "corridors" / panel / "network" / "lwr_mainline_topology.csv", dtype=str)
    mp = dict(zip(m.link_id, m.milepost))
    votes = []
    for r in topo.itertuples(index=False):
        for o in str(r.outgoing_link_ids).split(";"):
            if r.link_id in mp and o in mp and mp[o] != mp[r.link_id]:
                votes.append(np.sign(mp[o] - mp[r.link_id]))
    idx = np.arange(len(m))
    return idx if np.sum(votes) >= 0 else idx[::-1]


class PanelT2:
    def __init__(self, panel: str, panel_no: int):
        self.panel, self.no = panel, panel_no
        m = meta(panel)
        self.order = downstream_order(panel)
        mm = m.iloc[self.order].reset_index(drop=True)
        self.links = mm.link_id.to_numpy()
        self.vf = mm.free_speed_kmh.to_numpy(np.float32)
        self.cap = mm.capacity_vph.to_numpy(np.float32)
        self.lanes = mm.lanes.to_numpy(np.float32)
        self.length = mm.length_km.to_numpy(np.float32)
        self.cut = (0.6 * mm.free_speed_kmh).to_numpy(np.float32)
        self.km = np.cumsum(self.length) - self.length
        self.L = len(mm)

    def orient(self, a: np.ndarray) -> np.ndarray:
        """Reorder the last (link) axis of a milepost-ordered array into travel order."""
        return a[..., self.order]

    # ---------- labels / candidates on a full (observed) day ----------
    def queued(self, speed: np.ndarray) -> np.ndarray:
        return np.isfinite(speed) & (speed <= self.cut)

    def candidates(self, speed: np.ndarray) -> dict[str, list[int]]:
        """Origins satisfying the published selector rules (coverage, queue in horizon, condition)."""
        q = self.queued(speed)
        active = q.any(axis=1)
        cov = np.isfinite(speed)
        out = {"queue_onset": [], "queue_ongoing": []}
        for t in range(HIST, speed.shape[0] - HOR):
            if cov[t - HIST:t].mean() < 0.7 or not active[t + 1:t + HOR + 1].any():
                continue
            if not active[t - HIST:t + 1].any():
                # selector takes the earliest qualifying origin of an episode: queue first at T+30
                if not active[t + 1:t + HOR].any():
                    out["queue_onset"].append(t)
            elif active[t]:
                pers = np.repeat(q[t - 1][None], HOR, 0)
                fut = q[t + 1:t + HOR + 1]
                inter, union = (pers & fut).sum(), (pers | fut).sum()
                if union and inter / union <= 0.9:
                    out["queue_ongoing"].append(t)
        return out

    # ---------- features ----------
    def features(self, S, F, t, dow, cond, prior_tod, month_mem, use_T=True):
        """S, F: (288, L) oriented arrays of the day as visible at T (NaN = unavailable).
        prior_tod: (288, L) train-period queue frequency; month_mem: (288, L) same-month earlier-days frequency.
        Returns (L, n_features) base features shared by all horizon steps."""
        L = self.L
        s = S[t - HIST:t + 1].copy() / self.vf
        f = F[t - HIST:t + 1].copy() / self.cap
        if not use_T:
            s[-1] = np.nan
            f[-1] = np.nan
        cutn = self.cut / self.vf
        valid = np.isfinite(s)
        pos = np.where(valid, np.arange(HIST + 1)[:, None], -1).max(0)
        cols = np.arange(L)
        last_s = np.where(pos >= 0, s[np.maximum(pos, 0), cols], np.nan)
        last_f = np.where(pos >= 0, f[np.maximum(pos, 0), cols], np.nan)
        age = np.where(pos >= 0, HIST - pos, 20).astype(np.float32)
        q = np.where(valid, s <= cutn, False)
        last_q = (last_s <= cutn).astype(np.float32)
        last_q[~np.isfinite(last_s)] = np.nan
        with np.errstate(all="ignore"):
            rec_s = np.nanmean(s[-3:], 0); pre_s = np.nanmean(s[-6:-3], 0)
            rec_f = np.nanmean(f[-3:], 0); pre_f = np.nanmean(f[-6:-3], 0)
            min_s = np.nanmin(s, 0); mean_s = np.nanmean(s, 0); max_f = np.nanmax(f, 0)
            dens = last_f / np.maximum(last_s, 0.05)  # relative density proxy (q/cap)/(v/vf)
        hour = t / 12.0
        c = lambda v: np.full(L, v, np.float32)
        feats = {
            "panel": c(self.no), "pos": np.arange(L, dtype=np.float32) / L, "vf": self.vf,
            "cap_lane": self.cap / np.maximum(self.lanes, 1), "lanes": self.lanes, "length": self.length,
            "hour": c(hour), "dow": c(dow), "weekend": c(dow >= 5), "ongoing": c(cond == "queue_ongoing"),
            "last_s": last_s, "last_f": last_f, "age": age, "last_q": last_q, "dens": dens,
            "rec_s": rec_s, "pre_s": pre_s, "d_s": rec_s - pre_s, "rec_f": rec_f, "pre_f": pre_f, "d_f": rec_f - pre_f,
            "min_s": min_s, "mean_s": mean_s, "max_f": max_f, "n_q": q.sum(0).astype(np.float32),
            "cov": valid.mean(0).astype(np.float32), "any_q_panel": c(q[-1].any() or q[-2].any()),
            "n_q_panel": c(q[-3:].any(0).sum()),
        }
        for lag in (1, 2, 3, 4, 6, 9, 13):
            feats[f"s_l{lag}"] = s[-lag]; feats[f"f_l{lag}"] = f[-lag]
        # space-time patch around each link (travel order: +k = downstream)
        pad = lambda a, k: np.concatenate([np.full(max(-k, 0), np.nan), a, np.full(max(k, 0), np.nan)])[max(k, 0):max(k, 0) + L] if k else a
        recent_min = np.where(np.isfinite(s[-3:]).any(0), np.nanmin(np.where(np.isfinite(s[-3:]), s[-3:], np.inf), 0), np.nan)
        for k in range(-PATCH, PATCH + 1):
            if k == 0:
                continue
            feats[f"ps_{k}"] = pad(last_s, k)
            feats[f"pm_{k}"] = pad(recent_min, k)
            feats[f"pf_{k}"] = pad(last_f, k)
            feats[f"pd_{k}"] = pad(dens, k)
        # distance (links / km) to the nearest queued link downstream and upstream at the latest step
        qq = np.nan_to_num(last_q) > 0.5
        idx = np.where(qq)[0]
        dn = np.full(L, 99.0, np.float32); up = np.full(L, 99.0, np.float32)
        dkm = np.full(L, 99.0, np.float32); ukm = np.full(L, 99.0, np.float32)
        if len(idx):
            j = np.searchsorted(idx, np.arange(L))
            has_dn = j < len(idx)
            dn[has_dn] = idx[j[has_dn]] - np.arange(L)[has_dn]
            dkm[has_dn] = self.km[idx[j[has_dn]]] - self.km[has_dn]
            jj = np.searchsorted(idx, np.arange(L), side="right") - 1
            has_up = jj >= 0
            up[has_up] = np.arange(L)[has_up] - idx[jj[has_up]]
            ukm[has_up] = self.km[has_up] - self.km[idx[jj[has_up]]]
        feats.update(dq_dn=dn, dq_up=up, dq_dn_km=dkm, dq_up_km=ukm)
        # bottleneck memory: long-run and same-month, at the target time of day
        for name, mem in (("prior", prior_tod), ("mmem", month_mem)):
            if mem is None:
                continue
            for h in (0, 3, 6):
                feats[f"{name}_h{h}"] = mem[min(t + h, 287)]
            feats[f"{name}_win"] = np.nanmean(mem[t:min(t + 13, 288)], 0)
            feats[f"{name}_dn1"] = pad(mem[min(t + 6, 287)], 1)
            feats[f"{name}_up1"] = pad(mem[min(t + 6, 287)], -1)
        names = list(feats)
        X = np.column_stack([np.asarray(feats[k], np.float32) for k in names])
        X[~np.isfinite(X)] = np.nan
        return X, names


def queue_memory(speeds: np.ndarray, cut: np.ndarray) -> np.ndarray:
    """Queue frequency per (slot, link) over a stack of days (D, 288, L), smoothed over +-2 slots."""
    if len(speeds) == 0:
        return np.full((288, len(cut)), np.nan, np.float32)
    obs = np.isfinite(speeds)
    q = (obs & (speeds <= cut)).sum(0).astype(np.float32)
    n = obs.sum(0).astype(np.float32)
    k = np.ones(5, np.float32)
    qs = np.apply_along_axis(lambda a: np.convolve(a, k, "same"), 0, q)
    ns = np.apply_along_axis(lambda a: np.convolve(a, k, "same"), 0, n)
    return np.where(ns > 0, qs / np.maximum(ns, 1), np.nan).astype(np.float32)


def iou(pred: np.ndarray, true: np.ndarray) -> float:
    union = (pred | true).sum()
    return 1.0 if union == 0 else (pred & true).sum() / union
