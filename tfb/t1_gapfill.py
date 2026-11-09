"""Dedicated filler for Task 1 targets inside Task 2 blank blocks (T+5..T+90 on every link, ~2% of targets but ~40% of
the squared speed error). LightGBM predicts speed and per-lane flow as a residual on a profile-adjusted interpolation
across the gap, from the values just before/after the gap on the link and its neighbours.

eval : train on Jun-Jan days with simulated blocks; score February days carrying March's window schedule against the
       truth, next to CatBoost's in-block numbers from kernel A (speed RMSE 7.26, flow/lane RMSE 89.0)
final: train on all months; predict the in-block template cells of validation/private -> cache/t1_gapfill_{split}.csv
"""
from __future__ import annotations

import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import CACHE, RELEASE, load, meta, panels, slot_of
from t1_common import targets

B, EDGE = 18, 3  # block length (slots) and edge window used for before/after values
RNG = np.random.default_rng(5)
DUMP = os.environ.get("TFB_GF_DUMP")  # eval mode: write per-cell February predictions here
RAMP = os.environ.get("TFB_GF_RAMP") == "1"  # add the on-ramp jam sensor (released ramp flows inside the block)


def profile(speed, flow, dows, sel):
    out = {}
    for we in (False, True):
        m = sel & ((dows >= 5) == we)
        with np.errstate(all="ignore"):
            out[we] = (np.nanmean(speed[m], 0), np.nanmean(flow[m], 0))
    return out


def edge(a, lo, hi):
    """Mean over finite values of rows lo:hi (clipped), per link."""
    lo, hi = max(lo, 0), min(hi, a.shape[0])
    with np.errstate(all="ignore"):
        return np.nanmean(a[lo:hi], 0) if hi > lo else np.full(a.shape[1], np.nan)


def onramps(panel, split, dates):
    """On-ramp flow summed per mainline link, (D, 288, L); NaN where no ramp or no valid released cell."""
    links = {l: i for i, l in enumerate(meta(panel).link_id)}
    a = pd.read_csv(RELEASE / "corridors" / panel / "network" / "ramp_attachment_map.csv", dtype=str)
    rm = dict(zip(a.ramp_link_id, a.nearest_mainline_link_id.map(links)))
    files = {f.stem.split("synthetic_ramp_")[1].replace("_", "-"): f for f in (RELEASE / "corridors" / panel / split / "ramp_states").glob("*/*.parquet")}
    on = np.full((len(dates), 288, len(links)), np.nan, np.float32)
    for d, date in enumerate(dates):
        if date not in files:
            continue
        f = pd.read_parquet(files[date], columns=["timestamp", "ramp_link_id", "ramp_type", "flow_vph", "is_score_eligible", "pct_observed"])
        f = f[f.is_score_eligible.astype(bool) & f.flow_vph.notna() & (f.pct_observed >= 75) & f.ramp_type.str.upper().isin(["OR", "ON"])]
        g = pd.DataFrame({"t": f.timestamp.map(slot_of).to_numpy(), "j": f.ramp_link_id.map(rm).to_numpy(), "q": f.flow_vph.to_numpy()}).dropna()
        g = g.groupby(["t", "j"], as_index=False).q.sum()
        on[d, g.t.to_numpy().astype(int), g.j.to_numpy().astype(int)] = g.q.to_numpy()
    return on


def ramp_profile(on, v, vf, dows, sel):
    """Free-flow on-ramp profile per weekend flag: jammed cells are left out, else a recurrent jam would hide itself."""
    out = {}
    free = np.where(v / vf < 0.6, np.nan, on)
    for we in (False, True):
        with np.errstate(all="ignore"):
            pr = np.nanmean(free[sel & ((dows >= 5) == we)], 0)
        pr = pd.DataFrame(pr).rolling(13, center=True, min_periods=1).mean().to_numpy(np.float32)
        out[we] = np.where(pr > 30, pr, np.nan)
    return out


def sensor(low, w):
    """Share of ramp-off cells within +-w links, per slot; low: (S, L) float with NaN where no ramp reading."""
    ok = np.isfinite(low); a = np.nan_to_num(low); S, L = low.shape
    ca, cn = np.zeros((S, L + 1)), np.zeros((S, L + 1))
    ca[:, 1:] = np.cumsum(a, 1); cn[:, 1:] = np.cumsum(ok, 1)
    lo, hi = np.maximum(np.arange(L) - w, 0), np.minimum(np.arange(L) + w + 1, L)
    n = cn[:, hi] - cn[:, lo]
    with np.errstate(all="ignore"):
        return np.where(n > 0, (ca[:, hi] - ca[:, lo]) / n, np.nan)


def cell_features(v, q, o, pv, pq, vf, cap, lanes, dow, rr=None):
    """Features for every (slot in block, link). v, q: (288, L) day with the block blanked; pv, pq: profile (288, L)."""
    L = v.shape[1]
    s_idx = np.arange(o + 1, min(o + B + 1, 288))
    bv, bq = edge(v, o - EDGE + 1, o + 1), edge(q, o - EDGE + 1, o + 1)          # before (incl. row at T)
    av, aq = edge(v, o + B + 1, o + B + 1 + EDGE), edge(q, o + B + 1, o + B + 1 + EDGE)  # after
    pbv, pbq = edge(pv, o - EDGE + 1, o + 1), edge(pq, o - EDGE + 1, o + 1)
    pav, paq = edge(pv, o + B + 1, o + B + 1 + EDGE), edge(pq, o + B + 1, o + B + 1 + EDGE)
    rows = []
    if rr is not None:  # ramp-off indicator: on-ramp flow under 0.2 x its free-flow profile, seen only where a ramp reports
        low = np.where(np.isfinite(rr), (rr < 0.2).astype(np.float32), np.nan)[s_idx]
        sens = {w: sensor(low, w) for w in (0, 1, 3, 7, 15)}
        on3 = np.nan_to_num(sens[3]) > 0; pos_ = np.arange(1, len(s_idx) + 1) / (B + 1)
        first = np.where(on3.any(0), pos_[on3.argmax(0)], np.nan); last = np.where(on3.any(0), pos_[len(s_idx) - 1 - on3[::-1].argmax(0)], np.nan)
        with np.errstate(all="ignore"):
            blk3, blk7 = np.nanmean(sens[3], 0), np.nanmean(sens[7], 0)
    for k, s in enumerate(s_idx, start=1):
        w = k / (B + 1)
        rb_v, ra_v = bv - pbv, av - pav  # residuals vs profile at the two edges
        rb_q, ra_q = bq - pbq, aq - paq
        base_v = pv[s] + np.where(np.isfinite(rb_v) & np.isfinite(ra_v), (1 - w) * rb_v + w * ra_v, np.where(np.isfinite(rb_v), rb_v, np.nan_to_num(ra_v)))
        base_q = pq[s] + np.where(np.isfinite(rb_q) & np.isfinite(ra_q), (1 - w) * rb_q + w * ra_q, np.where(np.isfinite(rb_q), rb_q, np.nan_to_num(ra_q)))
        f = {"k": np.full(L, k), "w": np.full(L, w), "slot": np.full(L, s), "dow": np.full(L, dow), "we": np.full(L, dow >= 5),
             "vf": vf, "cap_lane": cap / lanes, "lanes": lanes, "pos": np.arange(L) / L,
             "pv": pv[s] / vf, "pq": pq[s] / cap, "base_v": base_v / vf, "base_q": base_q / cap,
             "bv": bv / vf, "av": av / vf, "bq": bq / cap, "aq": aq / cap, "rb_v": rb_v / vf, "ra_v": ra_v / vf,
             "rb_q": rb_q / cap, "ra_q": ra_q / cap, "jam_b": (bv / vf < 0.6).astype(float), "jam_a": (av / vf < 0.6).astype(float)}
        for d in (-2, -1, 1, 2):
            sh = lambda a: np.roll(a, d) if True else a
            f[f"bv{d}"] = sh(bv / vf); f[f"av{d}"] = sh(av / vf); f[f"rbv{d}"] = sh(rb_v / vf); f[f"rav{d}"] = sh(ra_v / vf)
            f[f"bq{d}"] = sh(bq / cap); f[f"aq{d}"] = sh(aq / cap)
        if rr is not None:
            f["rr"] = rr[s]
            for w_, a_ in sens.items():
                f[f"rs{w_}"] = a_[k - 1]
            f["rblk3"], f["rblk7"], f["rfirst"], f["rlast"] = blk3, blk7, first, last
        rows.append((s, np.column_stack([np.asarray(x, np.float32) for x in f.values()]), base_v, base_q))
    return rows, list(f.keys())


def day_rows(v, q, m, dow, origins, truth=None, pv=None, pq=None, rr=None):
    """Build features for all block cells of one day (v, q: (288, L) visible data). Returns X, base_v, base_q, slot, link (+ truth)."""
    Xs, bvs, bqs, sl, ln, tv, tq = [], [], [], [], [], [], []
    v, q = v.copy(), q.copy()
    for o in origins:  # all blocks of the day are blank at once
        v[o + 1:o + B + 1] = np.nan; q[o + 1:o + B + 1] = np.nan
    for o in origins:
        vv, qq = v, q
        rows, names = cell_features(vv, qq, o, pv, pq, m["vf"], m["cap"], m["lanes"], dow, rr)
        for s, X, bv, bq in rows:
            Xs.append(X); bvs.append(bv); bqs.append(bq); sl.append(np.full(len(bv), s)); ln.append(np.arange(len(bv)))
            if truth is not None:
                tv.append(truth[0][s]); tq.append(truth[1][s])
    out = [np.concatenate(Xs), np.concatenate(bvs), np.concatenate(bqs), np.concatenate(sl), np.concatenate(ln)]
    if truth is not None:
        out += [np.concatenate(tv), np.concatenate(tq)]
    return out, names


def panel_arrays(panel, split, masked):
    z = load(panel, split, masked=masked)
    mm = meta(panel)
    m = {"vf": mm.free_speed_kmh.to_numpy(np.float32), "cap": mm.capacity_vph.to_numpy(np.float32), "lanes": mm.lanes.to_numpy(np.float32)}
    return {"v": z["speed"], "q": z["flow"], "m": m, "dow": pd.to_datetime(z["dates"]).day_of_week.to_numpy(), "dates": z["dates"]}


def build_train(months_ok):
    X, yv, yq, W = [], [], [], []
    for panel in panels():
        full = panel_arrays(panel, "train", False); masked = panel_arrays(panel, "train", True)
        months = np.array([d[:7] for d in full["dates"]])
        on = onramps(panel, "train", full["dates"]) if RAMP else None
        for mo in np.unique(months):
            if not months_ok(mo):
                continue
            prof = profile(full["v"], full["q"], full["dow"], months != mo)
            rp = ramp_profile(on, full["v"], full["m"]["vf"], full["dow"], months != mo) if RAMP else None
            for d in np.where(months == mo)[0]:
                pv, pq = prof[full["dow"][d] >= 5]
                if d % 2:
                    continue  # every other day keeps the row count manageable on 4 CPUs
                origins = [int(RNG.integers(84, 252))]
                v, q = masked["v"][d].copy(), masked["q"][d].copy()
                for o in origins:  # the protected history hour carries no Task 1 mask
                    v[max(o - 11, 0):o + 1] = full["v"][d, max(o - 11, 0):o + 1]; q[max(o - 11, 0):o + 1] = full["q"][d, max(o - 11, 0):o + 1]
                (x, bv, bq, sl, ln, tv, tq), names = day_rows(v, q, full["m"], full["dow"][d], origins, (full["v"][d], full["q"][d]), pv, pq, on[d] / rp[full["dow"][d] >= 5] if RAMP else None)
                ok = np.isfinite(tv) & np.isfinite(tq) & np.isfinite(bv) & np.isfinite(bq)
                m = full["m"]
                X.append(x[ok]); yv.append(((tv - bv) / m["vf"][ln])[ok]); yq.append(((tq - bq) / m["cap"][ln])[ok])
                W.append(np.stack([m["vf"][ln] / 25, m["cap"][ln] / m["lanes"][ln] / 600], 1)[ok])
        print(panel, "train rows", sum(len(a) for a in yv), flush=True)
    return np.concatenate(X), np.concatenate(yv), np.concatenate(yq), np.concatenate(W), names


def fit(X, y, w):
    p = dict(objective="regression", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=5, num_threads=4, verbose=-1, seed=3)
    return lgb.train(p, lgb.Dataset(X, y, weight=w ** 2), 1500)


def evaluate():
    X, yv, yq, W, names = build_train(lambda mo: mo != "2031-02")
    mv, mq = fit(X, yv, W[:, 0]), fit(X, yq, W[:, 1]); del X
    se = {"gapfill": [0, 0, 0], "base": [0, 0, 0]}; dump = []
    for panel in panels():
        wp = RELEASE / "task2" / panel / "validation" / "window_index.csv"
        if not wp.exists():
            continue
        wi = pd.read_csv(wp); full = panel_arrays(panel, "train", False); masked = panel_arrays(panel, "train", True)
        months = np.array([d[:7] for d in full["dates"]]); feb = np.where(months == "2031-02")[0]
        tgt = targets(panel, "train"); prof = profile(full["v"], full["q"], full["dow"], months != "2031-02")
        if RAMP:
            on = onramps(panel, "train", full["dates"]); rp = ramp_profile(on, full["v"], full["m"]["vf"], full["dow"], months != "2031-02")
        sched = {}
        for w in wi.itertuples():
            if int(w.date[8:10]) <= len(feb):
                sched.setdefault(feb[int(w.date[8:10]) - 1], []).append(slot_of(w.forecast_origin))
        for d, origins in sched.items():
            pv, pq = prof[full["dow"][d] >= 5]
            v, q = masked["v"][d].copy(), masked["q"][d].copy()
            for o in origins:
                v[max(o - 11, 0):o + 1] = full["v"][d, max(o - 11, 0):o + 1]; q[max(o - 11, 0):o + 1] = full["q"][d, max(o - 11, 0):o + 1]
            (x, bv, bq, sl, ln, tv, tq), _ = day_rows(v, q, full["m"], full["dow"][d], origins, (full["v"][d], full["q"][d]), pv, pq, on[d] / rp[full["dow"][d] >= 5] if RAMP else None)
            ok = tgt[d][sl, ln] & np.isfinite(tv) & np.isfinite(tq)  # official target cells only
            m = full["m"]; bvf, bqf = np.nan_to_num(bv, nan=pv[sl, ln]), np.nan_to_num(bq, nan=pq[sl, ln])
            pvv = bvf + mv.predict(x) * m["vf"][ln]; pqq = bqf + mq.predict(x) * m["cap"][ln]
            if DUMP:  # per-cell February predictions, for blending with other in-block models (t1_blocks_blend.py)
                dump.append(pd.DataFrame({"panel": panel, "day": full["dates"][d], "slot": sl[ok], "j": ln[ok], "v_gf": pvv[ok], "q_gf": pqq[ok],
                                          "v_true": tv[ok], "q_true": tq[ok], "lanes": m["lanes"][ln[ok]]}))
            for k, (a, b) in (("gapfill", (pvv, pqq)), ("base", (bvf, bqf))):
                se[k][0] += ((a - tv)[ok] ** 2).sum(); se[k][1] += (((b - tq) / m["lanes"][ln])[ok] ** 2).sum(); se[k][2] += ok.sum()
        print(panel, "evaluated", flush=True)
    if DUMP:
        pd.concat(dump).to_csv(DUMP, index=False)
    for k, (sv, sq, n) in se.items():
        rv, rq = np.sqrt(sv / n), np.sqrt(sq / n)
        print(f"IN-BLOCK {k}: n={n} speed RMSE {rv:.3f}  flow/lane RMSE {rq:.2f}  S~{0.54*(1-rv/25)+0.46*(1-rq/600):.4f}  (CatBoost: 7.26 / 89.0 / 0.775)")


def final():
    X, yv, yq, W, names = build_train(lambda mo: True)
    mv, mq = fit(X, yv, W[:, 0]), fit(X, yq, W[:, 1]); del X
    for split in ("validation", "private"):
        parts = []
        for panel in panels():
            wp = RELEASE / "task2" / panel / split / "window_index.csv"
            if not wp.exists():
                continue
            wi = pd.read_csv(wp); arr = panel_arrays(panel, split, True); full = panel_arrays(panel, "train", False)
            prof = profile(full["v"], full["q"], full["dow"], np.ones(len(full["dates"]), bool))
            dates = list(arr["dates"]); links = meta(panel).link_id.to_numpy(); stations = meta(panel).station_id.to_numpy()
            if RAMP:
                rp = ramp_profile(onramps(panel, "train", full["dates"]), full["v"], full["m"]["vf"], full["dow"], np.ones(len(full["dates"]), bool))
                on = onramps(panel, split, dates)
            for date, g in wi.groupby("date"):
                d = dates.index(date); origins = sorted(slot_of(o) for o in g.forecast_origin)
                pv, pq = prof[arr["dow"][d] >= 5]
                (x, bv, bq, sl, ln), _ = day_rows(arr["v"][d], arr["q"][d], arr["m"], arr["dow"][d], origins, None, pv, pq, on[d] / rp[arr["dow"][d] >= 5] if RAMP else None)
                m = arr["m"]; bvf, bqf = np.nan_to_num(bv, nan=pv[sl, ln]), np.nan_to_num(bq, nan=pq[sl, ln])
                parts.append(pd.DataFrame({"panel": panel, "timestamp": [f"{date}T{s // 12:02d}:{(s % 12) * 5:02d}:00Z" for s in sl],
                                           "link_id": links[ln], "speed_kmh": np.clip(bvf + mv.predict(x) * m["vf"][ln], 3, 140),
                                           "flow_vph": np.clip(bqf + mq.predict(x) * m["cap"][ln], 0, None)}))
        pd.concat(parts).drop_duplicates(["panel", "timestamp", "link_id"]).to_csv(CACHE / f"t1_gapfill_{split}.csv", index=False)
        print(split, "written", flush=True)


if __name__ == "__main__":
    {"eval": evaluate, "final": final}[sys.argv[1]]()
