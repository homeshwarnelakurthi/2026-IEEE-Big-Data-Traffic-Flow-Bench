# Source: yeashusemwal, "TrafficFlowBench Trees + U-Net + CNN" (Kaggle notebook, 2026),
# kaggle.com/code/yeashusemwal/trafficflowbench-trees-u-net-cnn-lb-0-87. Redistributed for reproduction with credit; see NOTICE.
"""Task 1: LightGBM space-time imputer trained on train days (masked features -> unmasked truth)."""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.ndimage import convolve
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, ROOT, T, panels, network, day_files, date_of, regime_of, weekday_of, slots_of, day_grid, build_profile

CACHE = ROOT / "submissions" / "profiles"
RATE = {"R1": 0.2, "R2": 0.3, "R3": 0.5}
GAP_AUG = False
RNG = np.random.default_rng(0)

def ffill_idx(obs: np.ndarray, axis: int) -> np.ndarray:
    """index of last observed position <= i along axis (-1 if none)."""
    n = obs.shape[axis]
    ar = np.arange(n).reshape([-1 if a == axis else 1 for a in range(obs.ndim)])
    idx = np.where(obs, ar, -1)
    return np.maximum.accumulate(idx, axis=axis)

def bfill_idx(obs: np.ndarray, axis: int) -> np.ndarray:
    n = obs.shape[axis]
    ar = np.arange(n).reshape([-1 if a == axis else 1 for a in range(obs.ndim)])
    idx = np.where(obs, ar, n)
    return np.flip(np.minimum.accumulate(np.flip(idx, axis=axis), axis=axis), axis=axis)

def take_rows(G, idx, valid):
    """G[m, idx[m,t]] for temporal; returns NaN where invalid."""
    M, Tn = G.shape
    out = G[np.arange(M)[:, None], np.clip(idx, 0, Tn - 1)]
    return np.where(valid, out, np.nan)

def take_cols(G, idx, valid):
    M, Tn = G.shape
    out = G[np.clip(idx, 0, M - 1), np.arange(Tn)[None, :]]
    return np.where(valid, out, np.nan)

def shift(G, dm, dt):
    out = np.full_like(G, np.nan)
    M, Tn = G.shape
    ms, md = (slice(dm, None), slice(0, M - dm)) if dm >= 0 else (slice(0, M + dm), slice(-dm, None))
    ts, td = (slice(dt, None), slice(0, Tn - dt)) if dt >= 0 else (slice(0, Tn + dt), slice(-dt, None))
    out[md, td] = G[ms, ts]
    return out

def win_mean(G, obs, a, b):
    k = np.ones((2 * a + 1, 2 * b + 1))
    s = convolve(np.where(obs, G, 0.0), k, mode="constant", cval=0.0)
    n = convolve(obs.astype(float), k, mode="constant", cval=0.0)
    return np.where(n > 0, s / np.maximum(n, 1), np.nan), n

def features(g: dict, net: pd.DataFrame, prof: dict, weekday: int, regime: str):
    """Compute feature stack (M,T,F) from a masked day grid."""
    V, Q = g["v"], g["q"]
    lanes = net.lanes.to_numpy()[:, None]
    Ql = Q / lanes
    M = V.shape[0]
    obs = np.isfinite(V) & np.isfinite(Q)
    Vn = np.where(obs, V, np.nan); Qn = np.where(obs, Ql, np.nan)
    hv = prof["v"][weekday]; hq = prof["q"][weekday] / lanes; hvs = prof["v_std"][weekday]; hqf = prof["qfrac"][weekday]
    F = {}
    F["hist_v"] = hv; F["hist_q"] = hq; F["hist_vstd"] = hvs; F["hist_qfrac"] = hqf
    tt = np.broadcast_to(np.arange(T)[None, :], (M, T)).astype(np.float32)
    # temporal neighbours (strict)
    obs_prev = np.concatenate([np.zeros((M, 1), bool), obs[:, :-1]], axis=1)
    ip = ffill_idx(obs_prev, 1) - 1; vp = ip >= 0
    obs_next = np.concatenate([obs[:, 1:], np.zeros((M, 1), bool)], axis=1)
    inx = bfill_idx(obs_next, 1) + 1; vn = inx < T
    F["v_prev"] = take_rows(Vn, ip, vp); F["q_prev"] = take_rows(Qn, ip, vp); F["gap_prev"] = np.where(vp, tt - ip, 99)
    F["v_next"] = take_rows(Vn, inx, vn); F["q_next"] = take_rows(Qn, inx, vn); F["gap_next"] = np.where(vn, inx - tt, 99)
    F["hv_prev"] = take_rows(hv, ip, vp); F["hv_next"] = take_rows(hv, inx, vn)
    F["hq_prev"] = take_rows(hq, ip, vp); F["hq_next"] = take_rows(hq, inx, vn)
    with np.errstate(all="ignore"): wp = F["gap_next"] / (F["gap_prev"] + F["gap_next"])
    F["v_interp"] = np.where(vp & vn, wp * F["v_prev"] + (1 - wp) * F["v_next"], np.where(vp, F["v_prev"], F["v_next"]))
    F["q_interp"] = np.where(vp & vn, wp * F["q_prev"] + (1 - wp) * F["q_next"], np.where(vp, F["q_prev"], F["q_next"]))
    F["dev_v_interp"] = np.where(vp & vn, wp * (F["v_prev"] - F["hv_prev"]) + (1 - wp) * (F["v_next"] - F["hv_next"]),
                                 np.where(vp, F["v_prev"] - F["hv_prev"], F["v_next"] - F["hv_next"]))
    F["dev_q_interp"] = np.where(vp & vn, wp * (F["q_prev"] - F["hq_prev"]) + (1 - wp) * (F["q_next"] - F["hq_next"]),
                                 np.where(vp, F["q_prev"] - F["hq_prev"], F["q_next"] - F["hq_next"]))
    # spatial neighbours (strict) at same t
    mm = np.broadcast_to(np.arange(M)[:, None], (M, T)).astype(np.float32)
    obs_up = np.concatenate([np.zeros((1, T), bool), obs[:-1, :]], axis=0)
    iu = ffill_idx(obs_up, 0) - 1; vu = iu >= 0
    obs_dn = np.concatenate([obs[1:, :], np.zeros((1, T), bool)], axis=0)
    idn = bfill_idx(obs_dn, 0) + 1; vd = idn < M
    F["v_up"] = take_cols(Vn, iu, vu); F["q_up"] = take_cols(Qn, iu, vu); F["gap_up"] = np.where(vu, mm - iu, 99)
    F["v_dn"] = take_cols(Vn, idn, vd); F["q_dn"] = take_cols(Qn, idn, vd); F["gap_dn"] = np.where(vd, idn - mm, 99)
    F["hv_up"] = take_cols(hv, iu, vu); F["hv_dn"] = take_cols(hv, idn, vd)
    F["hq_up"] = take_cols(hq, iu, vu); F["hq_dn"] = take_cols(hq, idn, vd)
    # raw diagonal / adjacent cells
    for dm, dt in ((-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1), (-2, 1), (2, -1), (0, -2), (0, 2)):
        F[f"v_{dm}_{dt}"] = shift(Vn, dm, dt); F[f"q_{dm}_{dt}"] = shift(Qn, dm, dt)
    # window means
    for a, b in ((1, 1), (2, 3), (4, 6), (8, 12)):
        mv, n = win_mean(Vn, obs, a, b); mq, _ = win_mean(Qn, obs, a, b)
        F[f"wv_{a}_{b}"] = mv; F[f"wq_{a}_{b}"] = mq; F[f"wn_{a}_{b}"] = n
        mdv, _ = win_mean(Vn - hv, obs, a, b); F[f"wdv_{a}_{b}"] = mdv
        mdq, _ = win_mean(Qn - hq, obs, a, b); F[f"wdq_{a}_{b}"] = mdq
    # static
    F["vf"] = np.broadcast_to(net.free_speed_kmh.to_numpy()[:, None], (M, T))
    F["cap_lane"] = np.broadcast_to((net.capacity_vph / net.lanes).to_numpy()[:, None], (M, T))
    F["lanes"] = np.broadcast_to(lanes, (M, T)); F["length"] = np.broadcast_to(net.length_km.to_numpy()[:, None], (M, T))
    F["pos"] = mm / M; F["M"] = np.full((M, T), M, np.float32)
    F["slot"] = tt; F["weekday"] = np.full((M, T), weekday, np.float32); F["rate"] = np.full((M, T), RATE.get(regime, 0.3), np.float32)
    F["ramp_on"] = np.broadcast_to(net.ramp_on.to_numpy()[:, None], (M, T)); F["ramp_off"] = np.broadcast_to(net.ramp_off.to_numpy()[:, None], (M, T))
    names = list(F.keys())
    X = np.stack([np.asarray(F[k], dtype=np.float32) for k in names], axis=-1)
    return X, names

def train_rows(panel: str, net, prof, dates_keep, cache_dir):
    li = {l: i for i, l in enumerate(net.link_id)}; M = len(net)
    truth = {date_of(p): p for p in day_files(panel, "train", masked=False)}
    Xs, ys, meta = [], [], []
    for mp in day_files(panel, "train", masked=True):
        d = date_of(mp)
        if d not in dates_keep or d not in truth: continue
        g = day_grid(mp, li, M); gt = day_grid(truth[d], li, M)
        if GAP_AUG and RNG.random() < 0.6:  # emulate Task 2 horizon+buffer blanks (18 slots, all links) at congested times
            vc = net.v_cut.to_numpy()[:, None]; qt = np.nonzero(((gt["v"] <= vc) & np.isfinite(gt["v"])).sum(0) > 0)[0]; qt = qt[(qt > 20) & (qt < 268)]
            if len(qt) >= 2:
                for st in RNG.choice(qt, RNG.integers(1, 3), replace=False):
                    g["v"][:, st:st + 18] = np.nan; g["q"][:, st:st + 18] = np.nan
        X, names = features(g, net, prof, weekday_of(d), regime_of(mp))
        tgt = ~(np.isfinite(g["v"]) & np.isfinite(g["q"])) & np.isfinite(gt["v"]) & np.isfinite(gt["q"]) & gt["elig"]
        mi, ti = np.nonzero(tgt)
        Xs.append(X[mi, ti]); ys.append(np.stack([gt["v"][mi, ti], gt["q"][mi, ti] / net.lanes.to_numpy()[mi]], 1))
        meta.append(np.stack([np.full(len(mi), RATE[regime_of(mp)]), np.full(len(mi), hash(panel) % 1000)], 1))
    return (np.concatenate(Xs), np.concatenate(ys), np.concatenate(meta), names) if Xs else (None, None, None, None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-step", type=int, default=12)
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--out", type=Path, default=ROOT / "submissions" / "task1_v2")
    ap.add_argument("--models", type=Path, default=ROOT / "models")
    ap.add_argument("--panel", action="append")
    ap.add_argument("--skip-train", action="store_true"); ap.add_argument("--gap-aug", action="store_true")
    a = ap.parse_args()
    global GAP_AUG; GAP_AUG = a.gap_aug
    import lightgbm as lgb
    ps = a.panel or panels()
    a.out.mkdir(parents=True, exist_ok=True); a.models.mkdir(parents=True, exist_ok=True)
    nets = {p: network(p) for p in ps}; profs = {p: build_profile(p, nets[p], CACHE) for p in ps}
    models = {}
    if not a.skip_train:
        t0 = time.time(); Xtr, ytr, Xva, yva, mva = [], [], [], [], []
        for p in ps:
            all_dates = sorted(date_of(x) for x in day_files(p, "train", masked=True))
            hold = set(all_dates[-20:]); train_dates = set(all_dates[:-20][::a.train_step])
            va_dates = set(sorted(hold)[::4])
            X, y, m, names = train_rows(p, nets[p], profs[p], train_dates, CACHE); Xtr.append(X); ytr.append(y)
            X, y, m, names = train_rows(p, nets[p], profs[p], va_dates, CACHE); Xva.append(X); yva.append(y); mva.append(m)
            print(f"{p}: train rows {sum(len(x) for x in Xtr):,} ({time.time()-t0:.0f}s)", flush=True)
        Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr); Xva = np.concatenate(Xva); yva = np.concatenate(yva); mva = np.concatenate(mva)
        print("train", Xtr.shape, "valid", Xva.shape, flush=True)
        params = dict(objective="regression", learning_rate=0.05, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.7,
                      bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=10, verbose=-1, max_bin=255)
        for j, name in enumerate(("speed", "flow")):
            dtr = lgb.Dataset(Xtr, ytr[:, j], feature_name=names, free_raw_data=False); dva = lgb.Dataset(Xva, yva[:, j], reference=dtr)
            bst = lgb.train(params, dtr, a.rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
            bst.save_model(str(a.models / f"task1_{name}.txt")); models[name] = bst
            pred = bst.predict(Xva, num_iteration=bst.best_iteration)
            for r in (0.2, 0.3, 0.5):
                s = mva[:, 0] == r; rm = np.sqrt(np.mean((pred[s] - yva[s, j]) ** 2))
                print(f"  {name} R{r}: RMSE {rm:.3f}  S={max(0,1-rm/(25 if j==0 else 600)):.4f}", flush=True)
            base = Xva[:, names.index("v_interp" if j == 0 else "q_interp")]; ok = np.isfinite(base)
            print(f"  interp baseline RMSE {np.sqrt(np.nanmean((base[ok]-yva[ok,j])**2)):.3f}; hist RMSE {np.sqrt(np.mean((Xva[:,names.index('hist_v' if j==0 else 'hist_q')]-yva[:,j])**2)):.3f}")
            imp = sorted(zip(bst.feature_importance("gain"), names), reverse=True)[:15]; print("  top:", [(n, int(g)) for g, n in imp])
    else:
        for name in ("speed", "flow"): models[name] = lgb.Booster(model_file=str(a.models / f"task1_{name}.txt"))
    # predict
    for split in ("validation", "private"):
        outs = []
        for p in ps:
            net = nets[p]; li = {l: i for i, l in enumerate(net.link_id)}; M = len(net); lanes = net.lanes.to_numpy()
            tpl = pd.read_csv(DATA / "task1" / p / split / "sample_submission_state.csv", dtype=str)
            d_arr = tpl.timestamp.str[:10].to_numpy(); slot = slots_of(tpl.timestamp); lidx = tpl.link_id.map(li).to_numpy(dtype=int)
            pv = np.zeros(len(tpl), np.float32); pq = np.zeros(len(tpl), np.float32)
            for mp in day_files(p, split, masked=True):
                d = date_of(mp); rows = np.nonzero(d_arr == d)[0]
                if len(rows) == 0: continue
                g = day_grid(mp, li, M); X, names = features(g, net, profs[p], weekday_of(d), regime_of(mp))
                Xr = X[lidx[rows], slot[rows]]
                pv[rows] = models["speed"].predict(Xr); pq[rows] = models["flow"].predict(Xr) * lanes[lidx[rows]]
            vf = net.free_speed_kmh.to_numpy()[lidx]
            tpl["speed_kmh"] = np.clip(pv, 5.0, vf * 1.08); tpl["flow_vph"] = np.clip(pq, 100.0, None)
            outs.append(tpl); print(f"[{split}] {p} done", flush=True)
        out = pd.concat(outs, ignore_index=True); out.to_csv(a.out / f"task1_{split}.csv", index=False); print("wrote", a.out / f"task1_{split}.csv", len(out))

if __name__ == "__main__":
    main()
