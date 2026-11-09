# Source: yeashusemwal, "TrafficFlowBench Trees + U-Net + CNN" (Kaggle notebook, 2026),
# kaggle.com/code/yeashusemwal/trafficflowbench-trees-u-net-cnn-lb-0-87. Redistributed for reproduction with credit; see NOTICE.
"""Task 2: LightGBM queue forecaster trained on windows sampled from the unmasked train split."""
from __future__ import annotations
import argparse, sys, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, ROOT, T, panels, network, day_files, date_of, weekday_of, slots_of, day_grid, build_profile

CACHE = ROOT / "submissions" / "profiles"
H = 6; HIST = 12
Q2_PANELS = ["D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N", "D7_I405_S", "D12_I5_N", "D12_I5_S"]

def nan_shift(a, k):
    """shift along axis 0 by k (positive: value from index m-k)."""
    out = np.full_like(a, np.nan); 
    if k > 0: out[k:] = a[:-k]
    elif k < 0: out[:k] = a[-k:]
    else: out[:] = a
    return out

def window_features(Hv, Hq, Ho, net, prof, t0, weekday, onset):
    """Hv,Hq,Ho: (M,12) history ending at t0 (index 11 == t0). Returns X (M,H,F), names."""
    M = Hv.shape[0]
    vf = net.free_speed_kmh.to_numpy(); capl = (net.capacity_vph / net.lanes).to_numpy()
    R = Hv / vf[:, None]; Fq = Hq / capl[:, None]
    obs = np.isfinite(R)
    F = {}
    for k in range(6): F[f"r_{k}"] = R[:, -1 - k]
    F["r_mean_6_11"] = np.nanmean(R[:, :6], axis=1)
    F["r_min"] = np.nanmin(np.where(obs, R, np.inf), axis=1); F["r_min"][~obs.any(1)] = np.nan
    F["r_last"] = np.where(obs.any(1), R[np.arange(M), np.where(obs, np.arange(HIST), -1).max(1).clip(0)], np.nan)
    F["age_last"] = HIST - 1 - np.where(obs, np.arange(HIST), -1).max(1)
    F["r_slope"] = np.nanmean(R[:, -3:], axis=1) - np.nanmean(R[:, -6:-3], axis=1)
    for k in range(3): F[f"fq_{k}"] = Fq[:, -1 - k]; F[f"occ_{k}"] = Ho[:, -1 - k]
    F["fq_mean"] = np.nanmean(Fq, axis=1); F["occ_max"] = np.nanmax(np.where(np.isfinite(Ho), Ho, -np.inf), axis=1); F["occ_max"][~np.isfinite(Ho).any(1)] = np.nan
    r0 = R[:, -1]; r1 = R[:, -2]; rl = F["r_last"]
    for off in (-4, -3, -2, -1, 1, 2, 3, 4):
        F[f"r0_n{off}"] = nan_shift(rl, -off); F[f"r1_n{off}"] = nan_shift(r1, -off)
    dn = np.stack([nan_shift(rl, -k) for k in range(1, 7)], 1); up = np.stack([nan_shift(rl, k) for k in range(1, 7)], 1)
    F["min_dn6"] = np.nanmin(np.where(np.isfinite(dn), dn, np.inf), 1); F["min_dn6"][~np.isfinite(dn).any(1)] = np.nan
    F["min_up6"] = np.nanmin(np.where(np.isfinite(up), up, np.inf), 1); F["min_up6"][~np.isfinite(up).any(1)] = np.nan
    F["mean_dn3"] = np.nanmean(dn[:, :3], 1); F["mean_up3"] = np.nanmean(up[:, :3], 1)
    queued0 = rl <= 0.6; queued1 = r1 <= 0.6
    F["frac_q0"] = np.full(M, np.nanmean(queued0[np.isfinite(rl)]) if np.isfinite(rl).any() else 0)
    F["frac_q1"] = np.full(M, np.nanmean(queued1[np.isfinite(r1)]) if np.isfinite(r1).any() else 0)
    F["cor_min0"] = np.full(M, np.nanmin(rl) if np.isfinite(rl).any() else np.nan)
    F["cor_mean0"] = np.full(M, np.nanmean(rl)); F["cor_mean_hist"] = np.full(M, np.nanmean(R))
    # distance to nearest queued link downstream / upstream at t0 and t0-1
    idx = np.arange(M)
    def dist_to(q):
        qi = np.nonzero(q)[0]
        if len(qi) == 0: return np.full(M, 99.0), np.full(M, 99.0)
        pos = np.searchsorted(qi, idx)  # first queued index >= m
        dn = np.where(pos < len(qi), qi[np.clip(pos, 0, len(qi) - 1)] - idx, 99)
        up = np.where(pos > 0, idx - qi[np.clip(pos - 1, 0, len(qi) - 1)], 99)
        return dn.astype(float), up.astype(float)
    F["dq_dn0"], F["dq_up0"] = dist_to(queued0 & np.isfinite(rl)); F["dq_dn1"], F["dq_up1"] = dist_to(queued1 & np.isfinite(r1))
    # slow (r<=0.8) proximity as pre-congestion signal
    F["ds_dn0"], F["ds_up0"] = dist_to((rl <= 0.8) & np.isfinite(rl))
    hv = prof["v"][weekday] / vf[:, None]; hqf = prof["qfrac"][weekday]; hvs = prof["v_std"][weekday] / vf[:, None]
    F["hist_r_t0"] = hv[:, t0]; F["hist_qf_t0"] = hqf[:, t0]; F["dev_r_t0"] = rl - hv[:, t0 - 1]; F["hist_qf_tm1"] = hqf[:, t0 - 1]
    F["hist_qf_hist_mean"] = hqf[:, max(0, t0 - HIST):t0].mean(1)
    F["hist_qf_max_next12"] = hqf[:, t0 + 1:min(T, t0 + 13)].max(1)
    F["hist_r_dn1_t0"] = nan_shift(hv[:, t0], -1); F["hist_r_up1_t0"] = nan_shift(hv[:, t0], 1)
    if EXTRA_FEATS:
        # propagation dynamics: speed-ratio change over last 3 steps at link and neighbours, slow-link counts up/downstream
        d3 = R[:, -1] - R[:, -4]; d3l = np.where(np.isfinite(d3), d3, np.nanmean(R[:, -2:], 1) - np.nanmean(R[:, -5:-3], 1))
        F["d3"] = d3l
        for off in (-3, -2, -1, 1, 2, 3): F[f"d3_n{off}"] = nan_shift(d3l, -off)
        slow = (rl <= 0.8) & np.isfinite(rl); cs = np.cumsum(slow)
        F["n_slow_dn10"] = np.concatenate([cs[10:] - cs[:-10], np.full(10, np.nan)]) if M > 10 else np.full(M, np.nan)
        F["n_slow_up10"] = np.concatenate([np.full(10, np.nan), cs[10:] - cs[:-10]]) if M > 10 else np.full(M, np.nan)
        F["min_dn12"] = np.nanmin(np.where(np.isfinite(np.stack([nan_shift(rl, -k) for k in range(1, 13)], 1)), np.stack([nan_shift(rl, -k) for k in range(1, 13)], 1), np.inf), 1)
        F["min_dn12"][~np.isfinite(F["min_dn12"])] = np.nan
        F["r_min_hist_dn3"] = np.nanmin(np.stack([nan_shift(np.nanmin(R, 1), -k) for k in range(1, 4)], 1), 1)
        F["occ_slope"] = np.nanmean(Ho[:, -3:], 1) - np.nanmean(Ho[:, -6:-3], 1)
        F["frac_slow0"] = np.full(M, np.nanmean(rl <= 0.8))
    if NO_HIST:
        for k in [k for k in F if k.startswith("hist_") or k.startswith("dev_")]: F.pop(k)
    F["vf"] = vf; F["cap_lane"] = capl; F["lanes"] = net.lanes.to_numpy(); F["pos"] = idx / M; F["M"] = np.full(M, M)
    F["ramp_on"] = net.ramp_on.to_numpy(); F["ramp_off"] = net.ramp_off.to_numpy(); F["length"] = net.length_km.to_numpy()
    F["slot"] = np.full(M, t0); F["weekday"] = np.full(M, weekday); F["onset"] = np.full(M, float(onset))
    base_names = list(F.keys()); base = np.stack([np.asarray(F[k], np.float32) for k in base_names], 1)  # (M,Fb)
    per_h = []
    for h in range(1, H + 1):
        th = min(T - 1, t0 + h)
        ph = np.stack([np.full(M, h, np.float32), hv[:, th], hqf[:, th], hvs[:, th], nan_shift(hqf[:, th], -1), nan_shift(hqf[:, th], 1),
                       nan_shift(hqf[:, th], -2), nan_shift(hqf[:, th], 2), hqf[:, th] - hqf[:, t0], rl - hv[:, th]], 1)
        if NO_HIST: ph = ph[:, :1]
        per_h.append(np.concatenate([base, ph], 1))
    names = base_names + (["h"] if NO_HIST else ["h", "hist_r_th", "hist_qf_th", "hist_vstd_th", "hist_qf_th_dn1", "hist_qf_th_up1", "hist_qf_th_dn2", "hist_qf_th_up2", "hist_qf_delta", "dev_r_th"])
    return np.stack(per_h, 1), names

DENOISE = False
EXTRA_FEATS = False
NO_HIST = False

def denoise(V):
    """3-point temporal nan-median per link: approximates the noise-free state used for the true labels."""
    Vp = np.concatenate([V[:, :1], V[:, :-1]], 1); Vn = np.concatenate([V[:, 1:], V[:, -1:]], 1)
    S = np.stack([Vp, V, Vn], 0); med = np.nanmedian(S, axis=0)
    return np.where(np.isfinite(V), med, np.nan)

def sample_train_windows(panel, net, prof, dates, rng, per_day=2):
    li = {l: i for i, l in enumerate(net.link_id)}; M = len(net); vcut = net.v_cut.to_numpy()[:, None]
    Xs, ys, ws, es = [], [], [], []
    wid = 0
    for p in day_files(panel, "train", masked=False):
        d = date_of(p)
        if d not in dates: continue
        g = day_grid(p, li, M, with_occ=True); V = g["v"]; L = (denoise(V) if DENOISE else V) <= vcut; obs = np.isfinite(V)
        anyq = (L & obs).any(0)  # per t: some link queued (observed)
        wd = weekday_of(d)
        # history = t0-12 .. t0-1 (T-60..T-5), horizon = t0+1 .. t0+6 (T+5..T+30); T itself is never observed
        cands = np.arange(HIST, T - H)
        hist_any = np.array([anyq[t - HIST:t].any() for t in cands]); fut_any = np.array([anyq[t + 1:t + H + 1].any() for t in cands])
        onset_c = cands[~hist_any & fut_any]; ongoing_c = cands[anyq[cands - 1] & fut_any]
        for cond, cc in ((1, onset_c), (0, ongoing_c)):
            if len(cc) == 0: continue
            picks = []
            for t0 in rng.permutation(cc):
                if all(abs(t0 - q) >= 36 for q in picks): picks.append(t0)
                if len(picks) >= per_day: break
            for t0 in picks:
                X, names = window_features(V[:, t0 - HIST:t0], g["q"][:, t0 - HIST:t0], g["occ"][:, t0 - HIST:t0], net, prof, t0, wd, cond)
                y = L[:, t0 + 1:t0 + H + 1].astype(np.float32); e = (obs & g["elig"])[:, t0 + 1:t0 + H + 1]
                Xs.append(X.reshape(M * H, -1)); ys.append(y.reshape(-1)); es.append(e.reshape(-1)); ws.append(np.stack([np.full(M * H, wid), np.full(M * H, cond)], 1)); wid += 1
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(es), np.concatenate(ws), names

def iou_by_window(pred_bin, y, e, w):
    out = {}
    for wid in np.unique(w[:, 0]):
        s = (w[:, 0] == wid) & e
        p = pred_bin[s] > 0; t = y[s] > 0.5
        u = (p | t).sum(); out[wid] = (1.0 if u == 0 else (p & t).sum() / u, w[s, 1][0] if s.any() else -1)
    return out

def best_threshold(prob, y, e, w, cond):
    best = (0, 0.5)
    for th in np.arange(0.15, 0.86, 0.025):
        r = iou_by_window(prob > th, y, e, w); v = [x for x, c in r.values() if c == cond]
        if v and np.mean(v) > best[0]: best = (np.mean(v), th)
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-step", type=int, default=2)
    ap.add_argument("--per-day", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=ROOT / "submissions" / "task2_v2")
    ap.add_argument("--models", type=Path, default=ROOT / "models")
    ap.add_argument("--skip-train", action="store_true"); ap.add_argument("--denoise", action="store_true"); ap.add_argument("--onset-mink", type=int, default=0); ap.add_argument("--splits", nargs="+", default=["validation", "private"]); ap.add_argument("--exclude-org", action="store_true"); ap.add_argument("--extra", action="store_true"); ap.add_argument("--no-hist", action="store_true"); ap.add_argument("--window-weight", action="store_true")
    a = ap.parse_args()
    global DENOISE, EXTRA_FEATS, NO_HIST; DENOISE = a.denoise; EXTRA_FEATS = a.extra; NO_HIST = a.no_hist
    import lightgbm as lgb
    a.out.mkdir(parents=True, exist_ok=True); a.models.mkdir(parents=True, exist_ok=True)
    nets = {p: network(p) for p in Q2_PANELS}; profs = {p: build_profile(p, nets[p], CACHE) for p in Q2_PANELS}
    rng = np.random.default_rng(0)
    th_file = a.models / "task2_thresholds.txt"
    if not a.skip_train:
        t0 = time.time(); TR = []; VA = []
        for p in Q2_PANELS:
            dates = sorted(date_of(x) for x in day_files(p, "train", masked=False))
            excl = set(pd.read_csv(DATA / "task2" / p / "train" / "window_index.csv").date.astype(str)) if a.exclude_org else set()
            hold = [d for d in dates[-24:] if d not in excl]; tr = [d for d in dates[:-24] if d not in excl][::a.train_step]
            TR.append(sample_train_windows(p, nets[p], profs[p], set(tr), rng, a.per_day)); VA.append(sample_train_windows(p, nets[p], profs[p], set(hold), rng, 1))
            print(f"{p}: {TR[-1][0].shape} rows ({time.time()-t0:.0f}s)", flush=True)
        names = TR[0][4]
        def cat(L):
            X = np.concatenate([x[0] for x in L]); y = np.concatenate([x[1] for x in L]); e = np.concatenate([x[2] for x in L])
            w = []; off = 0
            for x in L: ww = x[3].copy(); ww[:, 0] += off; off = ww[:, 0].max() + 1; w.append(ww)
            return X, y, e, np.concatenate(w)
        Xtr, ytr, etr, wtr = cat(TR); Xva, yva, eva, wva = cat(VA)
        print("train", Xtr.shape, "labeled", etr.sum(), "pos rate", ytr[etr].mean(), "valid", Xva.shape, flush=True)
        params = dict(objective="binary", learning_rate=0.03, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=10, verbose=-1)
        lab = np.isfinite(ytr) & etr
        wts = None
        if a.window_weight:
            # per-window weight ~ 1/sqrt(n_true_cells + 5): the metric is a per-window IoU, so small-queue windows matter as much as big ones
            wid = wtr[:, 0]; npos = pd.Series(np.where(lab, ytr, 0)).groupby(wid).transform("sum").to_numpy()
            wts = (1.0 / np.sqrt(npos + 5.0))[lab]; wts *= len(wts) / wts.sum()
        dtr = lgb.Dataset(Xtr[lab], ytr[lab], weight=wts, feature_name=names); labv = np.isfinite(yva) & eva; dva = lgb.Dataset(Xva[labv], yva[labv], reference=dtr)
        bst = lgb.train(params, dtr, a.rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
        bst.save_model(str(a.models / "task2_queue.txt"))
        prob = bst.predict(Xva, num_iteration=bst.best_iteration)
        ths = {}
        for cond, nm in ((1, "onset"), (0, "ongoing")):
            sc, th = best_threshold(prob, yva, eva & np.isfinite(yva), wva, cond); ths[nm] = th
            print(f"  {nm}: best IoU {sc:.4f} at th {th:.3f}", flush=True)
        # persistence baseline on valid
        pers = (Xva[:, names.index("r_last")] <= 0.6).astype(float); r = iou_by_window(pers, yva, eva & np.isfinite(yva), wva)
        for cond, nm in ((1, "onset"), (0, "ongoing")): print(f"  persistence {nm}: {np.mean([x for x, c in r.values() if c == cond]):.4f}")
        imp = sorted(zip(bst.feature_importance("gain"), names), reverse=True)[:20]; print("  top:", [(n, int(g)) for g, n in imp])
        th_file.write_text(f"{ths['onset']} {ths['ongoing']}")
    else:
        bst = lgb.Booster(model_file=str(a.models / "task2_queue.txt")); names = bst.feature_name()
    th_on, th_ong = map(float, th_file.read_text().split())
    for split in a.splits:
        outs = []
        for p in Q2_PANELS:
            net = nets[p]; li = {l: i for i, l in enumerate(net.link_id)}; M = len(net)
            wi = pd.read_csv(DATA / "task2" / p / split / "window_index.csv"); hist = pd.read_parquet(DATA / "task2" / p / split / "window_history.parquet")
            tpl = pd.read_csv(DATA / "task2" / p / split / "sample_submission_queue.csv", dtype=str)
            tpl_pred = np.zeros(len(tpl), np.int64)
            for w in wi.itertuples(index=False):
                hw = hist[hist.window_id == w.window_id]; ts = sorted(hw.timestamp.unique()); tix = {t: i for i, t in enumerate(ts)}
                Hv = np.full((M, HIST), np.nan, np.float32); Hq = Hv.copy(); Ho = Hv.copy()
                li_ = hw.link_id.map(li).to_numpy(); ok = ~pd.isna(li_); ti_ = hw.timestamp.map(tix).to_numpy()
                Hv[li_[ok].astype(int), ti_[ok]] = hw.speed_kmh.to_numpy()[ok]; Hq[li_[ok].astype(int), ti_[ok]] = hw.flow_vph.to_numpy()[ok]; Ho[li_[ok].astype(int), ti_[ok]] = hw.occupancy.to_numpy()[ok]
                t0 = int(slots_of(pd.Series([w.forecast_origin]))[0]); wd = weekday_of(w.date); onset = int(w.condition == "queue_onset")
                X, _ = window_features(Hv, Hq, Ho, net, profs[p], t0, wd, onset)
                prob = bst.predict(X.reshape(M * H, -1), num_iteration=bst.best_iteration).reshape(M, H)
                pred = (prob > (th_on if onset else th_ong)).astype(int)
                if onset and a.onset_mink and pred.sum() == 0:
                    pred = (prob >= np.sort(prob.ravel())[-a.onset_mink]).astype(int)
                rows = np.nonzero((tpl.window_id == w.window_id).to_numpy())[0]
                fts = sorted(tpl.timestamp.iloc[rows].unique()); fix = {t: i for i, t in enumerate(fts)}
                lr = tpl.link_id.iloc[rows].map(li).to_numpy(); hr = tpl.timestamp.iloc[rows].map(fix).to_numpy()
                okr = ~pd.isna(lr); tpl_pred[rows[okr]] = pred[lr[okr].astype(int), hr[okr]]
            tpl["queue_pred"] = tpl_pred; outs.append(tpl); print(f"[{split}] {p}: mean pred {tpl_pred.mean():.3f}", flush=True)
        out = pd.concat(outs, ignore_index=True); out.to_csv(a.out / f"task2_{split}.csv", index=False); print("wrote", len(out))

if __name__ == "__main__":
    main()
