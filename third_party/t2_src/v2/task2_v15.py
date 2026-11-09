# Source: yeashusemwal, "TrafficFlowBench Trees + U-Net + CNN" (Kaggle notebook, 2026),
# kaggle.com/code/yeashusemwal/trafficflowbench-trees-u-net-cnn-lb-0-87. Redistributed for reproduction with credit; see NOTICE.
"""Task 2 v15 = v11 (LOO profiles, origin column T) + long-context features.
The organizer ruling (discussion 742068) allows any released data timestamped <= T. v11 used only the 60-minute
window history, so it could not see how old a queue was, how fast its tail was moving, or what the same day looked
like hours earlier. This adds, all from data <= T:
  * three hours of prior state from the split's own mainline_states_masked (blocks of means, queued fraction),
  * queue geometry over the history: tail/head order index, queue length, and their linear slopes (shockwave speed),
  * per-link queue age and time since the queue cleared,
  * ramp flows aggregated to the attached mainline link (last value and one-hour mean, per lane).
"""
import os, sys, time, json, argparse, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import task2_v10 as v10, task2_v11 as v11
from common import DATA, T, network, day_files, date_of, weekday_of, slots_of, day_grid

H = 6; HV = 13; EXT = 36  # three hours of prior context

def ramp_day(panel, split, date, net):
    """On/off ramp flow per mainline link and timestep (M, 288); NaN where unobserved."""
    M = len(net); li = {l: i for i, l in enumerate(net.link_id)}
    amap = pd.read_csv(DATA / "corridors" / panel / "network" / "ramp_attachment_map.csv", dtype=str)
    on = np.full((M, T), np.nan, np.float32); off = np.full((M, T), np.nan, np.float32)
    files = [f for f in (DATA / "corridors" / panel / split / "ramp_states").rglob("*.parquet") if date.replace("-", "_") in f.name]
    if not files: return on, off
    f = pd.read_parquet(files[0], columns=["timestamp", "ramp_link_id", "ramp_type", "flow_vph", "is_score_eligible"])
    f = f[f.is_score_eligible.astype(bool) & f.flow_vph.notna()]
    if f.empty: return on, off
    f = f.merge(amap.rename(columns={"nearest_mainline_link_id": "link_id"})[["ramp_link_id", "link_id"]], on="ramp_link_id", how="inner")
    f["mi"] = f.link_id.map(li); f = f[f.mi.notna()]; f["ti"] = slots_of(f.timestamp)
    for arr, kind in ((on, ("OR", "ON")), (off, ("FR", "OFF"))):
        s = f[f.ramp_type.str.upper().isin(kind)].groupby(["mi", "ti"], as_index=False).flow_vph.sum()
        if len(s): arr[s.mi.to_numpy().astype(int), s.ti.to_numpy().astype(int)] = s.flow_vph.to_numpy()
    return on, off

def long_context(net, Hv, ext_v, on, off, t0):
    """Per-link long-context features (M, F2). ext_v: (M, EXT+1) masked speeds T-180..T; Hv: (M, HV) history."""
    M = len(net); vc = net.v_cut.to_numpy()[:, None]; lanes = net.lanes.to_numpy(); idx = np.arange(M)
    F = {}
    R = ext_v / net.free_speed_kmh.to_numpy()[:, None]
    for i, sl in enumerate((slice(0, 12), slice(12, 24), slice(24, EXT + 1))):
        F[f"ext_r_mean_b{i}"] = np.nanmean(R[:, sl], 1); F[f"ext_q_frac_b{i}"] = np.nanmean((ext_v[:, sl] <= vc), 1)
    F["ext_r_min"] = np.nanmin(np.where(np.isfinite(R), R, np.inf), 1); F["ext_r_min"][~np.isfinite(F["ext_r_min"])] = np.nan
    F["ext_obs_frac"] = np.isfinite(ext_v).mean(1)
    # queued series over history (NaN-tolerant): carry the last observed state forward
    Q = np.where(np.isfinite(Hv), Hv <= vc, np.nan)
    Qf = pd.DataFrame(Q).ffill(axis=1).bfill(axis=1).to_numpy()
    age = np.zeros(M); since = np.full(M, HV, float)
    for m in range(M):
        s = Qf[m]
        if np.isnan(s).all(): age[m] = np.nan; since[m] = np.nan; continue
        k = HV - 1
        while k >= 0 and s[k] == 1: age[m] += 1; k -= 1
        if s[-1] == 1: since[m] = 0
        else:
            j = HV - 1
            while j >= 0 and s[j] == 0: j -= 1
            since[m] = HV - 1 - j if j >= 0 else HV
    F["queue_age"] = age; F["since_queue"] = since
    # corridor geometry per history step: tail (min index), head (max index), count
    tail = np.full(HV, np.nan); head = np.full(HV, np.nan); cnt = np.zeros(HV)
    for k in range(HV):
        q = np.nonzero(Qf[:, k] == 1)[0]; cnt[k] = len(q)
        if len(q): tail[k] = q.min(); head[k] = q.max()
    def slope(a, n=6):
        y = a[-n:]; ok = np.isfinite(y)
        return np.polyfit(np.arange(n)[ok], y[ok], 1)[0] if ok.sum() >= 2 else np.nan
    g = {"cur_tail": tail[-1], "cur_head": head[-1], "cur_count": cnt[-1], "tail_slope": slope(tail), "head_slope": slope(head),
         "count_slope": slope(cnt), "count_slope_long": slope(cnt, HV), "count_1h_ago": cnt[0],
         "ext_count_3h": float(np.nansum(ext_v[:, 0] <= vc[:, 0])), "ext_count_2h": float(np.nansum(ext_v[:, 12] <= vc[:, 0]))}
    for k, val in g.items(): F[k] = np.full(M, val if np.isfinite(val) else np.nan)
    F["dist_to_tail"] = idx - g["cur_tail"]; F["dist_to_head"] = idx - g["cur_head"]
    F["tail_proj_h6"] = np.full(M, g["cur_tail"] + 6 * (g["tail_slope"] if np.isfinite(g["tail_slope"]) else 0.0))
    F["dist_to_tail_proj"] = idx - F["tail_proj_h6"]
    # ramp flows up to T (per lane)
    w = slice(max(0, t0 - 11), t0 + 1)
    F["ramp_on_last"] = on[:, t0] / lanes; F["ramp_off_last"] = off[:, t0] / lanes
    F["ramp_on_mean1h"] = np.nanmean(on[:, w], 1) / lanes; F["ramp_off_mean1h"] = np.nanmean(off[:, w], 1) / lanes
    F["ramp_on_trend"] = (on[:, t0] - np.nanmean(on[:, w], 1)) / lanes
    names = list(F.keys())
    return np.stack([np.asarray(F[k], np.float32) for k in names], 1), names

def feats15(Hv, Hq, Ho, ext_v, on, off, net, prof, t0, wd, onset):
    X, names = v10.feats(Hv, Hq, Ho, net, prof, t0, wd, onset)
    L, lnames = long_context(net, Hv, ext_v, on, off, t0)
    X = np.concatenate([X, np.broadcast_to(L[:, None, :], (X.shape[0], X.shape[1], L.shape[1]))], 2)
    return X, names + lnames

def ext_window(mg, t0, M):
    """Masked speeds for T-180..T as an (M, EXT+1) array; missing steps are NaN."""
    out = np.full((M, EXT + 1), np.nan, np.float32)
    lo = max(0, t0 - EXT); out[:, EXT - (t0 - lo):] = mg["v"][:, lo:t0 + 1]
    return out

def sample_day(g, mg, on, off, net, prof, wd, rng, per_day):
    M = len(net); vc = net.v_cut.to_numpy()[:, None]
    V = g["v"]; obs = np.isfinite(V); Lraw = (V <= vc) & obs; L = (v10.t2.denoise(V) <= vc) & obs if hasattr(v10, "t2") else (V <= vc) & obs
    import task2_lgbm as t2
    L = (t2.denoise(V) <= vc) & obs
    anyq = Lraw.any(0); out = []
    cands = np.arange(max(HV - 1, 12), T - H)
    hist_any = np.array([anyq[t - (HV - 1):t].any() for t in cands]); fut_any = np.array([anyq[t + 1:t + H + 1].any() for t in cands])
    early_any = np.array([anyq[t:t + H].any() for t in cands]); last_any = np.array([anyq[t + H] for t in cands])
    groups = {1: cands[~hist_any & fut_any], 0: cands[anyq[cands - 1] & fut_any], 2: cands[~hist_any & ~early_any & last_any]}
    for cond, cc in groups.items():
        if len(cc) == 0: continue
        picks = []
        for t0 in rng.permutation(cc):
            if all(abs(t0 - q) >= 36 for q in picks): picks.append(t0)
            if len(picks) >= per_day: break
        for t0 in picks:
            sl = slice(t0 - (HV - 1), t0 + 1)
            Hv = V[:, sl].copy(); Hq = g["q"][:, sl].copy(); Ho = g["occ"][:, sl].copy()
            drop = rng.random(M) < rng.choice([0.2, 0.3, 0.5]); Hv[drop, -1] = np.nan; Hq[drop, -1] = np.nan; Ho[drop, -1] = np.nan
            X, names = feats15(Hv, Hq, Ho, ext_window(mg, t0, M), on, off, net, prof, t0, wd, 1 if cond in (1, 2) else 0)
            y = L[:, t0 + 1:t0 + H + 1].astype(np.float32); e = (obs & g["elig"])[:, t0 + 1:t0 + H + 1]
            out.append((X, y, e, cond, names))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out_v15"); ap.add_argument("--threads", type=int, default=2); ap.add_argument("--per-day", type=int, default=3)
    ap.add_argument("--day-step", type=int, default=1); ap.add_argument("--panels", nargs="*", default=v10.Q2); ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=15)
    a = ap.parse_args()
    import lightgbm as lgb
    os.makedirs(a.out, exist_ok=True); t_start = time.time(); rng = np.random.default_rng(a.seed)
    nets = {p: network(p) for p in a.panels}; stats = {p: v11.profile_stats(p, nets[p]) for p in a.panels}; full = {p: v11.profile_from(stats[p][0]) for p in a.panels}
    print(f"profile stats {time.time()-t_start:.0f}s", flush=True)
    G, O = {"tr": [], "va": []}, {"tr": [], "va": []}; names = None
    for p in a.panels:
        net = nets[p]; li = {l: i for i, l in enumerate(net.link_id)}; M = len(net); st, days = stats[p]
        excl = set(pd.read_csv(DATA / "task2" / p / "train" / "window_index.csv").date.astype(str))
        dates = sorted(days); hold = set(d for d in dates[-24:] if d not in excl)
        masked = {date_of(f): f for f in day_files(p, "train", masked=True)}
        for f in day_files(p, "train", masked=False)[::a.day_step]:
            d = date_of(f)
            if d in excl or d not in masked: continue
            part = "va" if d in hold else "tr"; loo = v11.profile_from(st, days[d])
            g = day_grid(f, li, M, with_occ=True); mg = day_grid(masked[d], li, M, with_occ=True); on, off = ramp_day(p, "train", d, net)
            for X, y, e, cond, nm in sample_day(g, mg, on, off, net, loo, weekday_of(d), rng, a.per_day if part == "tr" else 1):
                names = nm
                if cond in (0, 1): G[part].append((X.reshape(M * H, -1), y.reshape(-1), e.reshape(-1), cond))
                if cond == 2: O[part].append((X[:, H - 1], y[:, H - 1], e[:, H - 1]))
        print(f"{p}: general tr {len(G['tr'])}, onset-T30 tr {len(O['tr'])} ({time.time()-t_start:.0f}s)", flush=True)
    def stack(L, weight):
        X = np.concatenate([x[0] for x in L]); y = np.concatenate([x[1] for x in L]); e = np.concatenate([x[2] for x in L])
        w = np.concatenate([np.full(len(x[1]), 1.0 / np.sqrt(np.nansum(np.where(x[2], x[1], 0)) + 5.0)) for x in L]) if weight else None
        lab = e & np.isfinite(y); return X[lab], y[lab], (w[lab] * lab.sum() / w[lab].sum()) if weight else None
    params = dict(objective="binary", learning_rate=0.03, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, num_threads=a.threads, verbose=-1, seed=a.seed)
    models = {}
    for key, weight in (("general", False), ("general_w", True)):
        Xtr, ytr, wtr = stack(G["tr"], weight); Xva, yva, _ = stack(G["va"], False)
        print(f"{key}: train {Xtr.shape} pos {ytr.mean():.4f}", flush=True)
        bst = lgb.train(params, lgb.Dataset(Xtr, ytr, weight=wtr, feature_name=names), a.rounds, valid_sets=[lgb.Dataset(Xva, yva)],
                        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)])
        bst.save_model(f"{a.out}/task2_{key}.txt"); models[key] = bst
        if key == "general": print("top:", sorted(zip(bst.feature_importance("gain"), names), reverse=True)[:18], flush=True)
        del Xtr, ytr
    Xo = np.concatenate([x[0] for x in O["tr"]]); yo = np.concatenate([x[1] for x in O["tr"]]); eo = np.concatenate([x[2] for x in O["tr"]])
    Xov = np.concatenate([x[0] for x in O["va"]]); yov = np.concatenate([x[1] for x in O["va"]]); eov = np.concatenate([x[2] for x in O["va"]])
    lab, labv = eo & np.isfinite(yo), eov & np.isfinite(yov)
    print(f"onset-T30: train {Xo[lab].shape} pos {yo[lab].mean():.4f}", flush=True)
    bst = lgb.train(dict(params, num_leaves=63, min_data_in_leaf=50), lgb.Dataset(Xo[lab], yo[lab], feature_name=names), a.rounds,
                    valid_sets=[lgb.Dataset(Xov[labv], yov[labv])], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)])
    bst.save_model(f"{a.out}/task2_onsetT30.txt"); models["onset"] = bst
    rows = []
    for split in ("train", "validation", "private"):
        for p in a.panels:
            net = nets[p]; li = {l: i for i, l in enumerate(net.link_id)}; M = len(net); st, days = stats[p]
            wi = pd.read_csv(DATA / "task2" / p / split / "window_index.csv"); hist = pd.read_parquet(DATA / "task2" / p / split / "window_history.parquet")
            mfiles = {date_of(f): f for f in day_files(p, split, masked=True)}
            for w in wi.itertuples(index=False):
                t0 = int(slots_of(pd.Series([w.forecast_origin]))[0]); onset = int(w.condition == "queue_onset")
                prof = v11.profile_from(st, days[w.date]) if split == "train" else full[p]
                mg = day_grid(mfiles[w.date], li, M, with_occ=True); on, off = ramp_day(p, split, w.date, net)
                Hv, Hq, Ho = v10.load_window(hist, w, li, M, mg, t0)
                X, _ = feats15(Hv, Hq, Ho, ext_window(mg, t0, M), on, off, net, prof, t0, weekday_of(w.date), onset)
                pg = models["general"].predict(X.reshape(M * H, -1)).reshape(M, H); pw = models["general_w"].predict(X.reshape(M * H, -1)).reshape(M, H)
                p6 = models["onset"].predict(X[:, H - 1]) if onset else np.full(M, np.nan)
                for h in range(H):
                    rows.append(pd.DataFrame({"split": split, "panel": p, "window_id": w.window_id, "condition": w.condition, "h": h + 1,
                                              "link_id": net.link_id.to_numpy(), "p_general": pg[:, h], "p_general_w": pw[:, h],
                                              "p_onsetT30": p6 if h == H - 1 else np.nan}))
        print(f"probs dumped for {split} ({time.time()-t_start:.0f}s)", flush=True)
    pd.concat(rows, ignore_index=True).to_parquet(f"{a.out}/window_probs.parquet", index=False)
    json.dump({"features": names, "runtime_s": time.time() - t_start, "seed": a.seed, "profile": "leave-one-day-out", "context": "3h masked + ramps"}, open(f"{a.out}/run_info.json", "w"))
    print("done", f"{time.time()-t_start:.0f}s", flush=True)

if __name__ == "__main__":
    main()
