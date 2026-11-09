# Source: yeashusemwal, "TrafficFlowBench Trees + U-Net + CNN" (Kaggle notebook, 2026),
# kaggle.com/code/yeashusemwal/trafficflowbench-trees-u-net-cnn-lb-0-87. Redistributed for reproduction with credit; see NOTICE.
"""Task 2 v10: history extended to include the (partially observed) forecast origin T, plus an onset specialist
trained only on events whose queue first appears at T+30 (organizer-confirmed selection rule, discussion 742350).
Organizer train-window dates are excluded from training and scored honestly at the end; probabilities for
train/validation/private windows are dumped so thresholds can be tuned without retraining."""
import os, sys, time, json, argparse, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import task2_lgbm as t2
from common import DATA, T, network, day_files, date_of, weekday_of, slots_of, day_grid, build_profile

Q2 = ["D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N", "D7_I405_S", "D12_I5_N", "D12_I5_S"]
HV = 13  # T-60 .. T (the last column is the origin T, observed only where the Task 1 mask left it)
H = 6

def feats(Hv, Hq, Ho, net, prof, t0, wd, onset):
    t2.HIST = HV
    X, names = t2.window_features(Hv, Hq, Ho, net, prof, t0, wd, onset)
    obsT = np.isfinite(Hv[:, -1]).astype(np.float32)
    X = np.concatenate([X, np.broadcast_to(obsT[:, None, None], (X.shape[0], X.shape[1], 1))], 2)
    return X, names + ["obs_at_T"]

def sample_day(g, net, prof, wd, rng, per_day, excl_day):
    """Returns list of (X (M,H,F), y (M,H), e (M,H), cond, onset_T30) windows sampled from one unmasked train day."""
    M = len(net); vc = net.v_cut.to_numpy()[:, None]
    V = g["v"]; obs = np.isfinite(V); Lraw = (V <= vc) & obs; L = (t2.denoise(V) <= vc) & obs
    anyq = Lraw.any(0); out = []
    cands = np.arange(HV - 1, T - H)
    hist_any = np.array([anyq[t - (HV - 1):t].any() for t in cands]); fut_any = np.array([anyq[t + 1:t + H + 1].any() for t in cands])
    early_any = np.array([anyq[t:t + H].any() for t in cands])          # T .. T+25
    last_any = np.array([anyq[t + H] for t in cands])                   # T+30
    groups = {1: cands[~hist_any & fut_any], 0: cands[anyq[cands - 1] & fut_any], 2: cands[~hist_any & ~early_any & last_any]}
    for cond, cc in groups.items():
        if len(cc) == 0: continue
        picks = []
        for t0 in rng.permutation(cc):
            if all(abs(t0 - q) >= 36 for q in picks): picks.append(t0)
            if len(picks) >= per_day: break
        for t0 in picks:
            sl = slice(t0 - (HV - 1), t0 + 1)
            Hv = g["v"][:, sl].copy(); Hq = g["q"][:, sl].copy(); Ho = g["occ"][:, sl].copy()
            drop = rng.random(M) < rng.choice([0.2, 0.3, 0.5])          # emulate the Task 1 mask at T
            Hv[drop, -1] = np.nan; Hq[drop, -1] = np.nan; Ho[drop, -1] = np.nan
            X, names = feats(Hv, Hq, Ho, net, prof, t0, wd, 1 if cond in (1, 2) else 0)
            y = L[:, t0 + 1:t0 + H + 1].astype(np.float32); e = (obs & g["elig"])[:, t0 + 1:t0 + H + 1]
            out.append((X, y, e, cond, names))
    return out

def load_window(hist, w, li, M, masked_grid, t0):
    hw = hist[hist.window_id == w.window_id]; ts = sorted(hw.timestamp.unique()); tix = {x: i for i, x in enumerate(ts)}
    Hv = np.full((M, HV), np.nan, np.float32); Hq = Hv.copy(); Ho = Hv.copy()
    lk = hw.link_id.map(li).to_numpy(); ok = ~pd.isna(lk); ti = hw.timestamp.map(tix).to_numpy()
    for arr, col in ((Hv, "speed_kmh"), (Hq, "flow_vph"), (Ho, "occupancy")): arr[lk[ok].astype(int), ti[ok]] = hw[col].to_numpy()[ok]
    if masked_grid is not None:  # column T from the released masked mainline (allowed: timestamp == T)
        Hv[:, -1] = masked_grid["v"][:, t0]; Hq[:, -1] = masked_grid["q"][:, t0]; Ho[:, -1] = masked_grid["occ"][:, t0]
    return Hv, Hq, Ho

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out_v10"); ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--per-day", type=int, default=3); ap.add_argument("--day-step", type=int, default=1)
    ap.add_argument("--panels", nargs="*", default=Q2); ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--cache", default="profiles_cache")
    a = ap.parse_args()
    import lightgbm as lgb
    os.makedirs(a.out, exist_ok=True); t_start = time.time(); rng = np.random.default_rng(10)
    from pathlib import Path
    nets = {p: network(p) for p in a.panels}; profs = {p: build_profile(p, nets[p], Path(a.cache)) for p in a.panels}
    print(f"profiles ready {time.time()-t_start:.0f}s", flush=True)
    G, O = {"tr": [], "va": []}, {"tr": [], "va": []}; names = None
    for p in a.panels:
        net = nets[p]; li = {l: i for i, l in enumerate(net.link_id)}; M = len(net)
        excl = set(pd.read_csv(DATA / "task2" / p / "train" / "window_index.csv").date.astype(str))
        dates = sorted(date_of(x) for x in day_files(p, "train", masked=False)); hold = set(d for d in dates[-24:] if d not in excl)
        for f in day_files(p, "train", masked=False)[::a.day_step]:
            d = date_of(f)
            if d in excl: continue
            part = "va" if d in hold else "tr"
            g = day_grid(f, li, M, with_occ=True)
            for X, y, e, cond, nm in sample_day(g, net, profs[p], weekday_of(d), rng, a.per_day if part == "tr" else 1, excl):
                names = nm
                if cond in (0, 1): G[part].append((X.reshape(M * H, -1), y.reshape(-1), e.reshape(-1), cond))
                if cond == 2: O[part].append((X[:, H - 1], y[:, H - 1], e[:, H - 1]))
        print(f"{p}: general tr {len(G['tr'])} windows, onset-T30 tr {len(O['tr'])} ({time.time()-t_start:.0f}s)", flush=True)
    def stack_general(L, weight):
        X = np.concatenate([x[0] for x in L]); y = np.concatenate([x[1] for x in L]); e = np.concatenate([x[2] for x in L])
        w = np.concatenate([np.full(len(x[1]), 1.0 / np.sqrt(np.nansum(np.where(x[2], x[1], 0)) + 5.0)) for x in L]) if weight else None
        lab = e & np.isfinite(y); return X[lab], y[lab], (w[lab] * lab.sum() / w[lab].sum()) if weight else None
    params = dict(objective="binary", learning_rate=0.03, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, num_threads=a.threads, verbose=-1, seed=10)
    models = {}
    for key, weight in (("general", False), ("general_w", True)):
        Xtr, ytr, wtr = stack_general(G["tr"], weight); Xva, yva, _ = stack_general(G["va"], False)
        print(f"{key}: train {Xtr.shape} pos {ytr.mean():.4f}", flush=True)
        bst = lgb.train(params, lgb.Dataset(Xtr, ytr, weight=wtr, feature_name=names), a.rounds, valid_sets=[lgb.Dataset(Xva, yva)],
                        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)])
        bst.save_model(f"{a.out}/task2_{key}.txt"); models[key] = bst; del Xtr, ytr
    Xo = np.concatenate([x[0] for x in O["tr"]]); yo = np.concatenate([x[1] for x in O["tr"]]); eo = np.concatenate([x[2] for x in O["tr"]])
    Xov = np.concatenate([x[0] for x in O["va"]]); yov = np.concatenate([x[1] for x in O["va"]]); eov = np.concatenate([x[2] for x in O["va"]])
    lab, labv = eo & np.isfinite(yo), eov & np.isfinite(yov)
    print(f"onset-T30: train {Xo[lab].shape} pos {yo[lab].mean():.4f}", flush=True)
    po = dict(params, num_leaves=63, min_data_in_leaf=50)
    bst = lgb.train(po, lgb.Dataset(Xo[lab], yo[lab], feature_name=names), a.rounds, valid_sets=[lgb.Dataset(Xov[labv], yov[labv])],
                    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)])
    bst.save_model(f"{a.out}/task2_onsetT30.txt"); models["onset"] = bst
    # dump probabilities for every released window on all splits; T column read from each split's masked mainline
    rows = []
    for split in ("train", "validation", "private"):
        for p in a.panels:
            net = nets[p]; li = {l: i for i, l in enumerate(net.link_id)}; M = len(net)
            wi = pd.read_csv(DATA / "task2" / p / split / "window_index.csv"); hist = pd.read_parquet(DATA / "task2" / p / split / "window_history.parquet")
            mfiles = {date_of(f): f for f in day_files(p, split, masked=True)}
            for w in wi.itertuples(index=False):
                t0 = int(slots_of(pd.Series([w.forecast_origin]))[0]); onset = int(w.condition == "queue_onset")
                mg = day_grid(mfiles[w.date], li, M, with_occ=True) if w.date in mfiles else None
                Hv, Hq, Ho = load_window(hist, w, li, M, mg, t0)
                X, _ = feats(Hv, Hq, Ho, net, profs[p], t0, weekday_of(w.date), onset)
                pg = models["general"].predict(X.reshape(M * H, -1)).reshape(M, H); pw = models["general_w"].predict(X.reshape(M * H, -1)).reshape(M, H)
                p6 = models["onset"].predict(X[:, H - 1]) if onset else np.full(M, np.nan)
                for h in range(H):
                    rows.append(pd.DataFrame({"split": split, "panel": p, "window_id": w.window_id, "condition": w.condition, "h": h + 1,
                                              "link_id": net.link_id.to_numpy(), "p_general": pg[:, h], "p_general_w": pw[:, h],
                                              "p_onsetT30": p6 if h == H - 1 else np.nan}))
        print(f"probs dumped for {split} ({time.time()-t_start:.0f}s)", flush=True)
    P = pd.concat(rows, ignore_index=True); P.to_parquet(f"{a.out}/window_probs.parquet", index=False)
    json.dump({"features": names, "runtime_s": time.time() - t_start, "per_day": a.per_day, "day_step": a.day_step, "seed": 10}, open(f"{a.out}/run_info.json", "w"))
    print("done", f"{time.time()-t_start:.0f}s", flush=True)

if __name__ == "__main__":
    main()
