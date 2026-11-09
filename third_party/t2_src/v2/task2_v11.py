# Source: yeashusemwal, "TrafficFlowBench Trees + U-Net + CNN" (Kaggle notebook, 2026),
# kaggle.com/code/yeashusemwal/trafficflowbench-trees-u-net-cnn-lb-0-87. Redistributed for reproduction with credit; see NOTICE.
"""Task 2 v11 = v10 + leave-one-day-out (LOO) profiles.
The weekday x slot profile used as features is built from all train days, so on a train day it contains that day's own
traffic (~1/39 weight). v10 learned to lean on that leak (local ongoing IoU 0.869 -> 0.830 with a leak-free profile).
Here every training sample and every train-split evaluation window gets a profile with its own day removed; validation
and private windows use the full train profile, which is leak-free for them by construction."""
import os, sys, time, json, argparse, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import task2_v10 as v
from common import DATA, T, network, day_files, date_of, weekday_of, slots_of, day_grid

H = v.H

def profile_stats(panel, net):
    """Raw per-weekday sums so a single day can be subtracted exactly."""
    M = len(net); li = {l: i for i, l in enumerate(net.link_id)}; vcut = net.v_cut.to_numpy()[:, None]
    st = {k: np.zeros((7, M, T)) for k in ("sv", "sq", "sv2", "n", "nq")}; days = {}
    for f in day_files(panel, "train", masked=False):
        d = date_of(f); g = day_grid(f, li, M); w = weekday_of(d)
        ok = np.isfinite(g["v"]) & np.isfinite(g["q"]); vv = np.where(ok, g["v"], 0.0); qq = np.where(ok, g["q"], 0.0)
        c = dict(sv=vv, sq=qq, sv2=vv * vv, n=ok.astype(float), nq=(ok & (g["v"] <= vcut)).astype(float))
        for k in st: st[k][w] += c[k]
        days[d] = (w, {k: c[k].astype(np.float32) for k in c})
    return st, days

def profile_from(st, drop=None):
    """Profile dict {v,q,v_std,qfrac} (7,M,T); drop=(weekday, contributions) removes one day."""
    s = {k: a.copy() for k, a in st.items()}
    if drop is not None:
        w, c = drop
        for k in s: s[k][w] -= c[k]
    n = np.maximum(s["n"], 1); vm = s["sv"] / n; qm = s["sq"] / n
    vs = np.sqrt(np.maximum(s["sv2"] / n - vm * vm, 0)); qf = s["nq"] / n
    for w in range(7):
        e = s["n"][w] == 0
        if e.any():
            lm_v = np.where(s["n"][w].sum(1) > 0, s["sv"][w].sum(1) / np.maximum(s["n"][w].sum(1), 1), 100.0)
            lm_q = np.where(s["n"][w].sum(1) > 0, s["sq"][w].sum(1) / np.maximum(s["n"][w].sum(1), 1), 3000.0)
            vm[w][e] = np.broadcast_to(lm_v[:, None], e.shape)[e]; qm[w][e] = np.broadcast_to(lm_q[:, None], e.shape)[e]
    return {"v": vm.astype(np.float32), "q": qm.astype(np.float32), "v_std": vs.astype(np.float32), "qfrac": qf.astype(np.float32)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out_v11"); ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--per-day", type=int, default=3); ap.add_argument("--day-step", type=int, default=1)
    ap.add_argument("--panels", nargs="*", default=v.Q2); ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    import lightgbm as lgb
    os.makedirs(a.out, exist_ok=True); t_start = time.time(); rng = np.random.default_rng(a.seed)
    nets = {p: network(p) for p in a.panels}; stats = {}
    for p in a.panels: stats[p] = profile_stats(p, nets[p])
    full = {p: profile_from(stats[p][0]) for p in a.panels}
    print(f"profile stats ready {time.time()-t_start:.0f}s", flush=True)
    G, O = {"tr": [], "va": []}, {"tr": [], "va": []}; names = None
    for p in a.panels:
        net = nets[p]; li = {l: i for i, l in enumerate(net.link_id)}; M = len(net); st, days = stats[p]
        excl = set(pd.read_csv(DATA / "task2" / p / "train" / "window_index.csv").date.astype(str))
        dates = sorted(days); hold = set(d for d in dates[-24:] if d not in excl)
        for f in day_files(p, "train", masked=False)[::a.day_step]:
            d = date_of(f)
            if d in excl: continue
            part = "va" if d in hold else "tr"; loo = profile_from(st, days[d])
            g = day_grid(f, li, M, with_occ=True)
            for X, y, e, cond, nm in v.sample_day(g, net, loo, weekday_of(d), rng, a.per_day if part == "tr" else 1, excl):
                names = nm
                if cond in (0, 1): G[part].append((X.reshape(M * H, -1), y.reshape(-1), e.reshape(-1), cond))
                if cond == 2: O[part].append((X[:, H - 1], y[:, H - 1], e[:, H - 1]))
        print(f"{p}: general tr {len(G['tr'])}, onset-T30 tr {len(O['tr'])} ({time.time()-t_start:.0f}s)", flush=True)
    def stack_general(L, weight):
        X = np.concatenate([x[0] for x in L]); y = np.concatenate([x[1] for x in L]); e = np.concatenate([x[2] for x in L])
        w = np.concatenate([np.full(len(x[1]), 1.0 / np.sqrt(np.nansum(np.where(x[2], x[1], 0)) + 5.0)) for x in L]) if weight else None
        lab = e & np.isfinite(y); return X[lab], y[lab], (w[lab] * lab.sum() / w[lab].sum()) if weight else None
    params = dict(objective="binary", learning_rate=0.03, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, num_threads=a.threads, verbose=-1, seed=a.seed)
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
                prof = profile_from(st, days[w.date]) if split == "train" else full[p]   # LOO on train windows, full elsewhere
                mg = day_grid(mfiles[w.date], li, M, with_occ=True) if w.date in mfiles else None
                Hv, Hq, Ho = v.load_window(hist, w, li, M, mg, t0)
                X, _ = v.feats(Hv, Hq, Ho, net, prof, t0, weekday_of(w.date), onset)
                pg = models["general"].predict(X.reshape(M * H, -1)).reshape(M, H); pw = models["general_w"].predict(X.reshape(M * H, -1)).reshape(M, H)
                p6 = models["onset"].predict(X[:, H - 1]) if onset else np.full(M, np.nan)
                for h in range(H):
                    rows.append(pd.DataFrame({"split": split, "panel": p, "window_id": w.window_id, "condition": w.condition, "h": h + 1,
                                              "link_id": net.link_id.to_numpy(), "p_general": pg[:, h], "p_general_w": pw[:, h],
                                              "p_onsetT30": p6 if h == H - 1 else np.nan}))
        print(f"probs dumped for {split} ({time.time()-t_start:.0f}s)", flush=True)
    pd.concat(rows, ignore_index=True).to_parquet(f"{a.out}/window_probs.parquet", index=False)
    json.dump({"features": names, "runtime_s": time.time() - t_start, "per_day": a.per_day, "day_step": a.day_step, "seed": a.seed, "profile": "leave-one-day-out"},
              open(f"{a.out}/run_info.json", "w"))
    print("done", f"{time.time()-t_start:.0f}s", flush=True)

if __name__ == "__main__":
    main()
