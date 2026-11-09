"""Task 2 experiment: build generated windows from train, fit LightGBM onset/ongoing models,
and report window-level IoU on held-out months (June official-like + February) for feature ablations."""
from __future__ import annotations

import argparse
import pickle
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import CACHE, TASK2_EXCLUDED, load, panels
from t2_data import HIST, HOR, PanelT2, iou, queue_memory

HOLDOUT = ("2030-06", "2031-02")
RNG = np.random.default_rng(7)


def build(use_T: bool, ongoing_per_day: int):
    rows = {"queue_onset": [], "queue_ongoing": []}
    for pno, panel in enumerate(p for p in panels() if p not in TASK2_EXCLUDED):
        t0 = time.time()
        P = PanelT2(panel, pno)
        z = load(panel, "train", masked=False)
        S, F = P.orient(z["speed"]), P.orient(z["flow"])
        dates = z["dates"]; months = np.array([d[:7] for d in dates])
        dows = pd.to_datetime(dates).dayofweek.to_numpy()
        prior = {m: queue_memory(S[(months != m) & ~np.isin(months, HOLDOUT)], P.cut) for m in np.unique(months)}
        for d in range(len(dates)):
            m = months[d]
            same = np.where((months == m) & (np.arange(len(dates)) < d))[0]
            mmem = queue_memory(S[same], P.cut) if len(same) else np.full((288, P.L), np.nan, np.float32)
            cand = P.candidates(S[d])
            q = P.queued(S[d]); fin = np.isfinite(S[d])
            for cond, ts in cand.items():
                if cond == "queue_ongoing" and len(ts) > ongoing_per_day:
                    ts = sorted(RNG.choice(ts, ongoing_per_day, replace=False))
                for t in ts:
                    X, names = P.features(S[d], F[d], t, dows[d], cond, prior[m], mmem, use_T=use_T)
                    rows[cond].append(dict(panel=panel, month=m, day=d, t=t, X=X,
                                           y=q[t + 1:t + HOR + 1].copy(), ok=fin[t + 1:t + HOR + 1].copy()))
        print(panel, f"{time.time()-t0:.0f}s", {k: len(v) for k, v in rows.items()}, flush=True)
    return rows, names


def stack(ws, onset: bool):
    Xs, ys = [], []
    for w in ws:
        if onset:
            ok = w["ok"][-1]
            Xs.append(np.column_stack([w["X"][ok], np.full(ok.sum(), HOR, np.float32)])); ys.append(w["y"][-1][ok])
        else:
            for h in range(HOR):
                ok = w["ok"][h]
                Xs.append(np.column_stack([w["X"][ok], np.full(ok.sum(), h + 1, np.float32)])); ys.append(w["y"][h][ok])
    return np.concatenate(Xs), np.concatenate(ys).astype(np.float32)


def predict_window(model, w, onset: bool, th: float):
    L = w["X"].shape[0]
    if onset:
        p = model.predict(np.column_stack([w["X"], np.full(L, HOR, np.float32)]))
        pred = np.zeros((HOR, L), bool)
        pred[-1] = p >= th
        if not pred[-1].any():
            pred[-1, np.argmax(p)] = True
        return pred
    P = np.stack([model.predict(np.column_stack([w["X"], np.full(L, h + 1, np.float32)])) for h in range(HOR)])
    return P >= th


def evaluate(model, ws, onset, ths):
    res = {}
    for th in ths:
        s = [iou(predict_window(model, w, onset, th) & w["ok"], w["y"] & w["ok"]) for w in ws]
        res[th] = float(np.mean(s))
    return res


def run(rows, names, keep, tag):
    cols = [i for i, n in enumerate(names) if keep(n)] + [len(names)]  # + horizon step
    out = {}
    for cond in ("queue_onset", "queue_ongoing"):
        onset = cond == "queue_onset"
        tr = [w for w in rows[cond] if w["month"] not in HOLDOUT]
        X, y = stack(tr, onset)
        params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.7,
                      bagging_fraction=0.8, bagging_freq=1, lambda_l2=10, num_threads=12, verbosity=-1, seed=1)
        model = lgb.train(params, lgb.Dataset(X[:, cols], y), num_boost_round=600 if onset else 800)

        class Sub:  # restrict to kept columns at predict time
            def predict(self, Z):
                return model.predict(Z[:, cols])
        for hm in HOLDOUT:
            ws = [w for w in rows[cond] if w["month"] == hm]
            ths = (0.3, 0.4, 0.5, 0.6) if onset else (0.3, 0.4, 0.45, 0.5, 0.6)
            out[(cond, hm)] = evaluate(Sub(), ws, onset, ths)
        print(tag, cond, {k: {t: round(v, 4) for t, v in r.items()} for k, r in out.items() if k[0] == cond}, flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ongoing-per-day", type=int, default=4)
    a = ap.parse_args()
    results = {}
    for use_T in (False, True):
        cache = CACHE / f"t2_rows_T{int(use_T)}.pkl"
        if cache.exists():
            rows, names = pickle.load(open(cache, "rb"))
        else:
            rows, names = build(use_T, a.ongoing_per_day)
            pickle.dump((rows, names), open(cache, "wb"), protocol=4)
        base = lambda n: not n.startswith(("ps_", "pm_", "pf_", "pd_", "dq_", "prior", "mmem"))
        variants = {"base": base, "base+patch": lambda n: not n.startswith(("prior", "mmem")),
                    "base+patch+prior": lambda n: not n.startswith("mmem"), "all": lambda n: True}
        if not use_T:
            variants = {"base": base, "all": lambda n: True}
        for vname, keep in variants.items():
            results[(use_T, vname)] = run(rows, names, keep, f"T={use_T} {vname}")
    pickle.dump(results, open(CACHE / "t2_exp_results.pkl", "wb"))
