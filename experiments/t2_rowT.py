"""Task 2 with the row at T masked like the test data.

At test time the row at T comes from the Task 1-masked file, so 20/30/50% of its eligible cells are blank; our
earlier model saw it fully observed during training. Here training windows get the row at T masked at a rate drawn
from {0.2, 0.3, 0.5}.

eval : fit on Jul-Jan, score Jun/Feb windows whose row at T is masked the same way, for an unmasked-trained model
       vs a masked-trained model (same features: row at T + patch + long-run memory)
final: fit the masked-trained model on all months (3 seeds) and predict validation/private -> t2_pred_rowT_{split}.csv
"""
from __future__ import annotations

import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import CACHE, RELEASE, TASK2_EXCLUDED, load, panels
from t2_data import HOR, PanelT2, iou, queue_memory
from t2_exp import stack
from t2_final import WH, MASKED, window_day

HOLD = ("2030-06", "2031-02")
RATES = (0.2, 0.3, 0.5)


def mask_T(S, F, t, rng):
    S, F = S.copy(), F.copy()
    m = rng.random(S.shape[1]) < rng.choice(RATES)
    S[t, m] = np.nan; F[t, m] = np.nan
    return S, F


def build(rng, mask_train: bool, per_day=6):
    rows = {"queue_onset": [], "queue_ongoing": []}; names = None
    for pno, panel in enumerate(p for p in panels() if p not in TASK2_EXCLUDED):
        t0 = time.time(); P = PanelT2(panel, pno); z = load(panel, "train", masked=False)
        S, F = P.orient(z["speed"]), P.orient(z["flow"])
        months = np.array([d[:7] for d in z["dates"]]); dows = pd.to_datetime(z["dates"]).day_of_week.to_numpy()
        prior = {m: queue_memory(S[months != m], P.cut) for m in np.unique(months)}
        for d in range(len(months)):
            q, fin = P.queued(S[d]), np.isfinite(S[d])
            for cond, ts in P.candidates(S[d]).items():
                if cond == "queue_ongoing" and len(ts) > per_day:
                    ts = sorted(rng.choice(ts, per_day, replace=False))
                for t in ts:
                    hold = months[d] in HOLD
                    Sm, Fm = mask_T(S[d], F[d], t, rng) if (mask_train or hold) else (S[d], F[d])
                    X, names = P.features(Sm, Fm, t, dows[d], cond, prior[months[d]], None)
                    rows[cond].append(dict(month=months[d], X=X, y=q[t + 1:t + HOR + 1], ok=fin[t + 1:t + HOR + 1]))
        print(panel, f"{time.time()-t0:.0f}s", flush=True)
    return rows, names


def fit(rows, names, cond, months_ok, seed):
    onset = cond == "queue_onset"
    cols = [i for i, n in enumerate(names) if not n.startswith("mmem")] + [len(names)]
    X, y = stack([w for w in rows[cond] if months_ok(w["month"])], onset)
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.7,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=10, num_threads=4, verbose=-1, seed=seed)
    return lgb.train(params, lgb.Dataset(X[:, cols], y), 650 if onset else 850), cols


def window_iou(model, cols, w, onset, th):
    L = w["X"].shape[0]
    P = np.stack([model.predict(np.column_stack([w["X"], np.full(L, h + 1, np.float32)])[:, cols]) for h in range(HOR)])
    pred = P >= th
    if onset:
        pred[:-1] = False
        if not pred[-1].any():
            pred[-1, np.argmax(P[-1])] = True
    return iou(pred & w["ok"], w["y"] & w["ok"])


def evaluate():
    out = []
    for mask_train in (False, True):
        rows, names = build(np.random.default_rng(7), mask_train)
        for cond in rows:
            onset = cond == "queue_onset"
            m, cols = fit(rows, names, cond, lambda mo: mo not in HOLD, 1)
            for hm in HOLD:
                ws = [w for w in rows[cond] if w["month"] == hm]
                for th in (0.35, 0.4, 0.45, 0.5, 0.55):
                    out.append(dict(train_rowT_masked=mask_train, cond=cond, month=hm, th=th,
                                    iou=float(np.mean([window_iou(m, cols, w, onset, th) for w in ws]))))
            print(pd.DataFrame([o for o in out if o["cond"] == cond and o["train_rowT_masked"] == mask_train])
                  .pivot_table(index="month", columns="th", values="iou").round(4).to_string(), flush=True)
    r = pd.DataFrame(out); r.to_csv(CACHE / "t2_rowT_eval.csv", index=False)
    print(r.pivot_table(index=["cond", "month", "train_rowT_masked"], columns="th", values="iou").round(4).to_string())


def final():
    rows, names = build(np.random.default_rng(11), True)
    models = {c: [fit(rows, names, c, lambda mo: True, s) for s in (1, 2, 3)] for c in rows}
    th = {"queue_onset": 0.45, "queue_ongoing": 0.47}
    for split in ("validation", "private"):
        res = []
        for pno, panel in enumerate(p for p in panels() if p not in TASK2_EXCLUDED):
            P = PanelT2(panel, pno); prior = queue_memory(P.orient(load(panel, "train", masked=False)["speed"]), P.cut)
            WH[(panel, split)] = pd.read_parquet(RELEASE / "task2" / panel / split / "window_history.parquet")
            MASKED[(panel, split)] = load(panel, split, masked=True)
            for w in pd.read_csv(RELEASE / "task2" / panel / split / "window_index.csv").itertuples():
                S, F, t = window_day(P, panel, split, w)
                X, _ = P.features(S, F, t, pd.Timestamp(w.date).day_of_week, w.condition, prior, None)
                L = P.L
                prob = np.stack([np.mean([m.predict(np.column_stack([X, np.full(L, h + 1, np.float32)])[:, c]) for m, c in models[w.condition]], 0)
                                 for h in range(HOR)])
                pred = prob >= th[w.condition]
                if w.condition == "queue_onset":
                    pred[:-1] = False
                    if not pred[-1].any():
                        pred[-1, np.argmax(prob[-1])] = True
                ts = pd.date_range(pd.Timestamp(w.forecast_origin) + pd.Timedelta(minutes=5), periods=HOR, freq="5min").strftime("%Y-%m-%dT%H:%M:%SZ")
                for h in range(HOR):
                    res.append(pd.DataFrame({"window_id": w.window_id, "timestamp": ts[h], "link_id": P.links,
                                             "queue_pred": pred[h].astype(int), "prob": prob[h]}))
        pd.concat(res).to_csv(CACHE / f"t2_pred_rowT_{split}.csv", index=False); print(split, "written", flush=True)


if __name__ == "__main__":
    {"eval": evaluate, "final": final}[sys.argv[1]]()
