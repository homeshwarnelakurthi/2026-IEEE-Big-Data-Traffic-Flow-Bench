"""Task 2 final model: train onset/ongoing LightGBM on every train month (row at T + patch + long-run
bottleneck memory), bag seeds, and predict the official validation/private windows.

Inference is strictly causal: each window sees only window_history (T-60..T-5), the released row at T,
and memory computed from the train split. Writes cache/t2_pred_{split}.csv in template format."""
from __future__ import annotations

import pickle
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import CACHE, RELEASE, TASK2_EXCLUDED, load, panels, slot_of
from t2_data import HIST, HOR, PanelT2, queue_memory
from t2_exp import stack

SEEDS = (1, 2, 3)
TH = {"queue_onset": 0.45, "queue_ongoing": 0.47}
ONGOING_PER_DAY = 6
DROP = ("mmem",)
RNG = np.random.default_rng(11)


def keep_cols(names):
    return [i for i, n in enumerate(names) if not n.startswith(DROP)] + [len(names)]


def train_rows():
    out = CACHE / "t2_final_rows.pkl"
    if out.exists():
        return pickle.load(open(out, "rb"))
    rows = {"queue_onset": [], "queue_ongoing": []}
    names = None
    for pno, panel in enumerate(p for p in panels() if p not in TASK2_EXCLUDED):
        t0 = time.time()
        P = PanelT2(panel, pno)
        z = load(panel, "train", masked=False)
        S, F = P.orient(z["speed"]), P.orient(z["flow"])
        months = np.array([d[:7] for d in z["dates"]])
        dows = pd.to_datetime(z["dates"]).day_of_week.to_numpy()
        prior = {m: queue_memory(S[months != m], P.cut) for m in np.unique(months)}
        for d in range(len(months)):
            q, fin = P.queued(S[d]), np.isfinite(S[d])
            for cond, ts in P.candidates(S[d]).items():
                if cond == "queue_ongoing" and len(ts) > ONGOING_PER_DAY:
                    ts = sorted(RNG.choice(ts, ONGOING_PER_DAY, replace=False))
                for t in ts:
                    X, names = P.features(S[d], F[d], t, dows[d], cond, prior[months[d]], None)
                    rows[cond].append(dict(X=X, y=q[t + 1:t + HOR + 1], ok=fin[t + 1:t + HOR + 1]))
        print(panel, f"{time.time()-t0:.0f}s", {k: len(v) for k, v in rows.items()}, flush=True)
    pickle.dump((rows, names), open(out, "wb"), protocol=4)
    return rows, names


def fit(rows, names):
    models = {}
    cols = keep_cols(names)
    for cond in rows:
        onset = cond == "queue_onset"
        X, y = stack(rows[cond], onset)
        models[cond] = []
        for seed in SEEDS:
            params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                          feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10,
                          num_threads=12, verbosity=-1, seed=seed)
            models[cond].append(lgb.train(params, lgb.Dataset(X[:, cols], y), num_boost_round=650 if onset else 850))
            print("fitted", cond, seed, flush=True)
    return models, cols


def window_day(P: PanelT2, panel: str, split: str, w) -> tuple[np.ndarray, np.ndarray, int]:
    """Day arrays holding only what is visible at T: window_history rows and the released row at T."""
    t = slot_of(w.forecast_origin)
    S = np.full((288, P.L), np.nan, np.float32); F = S.copy()
    wh = WH[(panel, split)]
    h = wh[wh.window_id == w.window_id]
    col = {l: i for i, l in enumerate(P.links)}
    s = h.timestamp.map(slot_of).to_numpy(); j = h.link_id.astype(str).map(col).to_numpy()
    assert s.max() < t, "history must end before T"
    S[s, j] = h.speed_kmh.to_numpy(); F[s, j] = h.flow_vph.to_numpy()
    z = MASKED[(panel, split)]
    d = list(z["dates"]).index(w.date)
    S[t] = P.orient(z["speed"][d, t]); F[t] = P.orient(z["flow"][d, t])  # the row at T, allowed by the rules
    return S, F, t


def predict(models, cols, split):
    out = []
    for pno, panel in enumerate(p for p in panels() if p not in TASK2_EXCLUDED):
        P = PanelT2(panel, pno)
        full = load(panel, "train", masked=False)
        prior = queue_memory(P.orient(full["speed"]), P.cut)
        WH[(panel, split)] = pd.read_parquet(RELEASE / "task2" / panel / split / "window_history.parquet")
        MASKED[(panel, split)] = load(panel, split, masked=True)
        wi = pd.read_csv(RELEASE / "task2" / panel / split / "window_index.csv")
        for w in wi.itertuples():
            S, F, t = window_day(P, panel, split, w)
            X, _ = P.features(S, F, t, pd.Timestamp(w.date).day_of_week, w.condition, prior, None)
            ms = models[w.condition]
            prob = np.stack([np.mean([m.predict(np.column_stack([X, np.full(P.L, h + 1, np.float32)])[:, cols]) for m in ms], 0)
                             for h in range(HOR)])
            pred = prob >= TH[w.condition]
            if w.condition == "queue_onset":
                pred[:-1] = False
                if not pred[-1].any():
                    pred[-1, np.argmax(prob[-1])] = True
            times = pd.date_range(pd.Timestamp(w.forecast_origin) + pd.Timedelta(minutes=5), periods=HOR, freq="5min")
            ts = times.strftime("%Y-%m-%dT%H:%M:%SZ")
            for h in range(HOR):
                out.append(pd.DataFrame({"window_id": w.window_id, "timestamp": ts[h], "link_id": P.links,
                                         "queue_pred": pred[h].astype(int), "prob": prob[h]}))
        print(split, panel, "done", flush=True)
    return pd.concat(out, ignore_index=True)


WH, MASKED = {}, {}

if __name__ == "__main__":
    rows, names = train_rows()
    models, cols = fit(rows, names)
    pickle.dump((models, cols, names), open(CACHE / "t2_final_models.pkl", "wb"))
    for split in ("validation", "private"):
        p = predict(models, cols, split)
        p.to_csv(CACHE / f"t2_pred_{split}.csv", index=False)
        print(split, "rows", len(p), "positive share", p.queue_pred.mean().round(4))
