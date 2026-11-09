# LOCAL ANALYSIS ONLY: not used by run_all.py. Reads intermediate outputs of our Kaggle runs under $TFB_WORKSPACE (not in this repo).
"""Task 2 blend: public anubisyy queue models vs ours vs probability blend.

feb : fit ours on Jul-Jan generated windows, score public / ours / blends on February windows
apply: blend our final-model probabilities with public probabilities on the official windows
"""
from __future__ import annotations

import os
import pickle
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import CACHE, RELEASE, TASK2_EXCLUDED, load, panels, slot_of
from t2_data import HIST, HOR, PanelT2, iou
from t2_exp import HOLDOUT, stack

WS = __import__("os").environ.get("TFB_WORKSPACE", ".")  # folder holding our Kaggle run outputs

os.environ.setdefault("TFB_DATA_ROOT", str(RELEASE))
os.environ.setdefault("TFB_OUTPUT_ROOT", str(CACHE / "public_tmp"))
sys.path.insert(0, WS + r"/public_assets")
from catboost import CatBoostClassifier  # noqa: E402
from queue_features import Panel as PubOnsetPanel  # noqa: E402
from queue_ongoing import Panel as PubOngoingPanel  # noqa: E402

A = WS + r"/public_assets"
PUB_ON_LGB = lgb.Booster(model_file=A + r"/onset_lgb.txt")
PUB_ON_CAT = CatBoostClassifier(); PUB_ON_CAT.load_model(A + r"/onset.cbm")
PUB_OG_LGB = lgb.Booster(model_file=A + r"/ongoing_lgb.txt")
PUB_TH = {"queue_onset": 0.55, "queue_ongoing": 0.40}


class Pub:
    """Public probabilities for one panel, returned in our travel link order."""

    def __init__(self, panel, pno, ours: PanelT2):
        self.on = PubOnsetPanel(panel, pno); self.og = PubOngoingPanel(panel, pno)
        pos = {l: i for i, l in enumerate(self.on.links)}
        self.perm = np.array([pos[l] for l in ours.links])  # travel index -> fd index

    def probs(self, speed, flow, elig, origin, cond):
        """speed/flow/elig: (12, L) history T-60..T-5 in fd order. Returns (6, L) travel order."""
        L = len(self.on.links); P = np.zeros((HOR, L))
        if cond == "queue_onset":
            X = self.on.features(speed, flow, elig, origin, cond)[-L:]
            P[-1] = 0.5 * PUB_ON_LGB.predict(X) + 0.5 * PUB_ON_CAT.predict_proba(X)[:, 1]
        else:
            X = self.og.features(speed, flow, np.nan_to_num(elig, nan=0).astype(bool), origin, cond)
            P = PUB_OG_LGB.predict(X).reshape(HOR, L)
        return P[:, self.perm]


def decide(P, cond, th):
    pred = P >= th
    if cond == "queue_onset":
        pred[:-1] = False
        if not pred[-1].any():
            pred[-1, np.argmax(P[-1])] = True
    return pred


def feb():
    rows, names = pickle.load(open(CACHE / "t2_rows_T1.pkl", "rb"))
    cols = [i for i, n in enumerate(names) if not n.startswith("mmem")] + [len(names)]
    res = []
    for cond in ("queue_onset", "queue_ongoing"):
        onset = cond == "queue_onset"
        X, y = stack([w for w in rows[cond] if w["month"] not in HOLDOUT], onset)
        params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.7,
                      bagging_fraction=0.8, bagging_freq=1, lambda_l2=10, num_threads=12, verbosity=-1, seed=1)
        model = lgb.train(params, lgb.Dataset(X[:, cols], y), num_boost_round=650 if onset else 850)
        ws = [w for w in rows[cond] if w["month"] == "2031-02"]
        if not onset and os.environ.get("PERS_T") == "1":
            # official selector may measure persistence from the row at T rather than T-5: keep IoU <= 0.9
            keep, loaded = [], {}
            for w in ws:
                if w["panel"] not in loaded:
                    loaded = {w["panel"]: (PanelT2(w["panel"], 0), load(w["panel"], "train", masked=False)["speed"])}
                Pn, sp = loaded[w["panel"]]
                qT = Pn.queued(Pn.orient(sp[w["day"], w["t"]]))
                if iou(np.repeat(qT[None], HOR, 0) & w["ok"], w["y"] & w["ok"]) <= 0.9:
                    keep.append(w)
            print(f"PERS_T filter keeps {len(keep)}/{len(ws)} ongoing windows", flush=True)
            ws = keep
        cache = {}
        for pno, panel in enumerate(p for p in panels() if p not in TASK2_EXCLUDED):
            mine = [w for w in ws if w["panel"] == panel]
            if not mine:
                continue
            P = PanelT2(panel, pno); pub = Pub(panel, pno, P)
            z = load(panel, "train", masked=False)
            fd_of_mp = {l: i for i, l in enumerate(pub.on.links)}
            mp_links = __import__("common").meta(panel).link_id.to_numpy()
            to_fd = np.array([np.where(mp_links == l)[0][0] for l in pub.on.links])  # fd index -> milepost index
            for w in mine:
                d, t = w["day"], w["t"]
                sp = z["speed"][d, t - HIST:t][:, to_fd]; fl = z["flow"][d, t - HIST:t][:, to_fd]
                el = z["eligible"][d, t - HIST:t][:, to_fd].astype(np.float32)
                origin = pd.Timestamp(z["dates"][d]) + pd.Timedelta(minutes=5 * t)
                pp = pub.probs(sp, fl, el, origin, cond)
                L = w["X"].shape[0]
                po = np.stack([model.predict(np.column_stack([w["X"], np.full(L, h + 1, np.float32)])[:, cols]) for h in range(HOR)])
                cache[id(w)] = (pp, po)
        for wt in (0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0):  # weight on ours
            for th in (0.35, 0.4, 0.45, 0.5, 0.55):
                s = []
                for w in ws:
                    if id(w) not in cache:
                        continue
                    pp, po = cache[id(w)]
                    pred = decide(wt * po + (1 - wt) * pp, cond, th)
                    s.append(iou(pred & w["ok"], w["y"] & w["ok"]))
                res.append(dict(cond=cond, w_ours=wt, th=th, iou=np.mean(s), n=len(s)))
        r = pd.DataFrame([x for x in res if x["cond"] == cond])
        pubref = np.mean([iou(decide(cache[id(w)][0], cond, PUB_TH[cond]) & w["ok"], w["y"] & w["ok"]) for w in ws if id(w) in cache])
        print(cond, "public as shipped:", round(pubref, 4))
        print(r.pivot(index="w_ours", columns="th", values="iou").round(4).to_string(), flush=True)
    pd.DataFrame(res).to_csv(CACHE / "t2_blend_feb.csv", index=False)


def apply(w_ours: dict, th: dict):
    out = []
    for split in ("validation", "private"):
        ours = pd.read_csv(CACHE / f"t2_pred_{split}.csv", dtype={"link_id": str})
        for pno, panel in enumerate(p for p in panels() if p not in TASK2_EXCLUDED):
            P = PanelT2(panel, pno); pub = Pub(panel, pno, P)
            wi = pd.read_csv(RELEASE / "task2" / panel / split / "window_index.csv")
            wh = pd.read_parquet(RELEASE / "task2" / panel / split / "window_history.parquet")
            wh["link_id"] = wh.link_id.astype(str)
            for w in wi.itertuples():
                origin = pd.Timestamp(w.forecast_origin)
                times = pd.date_range(end=origin - pd.Timedelta(minutes=5), periods=HIST, freq="5min").strftime("%Y-%m-%dT%H:%M:%SZ")
                h = wh[wh.window_id == w.window_id]
                mat = lambda c: h.pivot(index="timestamp", columns="link_id", values=c).reindex(index=times, columns=pub.on.links).to_numpy(np.float32)
                pp = pub.probs(mat("speed_kmh"), mat("flow_vph"), mat("is_score_eligible"), origin, w.condition)
                o = ours[ours.window_id == w.window_id]
                stamps = sorted(o.timestamp.unique())
                po = np.stack([o[o.timestamp == s].set_index("link_id").prob.reindex(P.links).to_numpy() for s in stamps])
                pred = decide(w_ours[w.condition] * po + (1 - w_ours[w.condition]) * pp, w.condition, th[w.condition])
                for k, s in enumerate(stamps):
                    out.append(pd.DataFrame({"window_id": w.window_id, "timestamp": s, "link_id": P.links, "queue_pred": pred[k].astype(int)}))
    d = pd.concat(out, ignore_index=True)
    d.to_csv(CACHE / "t2_pred_blend.csv", index=False)
    print("blend rows", len(d), "positive share", d.queue_pred.mean().round(4))


if __name__ == "__main__":
    if sys.argv[1] == "feb":
        feb()
    else:
        apply({"queue_onset": float(sys.argv[2]), "queue_ongoing": float(sys.argv[3])},
              {"queue_onset": float(sys.argv[4]), "queue_ongoing": float(sys.argv[5])})
