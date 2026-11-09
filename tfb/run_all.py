"""Single entry script: rebuild the final TrafficFlowBench submission from the release and the saved models.

  Task 1  axial-NN imputers (v1 seeds 1-2, v2 seeds 1-8) + CatBoost residual ensemble, combined per cell (t1_combine);
          inside the Task 2 blank blocks 0.65 * NN v3 (seeds 5-10, trained on block cells) + 0.35 * LightGBM gap-filler
          (with the on-ramp jam sensor; released ramp flows inside the block, Task 1 only)
  Task 2  onset: public anubisyy onset models (0.5 LightGBM + 0.5 CatBoost, T+30 only, threshold 0.55, top-1 fallback);
          ongoing: 0.5 * our retrained 7-member ensemble of the yeashusemwal notebook + 0.5 * public anubisyy ongoing LightGBM >= 0.45
  Task 4  prior-regularised L2 ODME on each split's own counts and prior (lambda = 0.05)

Env: TFB_DATA_ROOT (kaggle_public), TFB_SOLUTION (folder holding models/), TFB_OUT (writable work dir),
     TFB_CODE (repository root holding tfb/ and third_party/; default: the parent of this script's folder).
Usage: python run_all.py [--stages t1,t2,t4,merge]
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(os.environ["TFB_DATA_ROOT"])
SOL = Path(os.environ["TFB_SOLUTION"])
OUT = Path(os.environ["TFB_OUT"]); OUT.mkdir(parents=True, exist_ok=True)
CODE, MODELS = Path(os.environ.get("TFB_CODE", Path(__file__).resolve().parents[1])), SOL / "models"
SPLITS = ("validation", "private")
# all 16 trained checkpoints; TFB_NN_WEIGHTS overrides (comma list)
NN_WEIGHTS = os.environ.get("TFB_NN_WEIGHTS", "v1_seed1,v1_seed2,v2_seed1,v2_seed2,v2_seed3,v2_seed4,v2_seed5,v2_seed6,v2_seed7,v2_seed8,v3_seed5,v3_seed6,v3_seed7,v3_seed8,v3_seed9,v3_seed10").split(",")
W_NN = float(os.environ.get("TFB_W_NN", "0.8"))  # NN share outside the blank blocks (rest CatBoost); 0.8 = final submission
KEY1 = ["panel", "timestamp", "station_id", "link_id", "mask_regime"]
T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def task2_days():
    wi = pd.concat([pd.read_csv(f) for f in glob.glob(str(DATA / "task2" / "*" / "*" / "window_index.csv"))])
    return set(zip(wi.panel, wi.date)), wi


# ------------------------------------------------------------------ Task 1
def t1_nn():
    os.environ["TFB_OUT"] = str(OUT / "nn")
    spec = importlib.util.spec_from_file_location("t1nn", CODE / "tfb" / "t1_nn.py"); nnm = importlib.util.module_from_spec(spec); spec.loader.exec_module(nnm)
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    P = {p: nnm.PanelData(p) for p in nnm.panels()}
    for w in NN_WEIGHTS:
        net = nnm.build_model().to(dev); net.load_state_dict(torch.load(MODELS / "nn" / f"{w}.pt", map_location=dev))
        for split in SPLITS:
            nnm.predict_split(net, P, dev, split, OUT / f"t1_{w}_{split}.csv")
        log("NN", w, "done")
    os.environ["TFB_OUT"] = str(OUT)


def t1_catboost():
    work = OUT / "cat_src"; shutil.copytree(CODE / "third_party" / "public_src", work, dirs_exist_ok=True)
    os.environ.update(TFB_DATA_ROOT=str(DATA), TFB_OUTPUT_ROOT=str(OUT / "cat_out"), TFB_THREADS="4")
    sys.path.insert(0, str(work))
    import state_model as base
    from prepare_state import features
    from state_inference import StatePredictor
    pred = StatePredictor(MODELS / "catboost")
    for split in SPLITS:
        parts = []
        for pi, name in enumerate(base.all_panels()):
            panel = base.Panel(name, pi); frame = base.month_frame(name, "", split=split)
            times, v, q, e = base.matrices(frame, panel)
            tpl = pd.read_csv(DATA / "task1" / name / split / "sample_submission_state.csv", dtype={"station_id": str, "link_id": str})
            rows = times.get_indexer(tpl.timestamp); cols = pd.Index(panel.links).get_indexer(tpl.link_id)
            X, bv, bq = features(panel, pi, times, v, q, e, rows, cols, tpl.mask_regime.map({"R1": 0, "R2": 1, "R3": 2}).to_numpy())
            tpl["speed_kmh"] = np.clip(bv + pred.residual(X, "speed"), 1, 180)
            tpl["flow_vph"] = np.clip((bq + pred.residual(X, "flow")) * panel.lanes[cols], 0, 150000)
            parts.append(tpl)
        pd.concat(parts).to_csv(OUT / f"t1_catboost_{split}.csv", index=False); log("CatBoost", split, "done")


def t1_combine():
    """Per cell: CatBoost inside Task 2 blank blocks (T+5..T+90, no link reports), NN v2 elsewhere on
    Task-2 days, all v1+v2 NN seeds on normal days; NN parts are blended W_NN (0.8) NN + (1 - W_NN) CatBoost."""
    t2d, wi = task2_days()
    rd = lambda f: pd.read_csv(f, usecols=KEY1 + ["speed_kmh", "flow_vph"], dtype={"station_id": str, "link_id": str}).set_index(KEY1)
    for split in SPLITS:
        cb = rd(OUT / f"t1_catboost_{split}.csv")
        n1, n2 = (sum(w.startswith(v) for w in NN_WEIGHTS) for v in ("v1", "v2"))
        v1 = sum(rd(OUT / f"t1_{w}_{split}.csv").reindex(cb.index) for w in NN_WEIGHTS if w.startswith("v1")) / n1
        v2 = sum(rd(OUT / f"t1_{w}_{split}.csv").reindex(cb.index) for w in NN_WEIGHTS if w.startswith("v2")) / n2
        panel = cb.index.get_level_values("panel"); ts = pd.Series(cb.index.get_level_values("timestamp"))
        day = ts.str[:10].to_numpy(); slot = (ts.str[11:13].astype(int) * 12 + ts.str[14:16].astype(int) // 5).to_numpy()
        t2day = pd.Series(list(zip(panel, day))).isin(t2d).to_numpy()
        inblk = np.zeros(len(cb), bool)
        for w in wi[wi.split == split].itertuples():
            o = int(w.forecast_origin[11:13]) * 12 + int(w.forecast_origin[14:16]) // 5
            inblk |= (panel == w.panel) & (day == w.date) & (slot > o) & (slot <= o + 18)
        nn = (n1 * v1 + n2 * v2) / (n1 + n2)
        nn[t2day] = v2[t2day]
        out = W_NN * nn + (1 - W_NN) * cb
        out[inblk] = cb[inblk]
        out.reset_index().to_csv(OUT / f"task1_{split}.csv", index=False)
        log("Task 1 combined", split, f"in-block share {inblk.mean():.4f}")


W_V3 = float(os.environ.get("TFB_W_V3", "0.65"))  # in-block weight on NN v3 (rest gap-filler)


def t1_gapfill():
    """Train the dedicated LightGBM gap-filler (t1_gapfill.py final); target cells inside Task 2 blank blocks get
    0.35 * gap-filler + 0.65 * NN v3 (seed mean)."""
    gf = OUT / "gapfill"; gf.mkdir(exist_ok=True)
    env = dict(os.environ, TFB_DATA_ROOT=str(DATA), TFB_CACHE=str(gf), TFB_GF_RAMP=os.environ.get("TFB_GF_RAMP", "1"))  # 1: on-ramp jam sensor on (final); 0: off
    if not all((gf / f"t1_gapfill_{s}.csv").exists() for s in SPLITS):  # reused when only the blend weights change (--stages t1mix)
        subprocess.run([sys.executable, str(CODE / "tfb" / "t1_gapfill.py"), "final"], cwd=str(CODE / "tfb"), env=env, check=True)
    key = ["panel", "timestamp", "link_id"]
    for split in SPLITS:
        base = pd.read_csv(OUT / f"task1_{split}.csv", dtype={"station_id": str, "link_id": str})
        g = pd.read_csv(gf / f"t1_gapfill_{split}.csv", dtype={"link_id": str}).rename(columns={"speed_kmh": "v_gf", "flow_vph": "q_gf"})
        m = base.merge(g, on=key, how="left"); hit = m.v_gf.notna()
        v3 = sum(pd.read_csv(OUT / f"t1_{w}_{split}.csv", usecols=KEY1 + ["speed_kmh", "flow_vph"], dtype={"station_id": str, "link_id": str})
                 .set_index(KEY1) for w in NN_WEIGHTS if w.startswith("v3")) / sum(w.startswith("v3") for w in NN_WEIGHTS)
        v3 = v3.reindex(pd.MultiIndex.from_frame(m[KEY1]))
        assert v3[hit.to_numpy()].notna().all().all()
        m.loc[hit, "speed_kmh"] = (1 - W_V3) * m.v_gf[hit] + W_V3 * v3.speed_kmh.to_numpy()[hit.to_numpy()]
        m.loc[hit, "flow_vph"] = (1 - W_V3) * m.q_gf[hit] + W_V3 * v3.flow_vph.to_numpy()[hit.to_numpy()]
        m.drop(columns=["v_gf", "q_gf"]).to_csv(OUT / f"task1_{split}.csv", index=False)
        log("Task 1 gap-filler", split, "cells", int(hit.sum()))


# ------------------------------------------------------------------ Task 2
def task2():
    import lightgbm as lgb
    from catboost import CatBoostClassifier
    work = OUT / "t2_src"; shutil.copytree(CODE / "third_party" / "t2_src", work, dirs_exist_ok=True)
    pub = OUT / "pub_src"; shutil.copytree(CODE / "third_party" / "public_src", pub, dirs_exist_ok=True)
    os.environ.update(TFB_DATA=str(DATA), TFB_DATA_ROOT=str(DATA), TFB_OUTPUT_ROOT=str(OUT / "pub_out"))
    sys.path[:0] = [str(work / "nb"), str(work / "v2"), str(pub)]
    import t2_gbm_infer as g2
    P = {}
    for family, names in (("v11", ["v11", "v12", "v13", "v14"]), ("v15", ["v15", "v16"]), ("v18", ["v20"])):
        dirs = [str(MODELS / "t2_members" / f"t2_{n}") for n in names]
        res = g2.dump(family, dirs, threads=4)
        P.update({n: res[d] for n, d in zip(names, dirs)}); log("Task 2 members", family, "done")
    key = ["window_id", "h", "link_id"]
    base_ = P["v11"].copy(); base_["link_id"] = base_.link_id.astype(str); idx = pd.MultiIndex.from_frame(base_[key])
    mean = lambda names, col: np.mean([P[n][col].to_numpy() for n in names], 0)
    cols = []
    for c in ("p_general", "p_general_w"):
        grp = 0.5 * mean(["v11", "v12", "v13", "v14"], c) + 0.5 * mean(["v15", "v16"], c)
        cols.append(0.8 * grp + 0.2 * P["v20"][c].to_numpy())
    p_mem = pd.Series(np.mean(cols, 0), index=idx)

    from queue_features import Panel as OnsetPanel, panel_names, HISTORY, HORIZON
    from queue_ongoing import Panel as OngoingPanel
    on_lgb = lgb.Booster(model_file=str(MODELS / "public_queue" / "onset_lgb.txt"))
    on_cat = CatBoostClassifier(); on_cat.load_model(str(MODELS / "public_queue" / "onset.cbm"))
    og_lgb = lgb.Booster(model_file=str(MODELS / "public_queue" / "ongoing_lgb.txt"))
    out = []
    for split in SPLITS:
        for no, name in enumerate(panel_names()):
            onp, ogp = OnsetPanel(name, no), OngoingPanel(name, no); folder = DATA / "task2" / name / split
            wi = pd.read_csv(folder / "window_index.csv"); hist = pd.read_parquet(folder / "window_history.parquet")
            for w in wi.itertuples(index=False):
                origin = pd.Timestamp(w.forecast_origin)
                times = pd.date_range(end=origin - pd.Timedelta(minutes=5), periods=HISTORY, freq="5min").strftime("%Y-%m-%dT%H:%M:%SZ")
                h = hist[hist.window_id == w.window_id].copy(); h["link_id"] = h.link_id.astype(str)
                assert pd.to_datetime(h.timestamp, utc=True).max() < pd.Timestamp(w.forecast_origin, tz="UTC") if pd.Timestamp(w.forecast_origin).tzinfo is None \
                    else pd.to_datetime(h.timestamp, utc=True).max() < pd.Timestamp(w.forecast_origin)  # strictly before T
                mat = lambda c: h.pivot(index="timestamp", columns="link_id", values=c).reindex(index=times, columns=onp.links).to_numpy(np.float32)
                sp, fl, el = mat("speed_kmh"), mat("flow_vph"), mat("is_score_eligible")
                L = len(onp.links); pred = np.zeros((HORIZON, L), int)
                if w.condition == "queue_onset":
                    X = onp.features(sp, fl, el, origin, w.condition)[-L:]
                    prob = 0.5 * on_lgb.predict(X) + 0.5 * on_cat.predict_proba(X)[:, 1]
                    sel = prob >= 0.55
                    if not sel.any():
                        sel[np.argmax(prob)] = True
                    pred[-1] = sel
                else:
                    p_pub = og_lgb.predict(ogp.features(sp, fl, np.nan_to_num(el, nan=0).astype(bool), origin, w.condition)).reshape(HORIZON, L)
                    mi = pd.MultiIndex.from_arrays([np.repeat(w.window_id, HORIZON * L), np.repeat(np.arange(1, HORIZON + 1), L), np.tile(onp.links, HORIZON)])
                    pm = p_mem.reindex(mi).to_numpy().reshape(HORIZON, L)
                    assert np.isfinite(pm).all(), w.window_id
                    pred = (0.5 * pm + 0.5 * p_pub >= 0.45).astype(int)
                stamps = [(origin + pd.Timedelta(minutes=5 * s)).strftime("%Y-%m-%dT%H:%M:%SZ") for s in range(1, HORIZON + 1)]
                out.append(pd.DataFrame({"window_id": w.window_id, "timestamp": np.repeat(stamps, L), "link_id": np.tile(onp.links, HORIZON),
                                         "queue_pred": pred.ravel()}))
    pd.concat(out).to_csv(OUT / "task2.csv", index=False); log("Task 2 written")


# ------------------------------------------------------------------ Task 4
def task4():
    # the Task 2 source tree also ships a module named "common"; make sure ours is the one imported here
    sys.modules.pop("common", None)
    sys.path.insert(0, str(CODE / "tfb")); os.environ["TFB_DATA_ROOT"] = str(DATA)
    import t4_variants as tv
    tv.LAM = 0.05
    parts = {s: [] for s in SPLITS}
    for panel in tv.panels():
        ps, A, links, counts, priors = tv.load(panel)
        for s in SPLITS:
            f = tv.solve_l2(A, counts[s].reindex(links).to_numpy(float), priors[s])
            tpl = pd.read_csv(DATA / "task4" / panel / s / "sample_submission_path_flow.csv", dtype=str)
            tpl["path_flow"] = tpl.path_id.map(dict(zip(ps.path_id, f))).astype(float).clip(lower=0); parts[s].append(tpl)
    for s in SPLITS:
        pd.concat(parts[s]).to_csv(OUT / f"task4_{s}.csv", index=False)
    log("Task 4 written")


def merge():
    env = dict(os.environ, TFB_DATA_ROOT=str(DATA), TFB_CACHE=str(OUT / "cache"))
    cmd = [sys.executable, str(CODE / "tfb" / "make_submission.py"), "--base", str(DATA / "sample_submission.csv"),
           "--state", f"1|{OUT / 'task1_validation.csv'}|{OUT / 'task1_private.csv'}",
           "--queue", str(OUT / "task2.csv"), "--odme", str(OUT / "task4_validation.csv"), str(OUT / "task4_private.csv"),
           "--out", str(OUT / "submission.csv")]
    subprocess.run(cmd, check=True, env=env); log("submission.csv written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stages", default="t1,t2,t4,merge"); a = ap.parse_args()
    st = a.stages.split(",")
    if "t1" in st:
        t1_nn(); t1_catboost(); t1_combine(); t1_gapfill()
    if "t1mix" in st:  # re-blend Task 1 from an earlier full run in the same TFB_OUT (NN/CatBoost/gap-filler outputs kept)
        t1_combine(); t1_gapfill()
    if "t2" in st:
        task2()
    if "t4" in st:
        task4()
    if "merge" in st:
        merge()
