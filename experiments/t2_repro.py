# LOCAL ANALYSIS ONLY: not used by run_all.py. Reads intermediate outputs of our Kaggle runs under $TFB_WORKSPACE (not in this repo).
"""Rebuild the yeashusemwal notebook's Task 2 from our retrained members and compare with its published queue labels.

onset   : public anubisyy labels (anubisyy), unchanged
ongoing : 0.5 * p_members + 0.5 * p_public_ongoing >= 0.45, where
          p_members = mean over (p_general, p_general_w) of 0.8*group + 0.2*v20,
          group = 0.5*mean(v11..v14) + 0.5*mean(v15, v16)   (missing members are skipped)
Writes cache/t2_pred_repro.csv in template format (window_id, timestamp, link_id, queue_pred)."""
import glob
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import CACHE, RELEASE

WS = __import__("os").environ.get("TFB_WORKSPACE", ".")  # folder holding our Kaggle run outputs

M = WS + r"/kaggle_kernels/t2_members"
LOCAL = WS + r"/t2repro"
os.environ.setdefault("TFB_DATA_ROOT", str(RELEASE)); os.environ.setdefault("TFB_OUTPUT_ROOT", str(CACHE / "public_tmp"))
sys.path.insert(0, WS + r"/public_assets")
from queue_features import panel_names, HISTORY, HORIZON  # noqa: E402
from queue_ongoing import Panel as OngoingPanel  # noqa: E402

key = ["window_id", "h", "link_id"]


def member(name):
    f = glob.glob(rf"{M}\{name}\out\**\window_probs.parquet", recursive=True) + glob.glob(rf"{LOCAL}\t2_{name}\window_probs.parquet")
    if not f:
        return None
    d = pd.read_parquet(f[0]); d = d[d.split != "train"].copy(); d["link_id"] = d.link_id.astype(str)
    return d.set_index(key).sort_index()


P = {n: member(n) for n in ("v11", "v12", "v13", "v14", "v15", "v16", "v20")}
have = [n for n, d in P.items() if d is not None]
print("members available:", have)
base = P[have[0]]
mean = lambda names, col: np.mean([P[n][col].reindex(base.index).to_numpy() for n in names], 0)
A = [n for n in ("v11", "v12", "v13", "v14") if n in have]; B = [n for n in ("v15", "v16") if n in have]
cols = {}
for c in ("p_general", "p_general_w"):
    grp = 0.5 * mean(A, c) + 0.5 * mean(B, c) if B else mean(A, c)
    cols[c] = 0.8 * grp + 0.2 * mean(["v20"], c) if "v20" in have else grp
p_members = pd.Series((cols["p_general"] + cols["p_general_w"]) / 2, index=base.index)

# public anubisyy ongoing probabilities, recomputed from its released checkpoint (strictly before T)
og = lgb.Booster(model_file=WS + r"/public_assets/ongoing_lgb.txt"); rows = []
for split in ("validation", "private"):
    for no, name in enumerate(panel_names()):
        panel = OngoingPanel(name, no); folder = RELEASE / "task2" / name / split
        wi = pd.read_csv(folder / "window_index.csv"); hist = pd.read_parquet(folder / "window_history.parquet")
        for w in wi[wi.condition == "queue_ongoing"].itertuples(index=False):
            origin = pd.Timestamp(w.forecast_origin)
            times = pd.date_range(end=origin - pd.Timedelta(minutes=5), periods=HISTORY, freq="5min").strftime("%Y-%m-%dT%H:%M:%SZ")
            h = hist[hist.window_id == w.window_id].copy(); h["link_id"] = h.link_id.astype(str)
            mat = lambda c: h.pivot(index="timestamp", columns="link_id", values=c).reindex(index=times, columns=panel.links).to_numpy(np.float32)
            p = og.predict(panel.features(mat("speed_kmh"), mat("flow_vph"), np.nan_to_num(mat("is_score_eligible"), nan=0).astype(bool), origin, w.condition))
            rows.append(pd.DataFrame({"window_id": w.window_id, "h": np.repeat(np.arange(1, HORIZON + 1), len(panel.links)),
                                      "link_id": np.tile(panel.links, HORIZON), "p_pub": p}))
pub = pd.concat(rows).set_index(key)

k = pd.read_csv(RELEASE / "submission_key.csv", usecols=["submission_id", "task", "window_id", "timestamp", "link_id"], dtype=str)
k = k[k.task == "queue"].copy()
wi = pd.concat([pd.read_csv(g) for g in glob.glob(str(RELEASE / "task2" / "*" / "*" / "window_index.csv"))])
k["h"] = ((pd.to_datetime(k.timestamp) - pd.to_datetime(k.window_id.map(dict(zip(wi.window_id, wi.forecast_origin))))).dt.total_seconds() // 300).astype(int)
k["cond"] = k.window_id.map(dict(zip(wi.window_id, wi.condition)))
idx = pd.MultiIndex.from_frame(k[key])
k["p_mem"] = p_members.reindex(idx).to_numpy(); k["p_pub"] = pub.p_pub.reindex(idx).to_numpy()
sid = k.submission_id.astype(int)
k["q_pub"] = pd.read_csv(WS + r"/public_outputs/catboost/submission.csv", usecols=["submission_id", "queue_pred"]).set_index("submission_id").queue_pred.reindex(sid).to_numpy()
k["q_nb"] = pd.read_csv(WS + r"/public_outputs/unet/submission.csv", usecols=["submission_id", "queue_pred"]).set_index("submission_id").queue_pred.reindex(sid).to_numpy()
ong = k.cond == "queue_ongoing"
assert k.loc[ong, ["p_mem", "p_pub"]].notna().all().all()
k["queue_pred"] = k.q_pub
k.loc[ong, "queue_pred"] = (0.5 * k.p_mem[ong] + 0.5 * k.p_pub[ong] >= 0.45).astype(int)

# fidelity vs the published yeashusemwal labels
o = k[ong]
agree = (o.queue_pred == o.q_nb).mean()
inter = ((o.queue_pred == 1) & (o.q_nb == 1)).groupby(o.window_id).sum(); union = ((o.queue_pred == 1) | (o.q_nb == 1)).groupby(o.window_id).sum()
print(f"ongoing cells: agreement with yeashusemwal {agree:.4%}; mean per-window IoU vs yeashusemwal {(inter/union.clip(lower=1)).mean():.4f}; "
      f"anubisyy vs yeashusemwal agreement {(o.q_pub == o.q_nb).mean():.4%}")
print("onset cells identical to yeashusemwal:", f"{(k[~ong].queue_pred == k[~ong].q_nb).mean():.4%}")
k[["window_id", "timestamp", "link_id", "queue_pred"]].to_csv(CACHE / "t2_pred_repro.csv", index=False)
