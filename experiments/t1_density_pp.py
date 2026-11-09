# LOCAL ANALYSIS ONLY: not used by run_all.py. Reads intermediate outputs of our Kaggle runs under $TFB_WORKSPACE (not in this repo).
"""Short-gap density post-process for Task 1 (idea from the public honghanhhh notebook), tuned on February.

For a target cell inside a short run of hidden cells on one link (run <= MAX_RUN, released cells on both sides),
the predicted density k = q/v is pulled toward the log-linear interpolation of the released densities at the two
neighbours: log k' = (1-beta) log k + beta log k_interp. The ratio r = k'/k is split between flow and speed:
q *= r**alpha, v *= r**(alpha-1).
Usage: python t1_density_pp.py eval            (February grid on the 0.9 NN + 0.1 CatBoost blend)
       python t1_density_pp.py apply <tag_in> <tag_out> beta alpha max_run   (cache/state_{split}_<tag>.csv)
"""
import sys

import numpy as np
import pandas as pd

from common import load, meta, panels, slot_of
from t1_common import targets


WS = __import__("os").environ.get("TFB_WORKSPACE", ".")  # folder holding our Kaggle run outputs

def neighbours(obs_ok):
    """For (D, S, L) bool of released cells: index of previous/next released slot along S (-1 / S if none)."""
    D, S, L = obs_ok.shape
    idx = np.arange(S)[None, :, None]
    prev = np.where(obs_ok, idx, -1); prev = np.maximum.accumulate(prev, axis=1)
    nxt = np.where(obs_ok, idx, S); nxt = np.minimum.accumulate(nxt[:, ::-1], axis=1)[:, ::-1]
    # exclude the cell itself: shift by one
    prev = np.concatenate([np.full((D, 1, L), -1), prev[:, :-1]], axis=1)
    nxt = np.concatenate([nxt[:, 1:], np.full((D, 1, L), S)], axis=1)
    return prev, nxt


def adjust(pv, pq, ov, oq, tgt, beta, alpha, max_run):
    """pv/pq predictions, ov/oq released data (NaN where hidden), tgt target mask; all (D, S, L)."""
    ok = np.isfinite(ov) & np.isfinite(oq) & (ov > 1) & (oq > 0)
    prev, nxt = neighbours(ok)
    S = ok.shape[1]
    run = nxt - prev - 1
    sel = tgt & (prev >= 0) & (nxt < S) & (run <= max_run) & np.isfinite(pv) & np.isfinite(pq) & (pv > 1) & (pq > 0)
    d, s, j = np.nonzero(sel)
    p0, n0 = prev[d, s, j], nxt[d, s, j]
    lk0 = np.log(oq[d, p0, j] / ov[d, p0, j]); lk1 = np.log(oq[d, n0, j] / ov[d, n0, j])
    w = (s - p0) / (n0 - p0)
    lki = (1 - w) * lk0 + w * lk1
    lk = np.log(pq[d, s, j] / pv[d, s, j])
    lr = beta * (lki - lk)
    pv, pq = pv.copy(), pq.copy()
    pq[d, s, j] *= np.exp(alpha * lr); pv[d, s, j] *= np.exp((alpha - 1) * lr)
    return pv, pq, len(d)


def feb_eval():
    from t2_data import downstream_order
    K = WS + r"/kaggle_kernels"

    def nn_frame(path):
        z = np.load(path); parts = []
        for p in panels():
            dates = load(p, "train", masked=True)["dates"]
            links = meta(p).link_id.to_numpy()[downstream_order(p)]
            d, s, j = z[f"{p}_d"], z[f"{p}_s"], z[f"{p}_j"]
            ts = pd.Series(dates[d]).str.cat(pd.Series([f"T{x//12:02d}:{(x%12)*5:02d}:00Z" for x in s]))
            parts.append(pd.DataFrame({"panel": p, "timestamp": ts, "link_id": links[j], "speed_kmh": z[f"{p}_v"], "flow_vph": z[f"{p}_q"]}))
        return pd.concat(parts, ignore_index=True)
    from t3_proxy import evaluate
    key = ["panel", "timestamp", "link_id"]
    cb = pd.read_csv(K + r"/t1_cat/out_v2/feb_state_catboost.csv", usecols=key + ["speed_kmh", "flow_vph"], dtype={"link_id": str})
    files = [K + r"/t1_nn/out_v1/feb_pred_seed1.npz", K + r"/t1_nn/out_v1/feb_pred_seed2.npz", K + r"/t1_nn/out/feb_pred_seed1.npz",
             K + r"/t1_nn/out/feb_pred_seed2.npz", K + r"/t1_nn_b/out/feb_pred_seed3.npz", K + r"/t1_nn_b/out/feb_pred_seed4.npz"]
    v = q = None
    for f in files:
        n = cb[key].merge(nn_frame(f), on=key, how="left")
        assert len(n) == len(cb) and n.speed_kmh.notna().all()
        v = n.speed_kmh.to_numpy() if v is None else v + n.speed_kmh.to_numpy()
        q = n.flow_vph.to_numpy() if q is None else q + n.flow_vph.to_numpy()
    fr = cb[key].copy()
    fr["speed_kmh"] = 0.9 * v / len(files) + 0.1 * cb.speed_kmh.to_numpy()
    fr["flow_vph"] = 0.9 * q / len(files) + 0.1 * cb.flow_vph.to_numpy()
    obs = {p: load(p, "train", masked=True) for p in panels()}

    def tf(beta, alpha, mr):
        def f(p, pv, pq, tr, tgt, days):
            o = obs[p]
            pv2, pq2, n = adjust(pv, pq, o["speed"], o["flow"], tgt, beta, alpha, mr)
            return pv2, pq2
        return f
    evaluate(fr, "base 0.9 NN(6) + 0.1 CB")
    for mr in (1, 3, 6):
        for beta in (0.3, 0.6, 1.0):
            for alpha in (0.0, 0.5, 1.0):
                evaluate(fr, f"mr={mr} beta={beta} alpha={alpha}", transform=tf(beta, alpha, mr))


def v3eval():
    """February: does adding NN v3 (seeds 5, 6) to the normal-day NN average help? Final = 0.9 ((1-w) NN6 + w v3) + 0.1 CB."""
    from t3_proxy import evaluate
    from t2_data import downstream_order
    K = WS + r"/kaggle_kernels"
    key = ["panel", "timestamp", "link_id"]
    cb = pd.read_csv(K + r"/t1_cat/out_v2/feb_state_catboost.csv", usecols=key + ["speed_kmh", "flow_vph"], dtype={"link_id": str})

    def nn_mean(files):
        v = q = 0
        for f in files:
            z = np.load(f); parts = []
            for p in panels():
                dates = load(p, "train", masked=True)["dates"]; links = meta(p).link_id.to_numpy()[downstream_order(p)]
                d, s_, j = z[f"{p}_d"], z[f"{p}_s"], z[f"{p}_j"]
                ts = pd.Series(dates[d]).str.cat(pd.Series([f"T{x//12:02d}:{(x%12)*5:02d}:00Z" for x in s_]))
                parts.append(pd.DataFrame({"panel": p, "timestamp": ts, "link_id": links[j], "v": z[f"{p}_v"], "q": z[f"{p}_q"]}))
            n = cb[key].merge(pd.concat(parts, ignore_index=True), on=key, how="left")
            assert n.v.notna().all()
            v = v + n.v.to_numpy() / len(files); q = q + n.q.to_numpy() / len(files)
        return v, q
    v6, q6 = nn_mean([K + r"/t1_nn/out_v1/feb_pred_seed1.npz", K + r"/t1_nn/out_v1/feb_pred_seed2.npz", K + r"/t1_nn/out/feb_pred_seed1.npz",
                      K + r"/t1_nn/out/feb_pred_seed2.npz", K + r"/t1_nn_b/out/feb_pred_seed3.npz", K + r"/t1_nn_b/out/feb_pred_seed4.npz"])
    v3, q3 = nn_mean([K + r"/t1_nn_v3/out/feb_pred_seed5.npz", K + r"/t1_nn_v3/out/feb_pred_seed6.npz"])
    for w in (0.0, 0.25, 0.4, 0.6):
        fr = cb[key].copy()
        fr["speed_kmh"] = 0.9 * ((1 - w) * v6 + w * v3) + 0.1 * cb.speed_kmh.to_numpy()
        fr["flow_vph"] = 0.9 * ((1 - w) * q6 + w * q3) + 0.1 * cb.flow_vph.to_numpy()
        evaluate(fr, f"v3 share of NN = {w}")


def apply(tag_in, tag_out, beta, alpha, mr):
    """cache/state_{split}_{tag_in}.csv -> cache/state_{split}_{tag_out}.csv with the post-process on the target cells."""
    from common import CACHE
    for split in ("validation", "private"):
        f = pd.read_csv(CACHE / f"state_{split}_{tag_in}.csv", dtype={"station_id": str, "link_id": str})
        tot = 0
        for p in panels():
            m = (f.panel == p).to_numpy()
            if not m.any():
                continue
            o = load(p, split, masked=True); tgt = targets(p, split)
            dates = {d: i for i, d in enumerate(o["dates"])}; col = {l: i for i, l in enumerate(meta(p).link_id)}
            g = f[m]
            d = g.timestamp.str[:10].map(dates).to_numpy(); s_ = g.timestamp.map(slot_of).to_numpy(); j = g.link_id.map(col).to_numpy()
            pv = np.full(o["speed"].shape, np.nan, np.float64); pq = pv.copy()
            pv[d, s_, j] = g.speed_kmh.to_numpy(); pq[d, s_, j] = g.flow_vph.to_numpy()
            pv2, pq2, n = adjust(pv, pq, o["speed"], o["flow"], tgt, beta, alpha, mr)
            f.loc[m, "speed_kmh"] = pv2[d, s_, j]; f.loc[m, "flow_vph"] = pq2[d, s_, j]; tot += n
        f.to_csv(CACHE / f"state_{split}_{tag_out}.csv", index=False)
        print(split, "adjusted cells", tot, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "eval":
        feb_eval()
    elif sys.argv[1] == "v3eval":
        v3eval()
    elif sys.argv[1] == "apply":
        apply(sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6]))
