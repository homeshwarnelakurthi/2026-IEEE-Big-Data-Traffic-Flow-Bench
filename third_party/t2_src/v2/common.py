# Source: yeashusemwal, "TrafficFlowBench Trees + U-Net + CNN" (Kaggle notebook, 2026),
# kaggle.com/code/yeashusemwal/trafficflowbench-trees-u-net-cnn-lb-0-87. Redistributed for reproduction with credit; see NOTICE.
"""Shared loaders: panels, network, day grids (links x 288) and weekday/time profiles."""
from __future__ import annotations
import datetime, json, glob, os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ["TFB_DATA"]) if "TFB_DATA" in os.environ else ROOT / "data" / "kaggle_public"
T = 288

def panels() -> list[str]:
    m = json.loads((DATA / "config" / "corridors.json").read_text())
    return [p["corridor_id"] for p in m["panels"]]

def families() -> dict[str, str]:
    m = json.loads((DATA / "config" / "corridors.json").read_text())
    return {p["corridor_id"]: p["family_id"] for p in m["panels"]}

def network(panel: str) -> pd.DataFrame:
    """Links in topology order with FD params (fd_parameters authoritative)."""
    nd = DATA / "corridors" / panel / "network"
    links = pd.read_csv(nd / "links.csv"); links["link_id"] = links.link_id.astype(str)
    fd = pd.read_csv(nd / "fd_parameters.csv"); fd["link_id"] = fd.link_id.astype(str)
    fd = fd.groupby("link_id", as_index=False)[["free_speed_kmh", "capacity_vph", "lanes", "k_jam"]].mean()
    net = links[["link_id", "length_km"]].merge(fd, on="link_id", how="left")
    topo = pd.read_csv(nd / "lwr_mainline_topology.csv", dtype=str)
    order = dict(zip(topo.mainline_link_id, topo.order_index.astype(int)))
    net["order_index"] = net.link_id.map(order)
    net = net.sort_values("order_index").drop_duplicates("link_id").reset_index(drop=True)
    net["lanes"] = net.lanes.fillna(4.0).clip(lower=1)
    net["free_speed_kmh"] = net.free_speed_kmh.fillna(105.0)
    net["capacity_vph"] = net.capacity_vph.fillna(1800 * net.lanes)
    net["v_cut"] = 0.6 * net.free_speed_kmh
    net["ramp_on"] = net.link_id.map(dict(zip(topo.mainline_link_id, topo.on_ramp_link_ids.fillna("").str.len() > 0))).fillna(False).astype(int)
    net["ramp_off"] = net.link_id.map(dict(zip(topo.mainline_link_id, topo.off_ramp_link_ids.fillna("").str.len() > 0))).fillna(False).astype(int)
    return net

def day_files(panel: str, split: str, masked: bool) -> list[Path]:
    sub = "mainline_states_masked" if masked else "mainline_states"
    return sorted(Path(p) for p in glob.glob(str(DATA / "corridors" / panel / split / sub / "**" / "*.parquet"), recursive=True))

def date_of(path: Path) -> str:
    s = path.stem.split("_")[-3:]
    return "-".join(s)

def regime_of(path: Path) -> str | None:
    for part in path.parts:
        if part.startswith("mask_regime="):
            return part.split("=")[1]
    return None

def weekday_of(date: str) -> int:
    y, m, d = map(int, date.split("-"))
    return datetime.date(y, m, d).weekday()

def slots_of(ts: pd.Series) -> np.ndarray:
    s = ts.astype(str).to_numpy()
    h = np.fromiter((int(x[11:13]) for x in s), dtype=np.int64, count=len(s))
    mi = np.fromiter((int(x[14:16]) for x in s), dtype=np.int64, count=len(s))
    return h * 12 + mi // 5

def day_grid(path: Path, link_index: dict[str, int], M: int, with_occ: bool = False):
    """Return dict of (M,T) float32 grids: v, q, (occ), elig (bool). NaN where value missing."""
    cols = ["timestamp", "link_id", "speed_kmh", "flow_vph", "pct_observed", "is_score_eligible"] + (["occupancy"] if with_occ else [])
    f = pd.read_parquet(path, columns=cols)
    li = f.link_id.astype(str).map(link_index).to_numpy()
    ok = ~pd.isna(li)
    li = li[ok].astype(int); t = slots_of(f.timestamp[ok])
    out = {}
    for name, col in (("v", "speed_kmh"), ("q", "flow_vph")) + ((("occ", "occupancy"),) if with_occ else ()):
        g = np.full((M, T), np.nan, dtype=np.float32); g[li, t] = f[col].to_numpy(dtype=np.float32)[ok]; out[name] = g
    e = np.zeros((M, T), dtype=bool); e[li, t] = f.is_score_eligible.to_numpy()[ok].astype(bool); out["elig"] = e
    p = np.zeros((M, T), dtype=np.float32); p[li, t] = f.pct_observed.to_numpy()[ok]; out["pct"] = p
    return out

def build_profile(panel: str, net: pd.DataFrame, cache_dir: Path, step: int = 1, exclude: set | None = None, tag: str = "") -> dict[str, np.ndarray]:
    """Weekday x slot median-ish (mean) profiles of speed and flow per link from unmasked train.
    exclude: train dates left out (for leak-free evaluation); tag distinguishes the cache file."""
    cache = cache_dir / f"{panel}_profile_v2{tag}.npz"
    if cache.exists():
        z = np.load(cache); return {"v": z["v"], "q": z["q"], "v_std": z["v_std"], "qfrac": z["qfrac"]}
    M = len(net); li = {l: i for i, l in enumerate(net.link_id)}
    sv = np.zeros((7, M, T)); sq = np.zeros((7, M, T)); sv2 = np.zeros((7, M, T)); n = np.zeros((7, M, T)); nq = np.zeros((7, M, T))
    vcut = net.v_cut.to_numpy()[:, None]
    for p in day_files(panel, "train", masked=False)[::step]:
        if exclude and date_of(p) in exclude: continue
        g = day_grid(p, li, M); w = weekday_of(date_of(p))
        ok = np.isfinite(g["v"]) & np.isfinite(g["q"])
        v = np.where(ok, g["v"], 0); q = np.where(ok, g["q"], 0)
        sv[w] += v; sq[w] += q; sv2[w] += v * v; n[w] += ok; nq[w] += ok & (g["v"] <= vcut)
    nn = np.maximum(n, 1)
    v = sv / nn; q = sq / nn; vstd = np.sqrt(np.maximum(sv2 / nn - v * v, 0)); qfrac = nq / nn
    # fill empty slots with link mean
    for w in range(7):
        for m in range(M):
            e = n[w, m] == 0
            if e.any():
                v[w, m, e] = v[w, m, ~e].mean() if (~e).any() else 100.0
                q[w, m, e] = q[w, m, ~e].mean() if (~e).any() else 3000.0
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, v=v.astype(np.float32), q=q.astype(np.float32), v_std=vstd.astype(np.float32), qfrac=qfrac.astype(np.float32))
    return {"v": v.astype(np.float32), "q": q.astype(np.float32), "v_std": vstd.astype(np.float32), "qfrac": qfrac.astype(np.float32)}
