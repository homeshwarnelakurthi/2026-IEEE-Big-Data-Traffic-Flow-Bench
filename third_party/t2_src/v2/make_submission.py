# Source: yeashusemwal, "TrafficFlowBench Trees + U-Net + CNN" (Kaggle notebook, 2026),
# kaggle.com/code/yeashusemwal/trafficflowbench-trees-u-net-cnn-lb-0-87. Redistributed for reproduction with credit; see NOTICE.
"""Merge per-task files (validation+private) into the Kaggle upload file, fast path via numpy joins."""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, ROOT

def key_cache() -> pd.DataFrame:
    cache = ROOT / "submissions" / "submission_key.parquet"
    if cache.exists(): return pd.read_parquet(cache)
    k = pd.read_csv(DATA / "submission_key.csv", dtype=str)
    k["submission_id"] = k.submission_id.astype(np.int64); k.to_parquet(cache); return k

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task1", type=Path, required=True, help="dir with task1_validation.csv / task1_private.csv")
    ap.add_argument("--task2", type=Path, required=True); ap.add_argument("--task4", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args(); t0 = time.time()
    k = key_cache(); print("key", len(k), f"{time.time()-t0:.0f}s", flush=True)
    out = pd.DataFrame({"submission_id": k.submission_id, "task": k.task, "speed_kmh": 0.0, "flow_vph": 0.0, "queue_pred": 0, "path_flow": 0.0})
    # state
    st = pd.concat([pd.read_csv(a.task1 / f"task1_{s}.csv", dtype={"panel": str, "timestamp": str, "station_id": str, "link_id": str, "mask_regime": str}) for s in ("validation", "private")], ignore_index=True)
    key_s = st.panel + "|" + st.timestamp + "|" + st.station_id + "|" + st.link_id + "|" + st.mask_regime
    m = k.task == "state"; kk = k.panel[m] + "|" + k.timestamp[m] + "|" + k.station_id[m] + "|" + k.link_id[m] + "|" + k.mask_regime[m]
    idx = pd.Index(key_s); pos = idx.get_indexer(kk); print("state unmatched", int((pos < 0).sum()), "dups", int(idx.duplicated().sum()), flush=True)
    out.loc[m, "speed_kmh"] = st.speed_kmh.to_numpy()[pos]; out.loc[m, "flow_vph"] = st.flow_vph.to_numpy()[pos]
    # queue
    qu = pd.concat([pd.read_csv(a.task2 / f"task2_{s}.csv", dtype=str) for s in ("validation", "private")], ignore_index=True)
    idx = pd.Index(qu.window_id + "|" + qu.timestamp + "|" + qu.link_id); m = k.task == "queue"
    pos = idx.get_indexer(k.window_id[m] + "|" + k.timestamp[m] + "|" + k.link_id[m]); print("queue unmatched", int((pos < 0).sum()), flush=True)
    out.loc[m, "queue_pred"] = qu.queue_pred.astype(float).round().astype(int).to_numpy()[np.maximum(pos, 0)] * (pos >= 0)
    # odme
    od = pd.concat([pd.read_csv(a.task4 / f"task4_{s}.csv", dtype=str) for s in ("validation", "private")], ignore_index=True)
    idx = pd.Index(od.panel + "|" + od.departure_time + "|" + od.path_id); m = k.task == "odme"
    pos = idx.get_indexer(k.panel[m] + "|" + k.departure_time[m] + "|" + k.path_id[m]); print("odme unmatched", int((pos < 0).sum()), flush=True)
    out.loc[m, "path_flow"] = od.path_flow.astype(float).to_numpy()[np.maximum(pos, 0)] * (pos >= 0)
    assert out.isna().sum().sum() == 0 and np.isfinite(out[["speed_kmh", "flow_vph", "path_flow"]].to_numpy()).all()
    print("state speed/flow stats", out.loc[out.task == "state", ["speed_kmh", "flow_vph"]].describe().loc[["min", "mean", "max"]].to_dict())
    print("queue mean", out.loc[out.task == "queue", "queue_pred"].mean(), "odme min", out.loc[out.task == "odme", "path_flow"].min())
    a.output.parent.mkdir(parents=True, exist_ok=True); out.to_csv(a.output, index=False, float_format="%.4f"); print("wrote", a.output, len(out), f"{time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
