"""Assemble an upload file: start from a base submission and replace task blocks with our predictions.

Usage: python make_submission.py --base BASE.csv --queue-val V.csv --queue-priv P.csv --out OUT.csv
Queue prediction files are in template format (window_id, timestamp, link_id, queue_pred)."""
import argparse

import numpy as np
import pandas as pd

from common import RELEASE

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--queue", nargs="*", default=[])
ap.add_argument("--odme", nargs="*", default=[])
ap.add_argument("--state", nargs="*", default=[], help='weighted groups "w|val.csv|priv.csv"; weights should sum to 1')
ap.add_argument("--out", required=True)
ap.add_argument("--partial", action="store_true", help="replace only the queue rows present in --queue files")
a = ap.parse_args()

base = pd.read_csv(a.base)
key = pd.read_csv(RELEASE / "submission_key.csv", usecols=["submission_id", "task", "window_id", "timestamp", "link_id"], dtype=str)
template = pd.read_csv(RELEASE / "sample_submission.csv", usecols=["submission_id", "task"])
assert len(base) == len(template) and (base.submission_id.values == template.submission_id.values).all()

if a.queue:
    q = pd.concat([pd.read_csv(f, dtype={"link_id": str}) for f in a.queue], ignore_index=True)
    kq = key[key.task == "queue"].merge(q[["window_id", "timestamp", "link_id", "queue_pred"]],
                                        on=["window_id", "timestamp", "link_id"], how="left")
    missing = kq.queue_pred.isna().sum()
    assert a.partial or missing == 0, f"{missing} queue rows have no prediction"
    kq = kq.dropna(subset=["queue_pred"])  # with --partial, rows without a prediction keep the base value
    old = base.set_index("submission_id").queue_pred
    new = kq.set_index(kq.submission_id.astype(int)).queue_pred.astype(int)
    changed = (old.reindex(new.index) != new).mean()
    base = base.set_index("submission_id"); base.loc[new.index, "queue_pred"] = new; base = base.reset_index()
    print(f"queue rows {len(new)}, changed vs base {changed:.3%}, positive share {new.mean():.4f}")

if a.state:
    # --state takes "weight:file_validation:file_private" groups; files are template-format state CSVs
    blend = None
    for spec in a.state:
        w, *files = spec.split("|")
        s = pd.concat([pd.read_csv(f, usecols=["panel", "timestamp", "station_id", "link_id", "mask_regime", "speed_kmh", "flow_vph"],
                                   dtype={"station_id": str, "link_id": str}) for f in files], ignore_index=True)
        s = s.set_index(["panel", "timestamp", "station_id", "link_id", "mask_regime"])[["speed_kmh", "flow_vph"]] * float(w)
        blend = s if blend is None else blend.add(s, fill_value=np.nan)
    ks = pd.read_csv(RELEASE / "submission_key.csv", usecols=["submission_id", "task", "panel", "timestamp", "station_id", "link_id", "mask_regime"], dtype=str)
    ks = ks[ks.task == "state"].join(blend, on=["panel", "timestamp", "station_id", "link_id", "mask_regime"])
    assert ks.speed_kmh.notna().all() and ks.flow_vph.notna().all(), "state rows without a prediction"
    new = ks.set_index(ks.submission_id.astype(int))
    base = base.set_index("submission_id")
    dv = (base.loc[new.index, "speed_kmh"] - new.speed_kmh).abs().mean()
    base.loc[new.index, "speed_kmh"] = new.speed_kmh.clip(1, 180); base.loc[new.index, "flow_vph"] = new.flow_vph.clip(lower=0)
    base = base.reset_index()
    print(f"state rows {len(new)}, mean |speed change| vs base {dv:.3f} km/h")

if a.odme:
    od = pd.concat([pd.read_csv(f, dtype={"path_id": str}) for f in a.odme], ignore_index=True)
    ko = pd.read_csv(RELEASE / "submission_key.csv", usecols=["submission_id", "task", "panel", "departure_time", "path_id"], dtype=str)
    ko = ko[ko.task == "odme"].merge(od[["panel", "departure_time", "path_id", "path_flow"]],
                                     on=["panel", "departure_time", "path_id"], how="left")
    assert ko.path_flow.notna().all(), f"{ko.path_flow.isna().sum()} odme rows have no prediction"
    new = ko.set_index(ko.submission_id.astype(int)).path_flow.astype(float)
    base = base.set_index("submission_id")
    rel = (base.loc[new.index, "path_flow"] - new).abs().sum() / base.loc[new.index, "path_flow"].sum()
    base.loc[new.index, "path_flow"] = new; base = base.reset_index()
    print(f"odme rows {len(new)}, L1 change vs base {rel:.3f}")

# evaluator rules
assert base.notna().all().all(), "blank cells"
assert base.submission_id.is_unique
assert base.queue_pred.isin([0, 1]).all()
assert np.isfinite(base.path_flow).all() and (base.path_flow >= 0).all()
for t in ("state", "queue", "odme"):
    print(t, (base.task == t).sum())
base.to_csv(a.out, index=False)
print("wrote", a.out)
