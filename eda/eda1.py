"""Quick EDA over the TrafficFlowBench release: schemas, noise, congestion, Task 2/4 structure."""
import glob, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

REL = Path(sys.argv[1])
pd.set_option("display.width", 220, "display.max_columns", 40)
panels = [p.name for p in sorted((REL / "corridors").iterdir())]
print("panels", panels)

P = "D12_I5_N"
net = REL / "corridors" / P / "network"
for f in ["links.csv", "fd_parameters.csv", "lwr_mainline_topology.csv", "ramp_attachment_map.csv", "path_set.csv", "path_link_incidence.csv"]:
    d = pd.read_csv(net / f)
    print(f"\n== {f} {d.shape}\n", d.head(3).to_string())

f0 = sorted(glob.glob(str(REL / "corridors" / P / "train/mainline_states/*/*.parquet")))[0]
d = pd.read_parquet(f0)
print("\n== mainline", d.shape, "\n", d.dtypes, "\n", d.head(3).to_string())
r0 = sorted(glob.glob(str(REL / "corridors" / P / "train/ramp_states/**/*.parquet"), recursive=True))[0]
print("\n== ramp", pd.read_parquet(r0).head(3).to_string())

# per-panel summary over a sample of train months
rows = []
for p in panels:
    fd = pd.read_csv(REL / "corridors" / p / "network/fd_parameters.csv")
    fs = sorted(glob.glob(str(REL / "corridors" / p / "train/mainline_states/*/*.parquet")))
    x = pd.concat([pd.read_parquet(f, columns=["timestamp", "link_id", "station_id", "speed_kmh", "flow_vph", "pct_observed", "is_score_eligible"]) for f in fs[::7]])
    x["link_id"] = x.link_id.astype(str); fd["link_id"] = fd.link_id.astype(str)
    x = x.merge(fd[["link_id", "free_speed_kmh", "lanes"]], on="link_id", how="left")
    e = x[x.is_score_eligible.astype(bool)]
    free = e[e.flow_vph / e.lanes < 800]
    q = e.speed_kmh <= 0.6 * e.free_speed_kmh
    rows.append(dict(panel=p, links=x.link_id.nunique(), stations=x.station_id.nunique(), lanes=fd.lanes.mean().round(1),
                     elig=e.shape[0] / x.shape[0], queued_frac=q.mean(),
                     free_speed_resid_sd=(free.speed_kmh - free.free_speed_kmh).std(),
                     speed_sd=e.speed_kmh.std(), flow_lane_sd=(e.flow_vph / e.lanes).std()))
print("\n== panel summary (train, every 7th day)\n", pd.DataFrame(rows).round(3).to_string())

# Task 2 structure
wi = pd.read_csv(REL / "task2" / P / "validation/window_index.csv")
print("\n== window_index\n", wi.head(10).to_string())
wh = pd.read_parquet(REL / "task2" / P / "validation/window_history.parquet")
print("\n== window_history", wh.shape, "\n", wh.head(3).to_string())
wt = pd.read_csv(REL / "task2" / P / "train/window_index.csv")
print("train windows", wt.shape, wt.iloc[:, :6].head(3).to_string())

# Task 4 structure
for s in ["train", "validation", "private"]:
    c = pd.read_csv(REL / "task4" / P / s / "synthetic_link_counts.csv")
    b = pd.read_csv(REL / "task4" / P / s / "synthetic_weak_prior.csv")
    print(f"\n== task4 {s} counts {c.shape} prior {b.shape} prior_sum {b.path_flow.sum():.0f}\n", c.head(2).to_string(), "\n", b.head(2).to_string())
ps = pd.read_csv(net / "path_set.csv"); inc = pd.read_csv(net / "path_link_incidence.csv")
print("paths", len(ps), "OD pairs", ps.groupby(["origin_zone", "destination_zone"]).ngroups, "links in incidence", inc.link_id.nunique(),
      "paths per OD", ps.groupby(["origin_zone", "destination_zone"]).size().describe().to_dict())
