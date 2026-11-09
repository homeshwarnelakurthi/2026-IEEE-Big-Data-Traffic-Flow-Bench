"""Checks: ramp<->zone mapping for ODME margins, T-row availability for Task 2, train truth queue stats."""
import sys, glob
from pathlib import Path
import numpy as np, pandas as pd
REL = Path(sys.argv[1]); P = "D12_I5_N"; net = REL/"corridors"/P/"network"
ram = pd.read_csv(net/"ramp_attachment_map.csv"); print("ramps by type", ram.ramp_type.value_counts().to_dict())
inc = pd.read_csv(net/"path_link_incidence.csv"); ps = pd.read_csv(net/"path_set.csv")
ron = sorted(inc.link_id[inc.link_id.str.startswith("RON")].unique())[:5]; print("RON sample", ron)
# first/last mainline link of each origin/destination zone
ps["first"] = ps.link_seq.str.split(";").str[0]; ps["last"] = ps.link_seq.str.split(";").str[-1]
oz = ps.groupby("origin_zone")["first"].agg(lambda s: s.mode()[0]); dz = ps.groupby("destination_zone")["last"].agg(lambda s: s.mode()[0])
onr = ram[ram.ramp_type=="OR"].groupby("nearest_mainline_link_id").size(); offr = ram[ram.ramp_type=="FR"].groupby("nearest_mainline_link_id").size()
print("origin zones whose first link has an OR ramp:", oz.isin(onr.index).mean().round(3), " dest zones whose last link has FR:", dz.isin(offr.index).mean().round(3))
# Task 2 T-row availability in validation masked file
wi = pd.read_csv(REL/"task2"/P/"validation/window_index.csv")
for _, w in wi.head(4).iterrows():
    f = glob.glob(str(REL/"corridors"/P/f"validation/mainline_states_masked/*/synthetic_mainline_{w.date.replace('-','_')}.parquet"))[0]
    d = pd.read_parquet(f, columns=["timestamp","speed_kmh","is_score_eligible"])
    at = d[d.timestamp==w.forecast_origin]; hz = d[(d.timestamp>w.forecast_origin)&(d.timestamp<=w.forecast_end)]
    print(w.window_id, w.condition, "T-row non-null frac", at.speed_kmh.notna().mean().round(2), "horizon non-null", hz.speed_kmh.notna().mean().round(2))
# train: queue episode stats on truth
fd = pd.read_csv(net/"fd_parameters.csv")
fs = sorted(glob.glob(str(REL/"corridors"/P/"train/mainline_states/*/*.parquet")))
x = pd.concat([pd.read_parquet(f, columns=["date","timestamp","link_id","speed_kmh","flow_vph"]) for f in fs])
x = x.merge(fd[["link_id","free_speed_kmh"]], on="link_id"); x["q"] = x.speed_kmh <= .6*x.free_speed_kmh
x["month"] = x.date.str[:7]; x["hour"] = x.timestamp.str[11:13].astype(int)
top = x.groupby("link_id").q.mean().nlargest(6).index
print("queued frac by month for top bottleneck links\n", x[x.link_id.isin(top)].pivot_table(index="link_id", columns="month", values="q", aggfunc="mean").round(2).to_string())
print("queued frac by hour (all links)\n", x.groupby("hour").q.mean().round(3).to_dict())
print("weekday queued frac", x.assign(wd=pd.to_datetime(x.date).dt.dayofweek).groupby("wd").q.mean().round(3).to_dict())
