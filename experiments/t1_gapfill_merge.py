"""run12 Task 1: the run11 per-cell recipe, with the dedicated gap-filler replacing CatBoost on target cells inside
Task 2 blank blocks. Reads cache/state_{split}_cellfix.csv and the gap-filler output; writes cache/state_{split}_gapfill.csv."""
import sys

import pandas as pd

from common import CACHE

src = sys.argv[1] if len(sys.argv) > 1 else str(CACHE)  # folder holding t1_gapfill_{split}.csv
key = ["panel", "timestamp", "link_id"]
for split in ("validation", "private"):
    base = pd.read_csv(CACHE / f"state_{split}_cellfix.csv", dtype={"station_id": str, "link_id": str})
    gf = pd.read_csv(f"{src}/t1_gapfill_{split}.csv", dtype={"link_id": str}).rename(columns={"speed_kmh": "v_gf", "flow_vph": "q_gf"})
    m = base.merge(gf, on=key, how="left")
    hit = m.v_gf.notna()
    m.loc[hit, "speed_kmh"] = m.v_gf[hit]; m.loc[hit, "flow_vph"] = m.q_gf[hit]
    m.drop(columns=["v_gf", "q_gf"]).to_csv(CACHE / f"state_{split}_gapfill.csv", index=False)
    print(split, f"target cells replaced by the gap-filler: {hit.sum()} ({hit.mean():.2%})")
