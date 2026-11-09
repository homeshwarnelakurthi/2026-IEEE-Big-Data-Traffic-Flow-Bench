"""Compare null patterns of ineligible cells: train unmasked vs validation masked vs window_history."""
import sys, glob
from pathlib import Path
import pandas as pd
REL = Path(sys.argv[1]); P = "D7_I405_N"
t = pd.read_parquet(sorted(glob.glob(str(REL/"corridors"/P/"train/mainline_states/*/*.parquet")))[100])
tm = pd.read_parquet(sorted(glob.glob(str(REL/"corridors"/P/"train/mainline_states_masked/*/*.parquet")))[100])
v = pd.read_parquet(sorted(glob.glob(str(REL/"corridors"/P/"validation/mainline_states_masked/*/*.parquet")))[3])
for name, d in [("train", t), ("train_masked", tm), ("val_masked", v)]:
    e = d.is_score_eligible.astype(bool)
    print(name, len(d), "null|elig", d.speed_kmh[e].isna().mean().round(3), "null|inelig", d.speed_kmh[~e].isna().mean().round(3),
          "pct_obs dist", d.pct_observed.describe()[["min","25%","50%"]].to_dict())
wh = pd.read_parquet(REL/"task2"/P/"validation/window_history.parquet")
e = wh.is_score_eligible.astype(bool); print("window_history null|elig", wh.speed_kmh[e].isna().mean().round(3), "null|inelig", wh.speed_kmh[~e].isna().mean().round(3))
print(sorted(glob.glob(str(REL/"corridors"/P/"train/mainline_states_masked/*")))[:5])
