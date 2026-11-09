# LOCAL ANALYSIS ONLY: not used by run_all.py. Reads intermediate outputs of our Kaggle runs under $TFB_WORKSPACE (not in this repo).
"""Rebuild run14's Task 1 with another gap-filler inside the blocks: 0.5 x NN v3 + 0.5 x gap-filler, other cells unchanged.
Usage: python t1_gf_swap.py <gapfill_dir> <out_dir> <tag> [--check] [--v3extra <dir>] [--w <v3 weight, default 0.5>]
  --check: only report the match with cache/state_*_v3gf.csv
  --v3extra: folder with more v3 seeds' state_{split}_seed*.csv; the v3 part becomes the mean over seeds 5-6 and these
"""
import sys

import numpy as np
import pandas as pd

from common import CACHE

WS = __import__("os").environ.get("TFB_WORKSPACE", ".")  # folder holding our Kaggle run outputs

src, out, tag = sys.argv[1:4]
check = "--check" in sys.argv
KEY = ["panel", "timestamp", "link_id"]
for split in ("validation", "private"):
    dt = {"station_id": str, "link_id": str}
    base = pd.read_csv(CACHE / f"state_{split}_v3gf.csv", dtype=dt)
    v3 = pd.read_csv(CACHE / f"state_{split}_v3.csv", dtype=dt, usecols=KEY + ["speed_kmh", "flow_vph"]).rename(columns={"speed_kmh": "v3", "flow_vph": "q3"})
    g = pd.read_csv(f"{src}/t1_gapfill_{split}.csv", dtype={"link_id": str}).rename(columns={"speed_kmh": "vg", "flow_vph": "qg"})
    m = base.merge(v3, on=KEY, how="left", validate="one_to_one").merge(g, on=KEY, how="left", validate="one_to_one")
    assert len(m) == len(base) and m.v3.notna().all()
    if "--v3extra" in sys.argv:
        import glob
        extra = sorted(glob.glob(f"{sys.argv[sys.argv.index('--v3extra') + 1]}/state_{split}_seed*.csv"))
        sv, sq = 2 * m.v3.to_numpy(), 2 * m.q3.to_numpy()
        for f in extra:
            e = m[KEY].merge(pd.read_csv(f, dtype=dt, usecols=KEY + ["speed_kmh", "flow_vph"]), on=KEY, how="left", validate="one_to_one")
            ok = m.vg.notna().to_numpy()
            assert e.speed_kmh[ok].notna().all(), f
            sv, sq = sv + e.speed_kmh.to_numpy(), sq + e.flow_vph.to_numpy()
        m["v3"], m["q3"] = sv / (2 + len(extra)), sq / (2 + len(extra))
        print(split, "v3 seeds", 2 + len(extra), [f.rsplit("_", 1)[-1] for f in extra], flush=True)
    hit = m.vg.notna().to_numpy()
    w = float(sys.argv[sys.argv.index("--w") + 1]) if "--w" in sys.argv else 0.5  # weight on NN v3
    nv, nq = (1 - w) * m.vg + w * m.v3, (1 - w) * m.qg + w * m.q3
    print(split, "in-block cells", int(hit.sum()), "of", len(m), "| max |new - base| speed", float(np.abs(nv - m.speed_kmh)[hit].max()),
          "| mean |new - base| speed", float(np.abs(nv - m.speed_kmh)[hit].mean()), "flow", float(np.abs(nq - m.flow_vph)[hit].mean()), flush=True)
    if not check:
        m.loc[hit, "speed_kmh"] = nv[hit]; m.loc[hit, "flow_vph"] = nq[hit]
        m[base.columns].to_csv(f"{out}/state_{split}_{tag}.csv", index=False)
