"""Task 4: L2 prior-regularised ODME at several lambdas; writes cache/t4_l2_lam{lam}_{split}.csv."""
import sys

import numpy as np
import pandas as pd

import t4_variants as tv
from common import CACHE, RELEASE, panels

lams = [float(x) for x in sys.argv[1:]] or [0.005, 0.5]
out = {}
for panel in panels():
    ps, A, links, counts, priors = tv.load(panel)
    for s in ("validation", "private"):
        c = counts[s].reindex(links).to_numpy(float); b = priors[s]
        tpl = pd.read_csv(RELEASE / "task4" / panel / s / "sample_submission_path_flow.csv", dtype=str)
        ref = None
        line = []
        for lam in [0.05] + lams:
            tv.LAM = lam
            f = tv.solve_l2(A, c, b)
            ref = f if lam == 0.05 else ref
            line.append(f"lam={lam}: S_link={1-np.abs(A@f-c).sum()/c.sum():.5f} L1to005={np.abs(f-ref).sum()/ref.sum():.4f} dev={np.abs(f-b).sum()/b.sum():.3f}")
            if lam != 0.05:
                t = tpl.copy(); t["path_flow"] = t.path_id.map(dict(zip(ps.path_id, f))).astype(float).clip(lower=0)
                out.setdefault((lam, s), []).append(t)
        print(f"[{s}] {panel} " + " | ".join(line), flush=True)
for (lam, s), parts in out.items():
    pd.concat(parts, ignore_index=True).to_csv(CACHE / f"t4_l2_lam{lam}_{s}.csv", index=False)
