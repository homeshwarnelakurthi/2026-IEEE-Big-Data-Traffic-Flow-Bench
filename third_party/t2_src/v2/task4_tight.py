# Source: yeashusemwal, "TrafficFlowBench Trees + U-Net + CNN" (Kaggle notebook, 2026),
# kaggle.com/code/yeashusemwal/trafficflowbench-trees-u-net-cnn-lb-0-87. Redistributed for reproduction with credit; see NOTICE.
"""Task 4: the same prior-regularised NNLS as task4_nnls.py (--prior single --counts own, lambda 0.05),
solved to convergence: FISTA warm start, then L-BFGS-B with tight tolerances (FISTA alone stops at 3000 iterations)."""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ.setdefault(v, "2")
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
sys.path.insert(0, str(Path(__file__).resolve().parent))
from task4_kl import load, DATA, ROOT, panels
from task4_nnls import nnls_fista

def solve(A, c, b, lam):
    def obj(f):
        r = A @ f - c; d = f - b; return 0.5 * (r @ r + lam * d @ d), A.T @ r + lam * d
    f0 = nnls_fista(A, c, b, lam)
    fit = minimize(obj, f0, jac=True, method="L-BFGS-B", bounds=[(0, None)] * len(b), options={"maxiter": 20000, "ftol": 1e-15, "gtol": 1e-8, "maxcor": 30})
    f = np.maximum(fit.x, 0); g = obj(f)[1]; pg = np.max(np.abs(np.where(f > 1e-7, g, np.minimum(g, 0))))
    return f, f0, pg

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=ROOT / "submissions" / "task4_tight"); ap.add_argument("--lam", type=float, default=0.05)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True); outs = {s: [] for s in ("validation", "private")}
    for panel in panels():
        ps, A, links, counts, priors = load(panel)
        for s in outs:
            cs = counts[s]; use = np.isin(links, cs.index); c = cs.reindex(np.array(links)[use]).to_numpy(float); b = priors[s]
            f, f0, pg = solve(A[use], c, b, a.lam)
            tpl = pd.read_csv(DATA / "task4" / panel / s / "sample_submission_path_flow.csv", dtype=str)
            tpl["path_flow"] = tpl.path_id.map(dict(zip(ps.path_id, f))).astype(float).clip(lower=0); outs[s].append(tpl)
            print(f"[{s}] {panel}: rel|f-f_fista|={np.abs(f - f0).sum() / np.abs(f).sum():.4f} proj_grad={pg:.2e} S_link={1 - np.abs(A[use] @ f - c).sum() / c.sum():.5f}", flush=True)
    for s in outs:
        d = pd.concat(outs[s], ignore_index=True); d.to_csv(a.out / f"task4_{s}.csv", index=False); print("wrote", s, len(d))

if __name__ == "__main__":
    main()
