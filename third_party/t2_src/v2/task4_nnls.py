# Source: yeashusemwal, "TrafficFlowBench Trees + U-Net + CNN" (Kaggle notebook, 2026),
# kaggle.com/code/yeashusemwal/trafficflowbench-trees-u-net-cnn-lb-0-87. Redistributed for reproduction with credit; see NOTICE.
"""Task 4 variant: prior-regularised NNLS (organizer objective) with a denoised (pooled gravity-fit) prior and the split's own counts."""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from task4_kl import load, gravity_fit, pooled_prior, SPLITS, DATA, ROOT, panels

def nnls_fista(A, c, b, lam, iters=3000):
    AtA = A.T @ A; Atc = A.T @ c
    L = np.linalg.eigvalsh(AtA)[-1] + lam; eta = 1.0 / L
    x = np.maximum(b, 0); y = x.copy(); t = 1.0
    for _ in range(iters):
        g = AtA @ y - Atc + lam * (y - b); xn = np.maximum(0, y - eta * g)
        tn = 0.5 * (1 + np.sqrt(1 + 4 * t * t)); y = xn + (t - 1) / tn * (xn - x); x, t = xn, tn
    return x

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=ROOT / "submissions" / "task4_gnnls")
    ap.add_argument("--lam", type=float, default=0.05); ap.add_argument("--prior", default="gravity", choices=["gravity", "geo3", "single"])
    ap.add_argument("--counts", default="own", choices=["own", "train"])
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    outs = {s: [] for s in ("validation", "private")}
    for panel in panels():
        ps, A, links, counts, priors = load(panel)
        if a.prior == "gravity":
            bf, rv = gravity_fit(ps, priors); b = bf * np.exp(rv / 2)  # back to arithmetic-mean scale
        elif a.prior == "geo3": b = pooled_prior(priors, SPLITS) * np.exp(0.63 ** 2 / 2)
        for s in outs:
            bs = priors[s] if a.prior == "single" else b
            cs = counts["train" if a.counts == "train" else s]
            use = np.isin(links, cs.index); c = cs.reindex(np.array(links)[use]).to_numpy(float)
            f = nnls_fista(A[use], c, bs, a.lam)
            sl = 1 - np.abs(A[use] @ f - c).sum() / c.sum()
            tpl = pd.read_csv(DATA / "task4" / panel / s / "sample_submission_path_flow.csv", dtype=str)
            tpl["path_flow"] = tpl.path_id.map(dict(zip(ps.path_id, f))).astype(float).clip(lower=0); outs[s].append(tpl)
            print(f"[{s}] {panel}: S_link={sl:.4f} zeros={(f<=1e-9).sum()} sum={f.sum():.0f} dev/|b|={np.abs(f-bs).sum()/bs.sum():.3f}", flush=True)
    for s in outs:
        d = pd.concat(outs[s], ignore_index=True); d.to_csv(a.out / f"task4_{s}.csv", index=False); print("wrote", s, len(d))

if __name__ == "__main__":
    main()
