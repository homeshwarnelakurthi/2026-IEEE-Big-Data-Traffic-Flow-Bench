# Source: yeashusemwal, "TrafficFlowBench Trees + U-Net + CNN" (Kaggle notebook, 2026),
# kaggle.com/code/yeashusemwal/trafficflowbench-trees-u-net-cnn-lb-0-87. Redistributed for reproduction with credit; see NOTICE.
"""Task 4: pooled-prior KL projection (iterative proportional fitting) onto exact link counts."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, ROOT, panels

SPLITS = ["train", "validation", "private"]

def load(panel):
    nd = DATA / "corridors" / panel / "network"
    ps = pd.read_csv(nd / "path_set.csv", dtype=str); pid = {p: i for i, p in enumerate(ps.path_id)}
    inc = pd.read_csv(nd / "path_link_incidence.csv", dtype=str)
    counts = {s: pd.read_csv(DATA / "task4" / panel / s / "synthetic_link_counts.csv", dtype={"link_id": str}).set_index("link_id")["count"] for s in SPLITS}
    priors = {s: pd.read_csv(DATA / "task4" / panel / s / "synthetic_weak_prior.csv", dtype={"path_id": str}).set_index("path_id").path_flow.reindex(ps.path_id).to_numpy(float) for s in SPLITS}
    links = sorted(set.union(*[set(c.index) for c in counts.values()])); lid = {l: i for i, l in enumerate(links)}
    inc = inc[inc.link_id.isin(lid) & inc.path_id.isin(pid)]
    A = np.zeros((len(links), len(ps))); A[inc.link_id.map(lid).to_numpy(), inc.path_id.map(pid).to_numpy()] = 1.0
    return ps, A, links, counts, priors

def ipf(A, c, b, epochs=300, tol=1e-7):
    """min KL(f||b) s.t. A f = c, f>=0 via cyclic multiplicative row scaling (consistent system)."""
    rows = [np.nonzero(A[l])[0] for l in range(A.shape[0])]
    f = b.copy()
    for ep in range(epochs):
        maxdev = 0.0
        for l, idx in enumerate(rows):
            if len(idx) == 0 or c[l] <= 0: continue
            s = f[idx].sum()
            if s <= 0: continue
            r = c[l] / s; f[idx] *= r; maxdev = max(maxdev, abs(r - 1))
        if maxdev < tol: break
    return f

def pooled_prior(priors, splits, mode="geo"):
    L = np.stack([np.log(np.maximum(priors[s], 1e-3)) for s in splits])
    return np.exp(L.mean(0)) if mode == "geo" else np.stack([priors[s] for s in splits]).mean(0)

def s_od(f, ft): return max(0, 1 - np.abs(f - ft).sum() / ft.sum())

def gravity_fit(ps, priors, splits=SPLITS):
    """OLS of stacked log priors on origin/dest dummies and path-length terms; returns fitted flow and resid var."""
    o = pd.get_dummies(ps.origin_zone).to_numpy(float); d = pd.get_dummies(ps.destination_zone).to_numpy(float)
    n = (ps.link_seq.str.count(";") + 1).to_numpy(float); ln = np.log(n)
    X = np.column_stack([o, d, ln, ln ** 2, n, np.sqrt(n)])
    Y = np.stack([np.log(np.maximum(priors[s], 1e-3)) for s in splits])
    beta, *_ = np.linalg.lstsq(X, Y.mean(0), rcond=None); fit = X @ beta
    rv = float(np.mean([(Y[i] - fit).var() for i in range(len(splits))]))
    return np.exp(fit), rv

def synthetic_check(A, ps, priors, rng, sigma=0.63, scale=2.0):
    """Treat a KL solution as pseudo-truth, regenerate priors with the inferred noise, compare methods."""
    b0 = pooled_prior(priors, SPLITS); ft = ipf(A, A @ (b0 / scale), b0 / scale)  # pseudo truth with realistic structure
    c = A @ ft
    pri = {s: scale * ft * np.exp(rng.normal(0, sigma, len(ft))) for s in SPLITS}
    res = {}
    for name, b in (("single", pri["validation"]), ("geo3", pooled_prior(pri, SPLITS)), ("arith3", pooled_prior(pri, SPLITS, "arith"))):
        for sc in ("gamma", "none"):
            g = c.sum() / (A @ b).sum() if sc == "gamma" else 1.0
            f = ipf(A, c, b * g); res[f"{name}/{sc}"] = round(s_od(f, ft), 4)
        res[f"{name}/prior_only_scaled"] = round(s_od(b * c.sum() / (A @ b).sum(), ft), 4)
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "submissions" / "task4_v2")
    ap.add_argument("--mode", default="gravity", choices=["gravity", "geo3", "single"])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    outs = {s: [] for s in ("validation", "private")}
    for panel in panels():
        ps, A, links, counts, priors = load(panel)
        if a.check: print(panel, synthetic_check(A, ps, priors, rng), flush=True)
        b = pooled_prior(priors, SPLITS) if a.mode == "geo3" else None
        if a.mode == "gravity":
            b, rv = gravity_fit(ps, priors); print(f"{panel}: gravity resid var per split {rv:.3f} (pure noise ~0.40)", flush=True)
        for s in ("validation", "private"):
            bs = b if b is not None else priors[s]
            c = counts[s].reindex(links).fillna(0).to_numpy(float); use = np.isin(links, counts[s].index)
            g = c[use].sum() / (A[use] @ bs).sum()
            f = ipf(A[use], c[use], bs * g)
            sl = 1 - np.abs(A[use] @ f - c[use]).sum() / c[use].sum(); adj = np.abs(f - bs * g).sum() / f.sum()
            tpl = pd.read_csv(DATA / "task4" / panel / s / "sample_submission_path_flow.csv", dtype=str)
            tpl["path_flow"] = tpl.path_id.map(dict(zip(ps.path_id, f))).astype(float).clip(lower=0)
            outs[s].append(tpl); print(f"[{s}] {panel}: S_link={sl:.5f} gamma={g:.3f} sum f={f.sum():.0f} min f={f.min():.3g} ipf_adj={adj:.3f}", flush=True)
    for s in outs:
        d = pd.concat(outs[s], ignore_index=True); d.to_csv(a.out / f"task4_{s}.csv", index=False); print("wrote", s, len(d))

if __name__ == "__main__":
    main()
