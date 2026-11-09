"""Task 4 ODME variants on each split's own counts.

l2      : min ||Af-c||^2 + lam||f-b||^2, f>=0 (the public solution; reproduced as a check)
ipf     : min KL(f||b) s.t. Af = c (multiplicative fitting from the split's own prior)
ipf_geo : same, from the geometric mean of the three months' priors rescaled to this split's total
ipf_grav: same, from a log-linear origin/destination/length model fitted to all three priors
Writes cache/t4_{variant}_{split}.csv in template format and prints diagnostics vs the public file."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from common import CACHE, RELEASE, panels

SPLITS = ("train", "validation", "private")
LAM = 0.05


def load(panel):
    net = RELEASE / "corridors" / panel / "network"
    ps = pd.read_csv(net / "path_set.csv", dtype=str)
    inc = pd.read_csv(net / "path_link_incidence.csv", dtype=str)
    counts = {s: pd.read_csv(RELEASE / "task4" / panel / s / "synthetic_link_counts.csv", dtype={"link_id": str})
              .set_index("link_id")["count"] for s in SPLITS}
    priors = {s: pd.read_csv(RELEASE / "task4" / panel / s / "synthetic_weak_prior.csv", dtype={"path_id": str})
              .set_index("path_id").path_flow.reindex(ps.path_id).to_numpy(float) for s in SPLITS}
    links = list(counts["validation"].index)
    lid = {l: i for i, l in enumerate(links)}; pid = {p: i for i, p in enumerate(ps.path_id)}
    inc = inc[inc.link_id.isin(lid)]
    A = np.zeros((len(links), len(ps)))
    A[inc.link_id.map(lid).to_numpy(), inc.path_id.map(pid).to_numpy()] = 1.0
    return ps, A, links, counts, priors


def solve_l2(A, c, b):
    def fg(f):
        r = A @ f - c; d = f - b
        return 0.5 * (r @ r) + 0.5 * LAM * (d @ d), A.T @ r + LAM * d
    x0 = np.clip(b + A.T @ np.linalg.solve(A @ A.T + LAM * np.eye(len(c)), c - A @ b), 0, None)
    r = minimize(fg, x0, jac=True, method="L-BFGS-B", bounds=[(0, None)] * len(b),
                 options=dict(maxiter=20000, ftol=1e-15, gtol=1e-8, maxcor=30))
    return r.x


def solve_ipf(A, c, b, epochs=2000, tol=1e-9):
    rows = [np.nonzero(A[l])[0] for l in range(A.shape[0])]
    f = b.copy()
    for _ in range(epochs):
        dev = 0.0
        for l, idx in enumerate(rows):
            s = f[idx].sum()
            if len(idx) == 0 or s <= 0 or c[l] <= 0:
                continue
            r = c[l] / s; f[idx] *= r; dev = max(dev, abs(r - 1))
        if dev < tol:
            break
    return f


def gravity(ps, priors):
    o = pd.get_dummies(ps.origin_zone).to_numpy(float); d = pd.get_dummies(ps.destination_zone).to_numpy(float)
    n = (ps.link_seq.str.count(";") + 1).to_numpy(float); ln = np.log(n)
    X = np.column_stack([o, d, ln, ln ** 2, n])
    Y = np.mean([np.log(np.maximum(priors[s], 1e-3)) for s in SPLITS], 0)
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return np.exp(X @ beta)


def main():
    pub = pd.concat([pd.read_csv(RELEASE.parent.parent / "public_outputs" / "unet" / "task4" / f"task4_{s}.csv", dtype={"path_id": str})
                     for s in ("validation", "private")])
    pub_map = {(r.panel, r.departure_time, r.path_id): r.path_flow for r in pub.itertuples()}
    out = {}
    for panel in panels():
        ps, A, links, counts, priors = load(panel)
        geo = np.exp(np.mean([np.log(np.maximum(priors[s], 1e-3)) for s in SPLITS], 0))
        grav = gravity(ps, priors)
        for s in ("validation", "private"):
            c = counts[s].reindex(links).to_numpy(float); b = priors[s]
            tpl = pd.read_csv(RELEASE / "task4" / panel / s / "sample_submission_path_flow.csv", dtype=str)
            pubf = np.array([pub_map[(panel, tpl.departure_time.iloc[0], p)] for p in ps.path_id])
            sols = {"l2": solve_l2(A, c, b),
                    "ipf": solve_ipf(A, c, b),
                    "ipf_geo": solve_ipf(A, c, geo * b.sum() / geo.sum()),
                    "ipf_grav": solve_ipf(A, c, grav * b.sum() / grav.sum())}
            line = []
            for k, f in sols.items():
                sl = 1 - np.abs(A @ f - c).sum() / c.sum()
                dpub = np.abs(f - pubf).sum() / pubf.sum()
                dev = np.abs(f - b).sum() / b.sum()
                line.append(f"{k}: S_link={sl:.4f} L1toPub={dpub:.3f} dev/prior={dev:.3f} sum={f.sum():.0f}")
                t = tpl.copy(); t["path_flow"] = t.path_id.map(dict(zip(ps.path_id, f))).astype(float).clip(lower=0)
                out.setdefault((k, s), []).append(t)
            print(f"[{s}] {panel} prior_sum={b.sum():.0f} | " + " | ".join(line), flush=True)
    for (k, s), parts in out.items():
        pd.concat(parts, ignore_index=True).to_csv(CACHE / f"t4_{k}_{s}.csv", index=False)


if __name__ == "__main__":
    main()
