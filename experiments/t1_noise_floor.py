"""Estimate white measurement noise per panel from second differences in time: var(x_t - (x_{t-1}+x_{t+1})/2) = 1.5 sigma^2 + signal."""
import numpy as np
from common import load, meta, panels
from t1_common import family_mean
per = {}
for p in panels():
    z = load(p, "train", masked=False); lanes = meta(p).lanes.to_numpy(np.float32)
    v, q = z["speed"][::5], z["flow"][::5] / lanes
    def sig(a):
        r = a[:, 1:-1] - 0.5 * (a[:, :-2] + a[:, 2:])
        return np.sqrt(np.nanmedian(r ** 2) / 1.5 / 0.455)  # median of chi2_1 = 0.455 -> robust variance
    sv, sq = sig(v), sig(q)
    S = 0.54 * (1 - sv / 25) + 0.46 * (1 - sq / 600)
    per[p] = S
    print(p, "sigma_v", round(float(sv), 2), "sigma_q/lane", round(float(sq), 1), "S_state if RMSE=noise", round(float(S), 4))
print("family-mean ceiling (perfect noise-free prediction):", round(family_mean(per), 4))
