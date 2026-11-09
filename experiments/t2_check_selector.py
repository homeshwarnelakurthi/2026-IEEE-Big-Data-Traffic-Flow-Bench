"""Check that the generated candidates contain the official train windows."""
import numpy as np
import pandas as pd

from common import RELEASE, TASK2_EXCLUDED, load, panels, slot_of
from t2_data import PanelT2

hit = {"queue_onset": [0, 0], "queue_ongoing": [0, 0]}
ncand = {"queue_onset": 0, "queue_ongoing": 0}
for i, p in enumerate(x for x in panels() if x not in TASK2_EXCLUDED):
    P = PanelT2(p, i)
    z = load(p, "train", masked=False)
    S = P.orient(z["speed"]); dates = list(z["dates"])
    wi = pd.read_csv(RELEASE / "task2" / p / "train" / "window_index.csv")
    for w in wi.itertuples():
        d = dates.index(w.date); t = slot_of(w.forecast_origin)
        c = P.candidates(S[d])
        hit[w.condition][0] += t in c[w.condition]; hit[w.condition][1] += 1
        q = P.queued(S[d])
        if w.condition == "queue_onset" and t not in c[w.condition]:
            print(" miss", p, w.date, t, "hist active", q[t-12:t+1].any(1).astype(int), "fut", q[t+1:t+7].any(1).astype(int))
    for d in range(30):
        c = P.candidates(S[d]); ncand["queue_onset"] += len(c["queue_onset"]); ncand["queue_ongoing"] += len(c["queue_ongoing"])
print("official windows recovered:", hit, "candidates in June:", ncand)
