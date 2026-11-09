"""Task 4 structure: which links carry counts, how the prior relates to counts, zone margins."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
REL = Path(sys.argv[1])
for P in ["D12_I5_N", "D7_I10_E", "D12_I405_S"]:
    net = REL/"corridors"/P/"network"
    ps = pd.read_csv(net/"path_set.csv"); inc = pd.read_csv(net/"path_link_incidence.csv")
    links = inc.link_id.drop_duplicates().tolist(); li = {l:i for i,l in enumerate(links)}; pi = {p:i for i,p in enumerate(ps.path_id)}
    A = np.zeros((len(links), len(ps))); A[inc.link_id.map(li), inc.path_id.map(pi)] = 1
    pref = pd.Series(links).str.extract(r"^([A-Z]+)")[0]
    print(f"\n#### {P} paths={len(ps)} links={len(links)} zones o={ps.origin_zone.nunique()} d={ps.destination_zone.nunique()} link prefixes={pref.value_counts().to_dict()}")
    for s in ["train","validation","private"]:
        c = pd.read_csv(REL/"task4"/P/s/"synthetic_link_counts.csv"); b = pd.read_csv(REL/"task4"/P/s/"synthetic_weak_prior.csv").set_index("path_id").path_flow.reindex(ps.path_id).to_numpy()
        cc = c.set_index("link_id")["count"]
        m = np.array([l in cc.index for l in links]); cv = cc.reindex(np.array(links)[m]).to_numpy()
        Ab = A[m] @ b
        cpref = pd.Series(np.array(links)[m]).str.extract(r"^([A-Z]+)")[0].value_counts().to_dict()
        print(f" {s}: counted links {m.sum()} {cpref}  sum c={cv.sum():.0f} sum Ab={Ab.sum():.0f} ratio med={np.median(cv/np.maximum(Ab,1)):.3f} rel L1(Ab,c)={np.abs(Ab-cv).sum()/cv.sum():.3f} prior zeros={(b==0).mean():.3f} prior cv={b.std()/b.mean():.2f}")
    # same-path prior agreement across splits
    bs = {s: pd.read_csv(REL/"task4"/P/s/"synthetic_weak_prior.csv").set_index("path_id").path_flow.reindex(ps.path_id).to_numpy() for s in ["train","validation","private"]}
    print("  prior corr train/val/priv:", np.corrcoef(np.log1p(bs["train"]), np.log1p(bs["validation"]))[0,1].round(3), np.corrcoef(np.log1p(bs["validation"]), np.log1p(bs["private"]))[0,1].round(3),
          " L1(val,priv)/sum=", (np.abs(bs["validation"]-bs["private"]).sum()/bs["validation"].sum()).round(3))
