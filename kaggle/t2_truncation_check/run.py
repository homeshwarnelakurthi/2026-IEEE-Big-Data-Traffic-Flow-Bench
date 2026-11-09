# Task 2 award check: rerun the Task 2 stage with every released row stamped after a window's origin T removed; predictions must not change.
# Pass k: on each (panel, split, day) the mainline and ramp files are cut at the k-th origin of that day and days without a window are dropped;
# the k-th window of each day is compared with an untruncated baseline run. Probabilities are compared as well as the 0/1 output.
import glob, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd

g = lambda pat: glob.glob("/kaggle/input/**/" + pat, recursive=True)
DATA = Path(g("kaggle_public")[0])
SRC = os.path.dirname(os.path.dirname(g("tfb/run_all.py")[0]))
CODE = "/kaggle/working/code"; shutil.copytree(SRC, CODE)
SOL = os.path.dirname(os.path.dirname(os.path.dirname(g("models/nn/v2_seed1.pt")[0])))
RUN = CODE + "/tfb/run_all.py"; s = open(RUN).read()
rep = [
    ("                L = len(onp.links); pred = np.zeros((HORIZON, L), int)\n",
     "                L = len(onp.links); pred = np.zeros((HORIZON, L), int); pa, pb = (np.full((HORIZON, L), np.nan) for _ in range(2))\n"),
    ("                    pred[-1] = sel\n", "                    pred[-1] = sel; pa[-1] = prob\n"),
    ("                    pred = (0.5 * pm + 0.5 * p_pub >= 0.45).astype(int)\n",
     "                    pred = (0.5 * pm + 0.5 * p_pub >= 0.45).astype(int); pa, pb = pm, p_pub\n"),
    ('                                         "queue_pred": pred.ravel()}))\n',
     '                                         "queue_pred": pred.ravel(), "cond": w.condition, "p_a": pa.ravel(), "p_b": pb.ravel()}))\n'),
]
for a, b in rep:
    assert s.count(a) == 1, a
    s = s.replace(a, b)
open(RUN, "w").write(s)

wi = pd.concat([pd.read_csv(f) for f in glob.glob(str(DATA / "task2" / "*" / "*" / "window_index.csv"))], ignore_index=True)
wi = wi[wi.split.isin(["validation", "private"])].reset_index(drop=True)
wi["k"] = wi.sort_values("forecast_origin").groupby(["panel", "split", "date"]).cumcount()
PASSES = int(wi.k.max()) + 1
print("windows", len(wi), "passes", PASSES, flush=True)


def date_of(f):
    return "-".join(Path(f).stem.split("_")[-3:])


def build(k, root):
    root.mkdir(parents=True)
    for x in DATA.iterdir():
        if x.name != "corridors":
            os.symlink(x, root / x.name)
    kept = removed = 0
    for pdir in [d for d in (DATA / "corridors").iterdir() if d.is_dir()]:
        for x in pdir.iterdir():
            dst = root / "corridors" / pdir.name / x.name; dst.parent.mkdir(parents=True, exist_ok=True)
            if x.name not in ("validation", "private"):
                os.symlink(x, dst); continue
            cut = wi[(wi.panel == pdir.name) & (wi.split == x.name)]
            for f in x.rglob("*.parquet"):
                day = cut[cut.date == date_of(f)].sort_values("forecast_origin")
                if day.empty:
                    continue
                T = day.forecast_origin.iloc[min(k, len(day) - 1)]
                d = pd.read_parquet(f); n = len(d); d = d[d.timestamp.astype(str) <= T]
                assert d.timestamp.astype(str).max() <= T
                out = dst / f.relative_to(x); out.parent.mkdir(parents=True, exist_ok=True); d.to_parquet(out, index=False)
                kept += len(d); removed += n - len(d)
    print(f"pass {k}: kept {kept:,} rows, removed {removed:,} on window days; non-window days dropped", flush=True)


def run(data, out):
    env = dict(os.environ, TFB_DATA_ROOT=str(data), TFB_SOLUTION=SOL, TFB_CODE=CODE, TFB_OUT=out, TFB_CACHE=out + "/cache")
    subprocess.run([sys.executable, RUN, "--stages", "t2"], env=env, check=True)
    t = pd.read_csv(out + "/task2.csv", dtype={"link_id": str})
    for sub in ("pub_out", "t2_src", "pub_src", "cache"):
        shutil.rmtree(out + "/" + sub, ignore_errors=True)
    return t


KEY = ["window_id", "timestamp", "link_id"]
base = run(DATA, "/kaggle/working/base").set_index(KEY)
rows, final = [], []
for k in range(PASSES):
    root = Path(f"/kaggle/working/trunc{k}"); build(k, root)
    t = run(root, f"/kaggle/working/p{k}"); shutil.rmtree(root)
    ids = wi[wi.k == k].window_id
    t = t[t.window_id.isin(ids)]; final.append(t); t = t.set_index(KEY)
    b = base.loc[t.index]
    for w, gt in t.groupby(level=0):
        gb = b.loc[gt.index]
        rows.append(dict(window_id=w, k=k, cond=gt.cond.iloc[0], cells=len(gt), diff_01=int((gt.queue_pred != gb.queue_pred).sum()),
                         dp_a=float(np.nanmax(np.abs(gt.p_a - gb.p_a).fillna(0))), dp_b=float(np.nanmax(np.abs(gt.p_b - gb.p_b).fillna(0)))))
    shutil.rmtree(f"/kaggle/working/p{k}", ignore_errors=True)
r = pd.DataFrame(rows); r.to_csv("/kaggle/working/trunc_compare.csv", index=False)
pd.concat(final).to_csv("/kaggle/working/task2_truncated.csv", index=False)
base.reset_index().to_csv("/kaggle/working/task2_base.csv", index=False)
print("WINDOWS", len(r), "of", len(wi), "| windows with any 0/1 change", int((r.diff_01 > 0).sum()), "| cells changed", int(r.diff_01.sum()), "of", int(r.cells.sum()), flush=True)
print("MAX prob change members/onset", r.dp_a.max(), "public", r.dp_b.max(), flush=True)
print(r.sort_values("diff_01", ascending=False).head(10).to_string(), flush=True)
shutil.rmtree("/kaggle/working/base", ignore_errors=True); shutil.rmtree(CODE, ignore_errors=True)
