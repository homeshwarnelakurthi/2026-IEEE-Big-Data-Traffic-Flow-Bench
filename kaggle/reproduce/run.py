import glob, os, shutil, subprocess, sys
g = lambda pat: glob.glob("/kaggle/input/**/" + pat, recursive=True)
DATA = g("kaggle_public")[0]
CODE = os.path.dirname(os.path.dirname(g("tfb/run_all.py")[0]))
print("DATA", DATA, "CODE", CODE, flush=True)
SOL = os.path.dirname(os.path.dirname(os.path.dirname(g("models/nn/v2_seed1.pt")[0])))
RUN = "/kaggle/working/run"
env = dict(os.environ, TFB_DATA_ROOT=DATA, TFB_SOLUTION=SOL, TFB_CODE=CODE, TFB_OUT=RUN, TFB_CACHE=RUN + "/cache")
# full run with the defaults (all 16 NN seeds, W_V3 0.65, W_NN 0.8) -> run/submission.csv = final sub30_wnn80
r = subprocess.run([sys.executable, CODE + "/tfb/run_all.py"], env=env)
for p in ("nn", "cat_out", "pub_out", "t2_src", "pub_src", "cat_src", "cache"):
    shutil.rmtree(RUN + "/" + p, ignore_errors=True)
for f in glob.glob(RUN + "/t1_*.csv"):
    os.remove(f)
sys.exit(r.returncode)
