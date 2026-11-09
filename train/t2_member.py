"""Retrain one Task 2 LightGBM member of the yeashusemwal notebook from its published source in third_party/t2_src/v2.

  member  script          seed        member  script          seed
  v11     task2_v11.py    11          v15     task2_v15.py    15
  v12     task2_v11.py    12          v16     task2_v15.py    16
  v13     task2_v11.py    13          v20     task2_v18.py    20
  v14     task2_v11.py    14
Usage (Kaggle CPU, competition data + tfb-code attached, about 45 min): python train/t2_member.py v11
Output: /kaggle/working/t2_<member>/ -> copy to tfb-solution models/t2_members/t2_<member>/.
"""
import glob, os, shutil, subprocess, sys

MEMBERS = {"v11": ("task2_v11.py", 11), "v12": ("task2_v11.py", 12), "v13": ("task2_v11.py", 13), "v14": ("task2_v11.py", 14),
           "v15": ("task2_v15.py", 15), "v16": ("task2_v15.py", 16), "v20": ("task2_v18.py", 20)}
member = sys.argv[1]; script, seed = MEMBERS[member]
DATA = os.environ.get("TFB_DATA_ROOT") or glob.glob("/kaggle/input/**/kaggle_public", recursive=True)[0]
SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "t2_src", "v2")
OUT = os.environ.get("TFB_OUT", "/kaggle/working")
WORK = os.path.join(OUT, "t2_src_work"); shutil.copytree(SRC, WORK, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
env = dict(os.environ, TFB_DATA=DATA, OMP_NUM_THREADS="4")
r = subprocess.run([sys.executable, script, "--out", os.path.join(OUT, f"t2_{member}"), "--threads", "4", "--seed", str(seed)], cwd=WORK, env=env)
shutil.rmtree(WORK, ignore_errors=True)
sys.exit(r.returncode)
