# Source: anubisyy, "CatBoost-LGBM-NNLS pipeline" (Kaggle notebook, 2026), kaggle.com/code/anubisyy/catboost-lgbm-nnls-pipeline-top-50-0-86517.
# Redistributed for reproduction with credit; see NOTICE. Changes by Homeshwar Nelakurthi: state_inference.py replaced by the
# notebook's tfb_source version (fixed 0.5/0.5 blend, no state_settings.json).
from pathlib import Path
import os
RELEASE = Path(os.environ["TFB_DATA_ROOT"])
OUT = Path(os.environ.get("TFB_OUTPUT_ROOT", "/kaggle/working"))
OUT.mkdir(parents=True, exist_ok=True)
THREADS = int(os.environ.get("TFB_THREADS", "4"))
