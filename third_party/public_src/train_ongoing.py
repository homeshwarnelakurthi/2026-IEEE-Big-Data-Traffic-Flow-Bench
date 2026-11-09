# Source: anubisyy, "CatBoost-LGBM-NNLS pipeline" (Kaggle notebook, 2026), kaggle.com/code/anubisyy/catboost-lgbm-nnls-pipeline-top-50-0-86517.
# Redistributed for reproduction with credit; see NOTICE. Changes by Homeshwar Nelakurthi: state_inference.py replaced by the
# notebook's tfb_source version (fixed 0.5/0.5 blend, no state_settings.json).
from queue_ongoing import fit_model
from config import OUT
model, settings = fit_model()
model.save_model(str(OUT/'ongoing_lgb.txt'))
