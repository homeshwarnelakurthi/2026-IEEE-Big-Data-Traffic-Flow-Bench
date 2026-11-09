# Source: anubisyy, "CatBoost-LGBM-NNLS pipeline" (Kaggle notebook, 2026), kaggle.com/code/anubisyy/catboost-lgbm-nnls-pipeline-top-50-0-86517.
# Redistributed for reproduction with credit; see NOTICE. Changes by Homeshwar Nelakurthi: state_inference.py replaced by the
# notebook's tfb_source version (fixed 0.5/0.5 blend, no state_settings.json).
from pathlib import Path
import argparse,json,time
import numpy as np
from catboost import CatBoostRegressor
from config import OUT, THREADS
ap=argparse.ArgumentParser();ap.add_argument('--channel',choices=['speed','flow'],required=True);ap.add_argument('--device',required=True);ap.add_argument('--iterations',type=int,default=9000);ap.add_argument('--depth',type=int,default=8);ap.add_argument('--tag',default='d8');a=ap.parse_args()
t=time.time();key='yv' if a.channel=='speed' else 'yq'
tr=np.load(OUT/'train.npz');va=np.load(OUT/'valid.npz')
model=CatBoostRegressor(iterations=a.iterations,depth=a.depth,learning_rate=.06,l2_leaf_reg=10,loss_function='RMSE',eval_metric='RMSE',task_type='GPU',devices=a.device,gpu_ram_part=.8,thread_count=THREADS,random_seed=20261005,bootstrap_type='Bernoulli',subsample=.85,early_stopping_rounds=150,verbose=100,train_dir=str(OUT/('catboost_'+a.channel+'_'+a.tag)))
model.fit(tr['X'],tr[key],eval_set=(va['X'],va[key]),use_best_model=True)
model.save_model(str(OUT/(a.channel+'_'+a.tag+'.cbm')))
pred=model.predict(va['X'],thread_count=THREADS);np.save(OUT/('valid_'+a.channel+'_'+a.tag+'.npy'),pred)
metrics={'best_iteration':model.get_best_iteration(),'rmse':float(np.sqrt(np.mean((va[key]-pred)**2))),'elapsed_seconds':time.time()-t,'train_rows':len(tr[key]),'valid_rows':len(va[key]),'device':a.device}
(OUT/(a.channel+'_'+a.tag+'_metrics.json')).write_text(json.dumps(metrics,indent=2));print(metrics,flush=True)
