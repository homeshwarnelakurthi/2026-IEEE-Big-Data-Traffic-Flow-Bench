# Source: anubisyy, "CatBoost-LGBM-NNLS pipeline" (Kaggle notebook, 2026), kaggle.com/code/anubisyy/catboost-lgbm-nnls-pipeline-top-50-0-86517.
# Redistributed for reproduction with credit; see NOTICE. Changes by Homeshwar Nelakurthi: state_inference.py replaced by the
# notebook's tfb_source version (fixed 0.5/0.5 blend, no state_settings.json).
"""Train the two-stage onset model using June–January and validate on February."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from config import RELEASE, OUT, THREADS
from queue_features import Panel, panel_names, HISTORY, HORIZON, candidate_origins
Xtrain=[];ytrain=[];valid=[]
for pi,name in enumerate(panel_names()):
    p=Panel(name,pi)
    for file in sorted((RELEASE/'corridors'/name/'train/mainline_states').glob('**/*.parquet')):
        times,v,q,e=p.train_day(file)
        cand=candidate_origins(p,times,v,e)['queue_onset'];cs=set(cand)
        for t in [a for a in cand if a-1 not in cs]:
            x=p.features(v[t-HISTORY:t],q[t-HISTORY:t],e[t-HISTORY:t],pd.Timestamp(times[t]),'queue_onset')[-len(p.links):]
            y=(v[t+HORIZON]<=p.cut)&np.isfinite(v[t+HORIZON]); eligible=e[t+HORIZON]&np.isfinite(v[t+HORIZON])
            if file.parent.name.endswith('2031-02'):
                valid.append((name,x,y,eligible))
            else:
                Xtrain.append(x[eligible]);ytrain.append(y[eligible])
    print(name,'train rows',sum(len(a) for a in ytrain),'valid windows',len(valid),flush=True)
X=np.concatenate(Xtrain);y=np.concatenate(ytrain)
Xv=np.concatenate([a[1][a[3]] for a in valid]);yv=np.concatenate([a[2][a[3]] for a in valid])
params={'objective':'binary','metric':'binary_logloss','learning_rate':.04,'num_leaves':15,'min_data_in_leaf':100,'lambda_l2':10,'feature_fraction':.9,'num_threads':THREADS,'verbosity':-1,'seed':20261004}
model=lgb.train(params,lgb.Dataset(X,label=y),num_boost_round=900,valid_sets=[lgb.Dataset(Xv,label=yv)],callbacks=[lgb.early_stopping(70),lgb.log_evaluation(100)])
model.save_model(str(OUT/'onset_stage1.txt'))

model = lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=1500,
                  init_model=str(OUT/'onset_stage1.txt'),
                  valid_sets=[lgb.Dataset(Xv, label=yv)],
                  callbacks=[lgb.early_stopping(70), lgb.log_evaluation(100)])
model.save_model(str(OUT/'onset_lgb.txt'))
print("Selected onset threshold: 0.50; predict the final horizon step with a one-link fallback.")
