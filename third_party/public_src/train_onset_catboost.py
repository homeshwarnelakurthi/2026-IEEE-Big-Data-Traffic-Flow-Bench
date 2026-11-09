# Source: anubisyy, "CatBoost-LGBM-NNLS pipeline" (Kaggle notebook, 2026), kaggle.com/code/anubisyy/catboost-lgbm-nnls-pipeline-top-50-0-86517.
# Redistributed for reproduction with credit; see NOTICE. Changes by Homeshwar Nelakurthi: state_inference.py replaced by the
# notebook's tfb_source version (fixed 0.5/0.5 blend, no state_settings.json).
"""GPU onset classifier with causal history-dropout augmentation."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
import os
from config import RELEASE, OUT, THREADS
from queue_features import Panel,panel_names,HISTORY,HORIZON,candidate_origins
xx=[];yy=[];valid=[];rng=np.random.default_rng(20261005)
for pi,name in enumerate(panel_names()):
    p=Panel(name,pi)
    for file in sorted((RELEASE/'corridors'/name/'train/mainline_states').glob('**/*.parquet')):
        times,v,q,e=p.train_day(file);cand=candidate_origins(p,times,v,e)['queue_onset'];cs=set(cand);isval=file.parent.name.endswith('2031-02')
        for t in [a for a in cand if a-1 not in cs]:
            truth=(v[t+1:t+HORIZON+1]<=p.cut)&np.isfinite(v[t+1:t+HORIZON+1]);eligible=e[t+1:t+HORIZON+1]&np.isfinite(v[t+1:t+HORIZON+1])
            for rep in range(1 if isval else 2):
                hv=v[t-HISTORY:t].copy();hq=q[t-HISTORY:t].copy();he=e[t-HISTORY:t].copy()
                if rep:
                    drop=rng.random(hv.shape)<.05;hv[drop]=np.nan;hq[drop]=np.nan;he[drop]=False
                x=p.features(hv,hq,he,pd.Timestamp(times[t]),'queue_onset')[-len(p.links):]
                if isval:valid.append((name,x,truth,eligible))
                else:xx.append(x[eligible[-1]]);yy.append(truth[-1,eligible[-1]])
    print(name,'train rows',sum(len(a) for a in yy),'valid windows',len(valid),flush=True)
X=np.concatenate(xx);y=np.concatenate(yy);Xv=np.concatenate([a[1][a[3][-1]] for a in valid]);yv=np.concatenate([a[2][-1,a[3][-1]] for a in valid])
m=CatBoostClassifier(iterations=3000,depth=7,learning_rate=.05,l2_leaf_reg=10,task_type='GPU',devices=os.environ.get('TFB_GPU_DEVICE', '0'),thread_count=THREADS,random_seed=20261005,loss_function='Logloss',early_stopping_rounds=150,verbose=100,train_dir=str(OUT/'catboost'))
m.fit(X,y,eval_set=(Xv,yv),use_best_model=True);m.save_model(str(OUT/'onset.cbm'))
old=lgb.Booster(model_file=str(OUT/'onset_lgb.txt'));records=[]
for name,x,truth,e in valid:
    p=m.predict_proba(x,thread_count=THREADS)[:,1];b=old.predict(x,num_threads=THREADS)
    for weight in (0,.5,1):
        prob=(1-weight)*b+weight*p
        for threshold in (.3,.35,.4,.45,.5,.55,.6):
            pred=np.zeros_like(truth);pred[-1]=prob>=threshold
            if not pred.any():pred[-1,np.argmax(prob)]=True
            iou=((pred&truth)&e).sum()/max(((pred|truth)&e).sum(),1)
            records.append({'panel':name,'gpu_weight':weight,'threshold':threshold,'iou':iou})
f=pd.DataFrame(records);f.to_csv(OUT/'window_scores.csv',index=False)
s=f.groupby(['gpu_weight','threshold','panel']).iou.mean().groupby(['gpu_weight','threshold']).mean().sort_values(ascending=False);s.to_csv(OUT/'scores.csv');print(s.to_string(),flush=True)
best=s.index[0];(OUT/'onset_settings.json').write_text(json.dumps({'gpu_weight':float(best[0]),'threshold':float(best[1]),'proxy':float(s.iloc[0]),'train_rows':len(y),'valid_windows':len(valid)},indent=2))
