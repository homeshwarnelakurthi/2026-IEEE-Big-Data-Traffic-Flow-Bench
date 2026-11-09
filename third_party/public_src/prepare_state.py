# Source: anubisyy, "CatBoost-LGBM-NNLS pipeline" (Kaggle notebook, 2026), kaggle.com/code/anubisyy/catboost-lgbm-nnls-pipeline-top-50-0-86517.
# Redistributed for reproduction with credit; see NOTICE. Changes by Homeshwar Nelakurthi: state_inference.py replaced by the
# notebook's tfb_source version (fixed 0.5/0.5 blend, no state_settings.json).
"""State features with lag/lead context and two independently synthesized masks."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from config import OUT, THREADS
import state_model as base

def features(panel,pi,times,v,q,e,rows,cols,regime):
    X,bv,bq=base.compute_features(panel,pi,times,v,q,rows,cols,regime)
    extra=[];lanes=np.maximum(panel.lanes[cols],1)
    for offset in (-12,-6,-3,-2,-1,1,2,3,6,12):
        rr=rows+offset;valid=(rr>=0)&(rr<len(times));rr=np.clip(rr,0,len(times)-1)
        vv=v[rr,cols];qq=q[rr,cols]/lanes
        extra.extend([np.where(valid,vv,np.nan),np.where(valid,qq,np.nan),
                      np.where(valid,qq/np.maximum(vv,1),np.nan),
                      np.where(valid&e[rr,cols]&np.isfinite(vv),1,0)])
    return np.column_stack([X,*extra]).astype(np.float32),bv,bq

def main():
    xx=[];yv=[];yq=[];val={k:[] for k in ('X','yv','yq','bv','bq','panel','regime')};audit=[]
    original_seed=base.SEED
    for pi,name in enumerate(base.all_panels()):
        p=base.Panel(name,pi)
        for month in (*base.TRAIN_MONTHS,base.VALID_MONTH):
            times,v,q,e=base.matrices(base.month_frame(name,month),p)
            isval=month==base.VALID_MONTH
            audit.append({'panel':name,'month':month,'cells':v.size,'speed_missing':float(np.isnan(v).mean()),'eligible':float(e.mean()),'finite_ineligible':float((np.isfinite(v)&~e).mean()),'negative_flow':int((q<0).sum())})
            for rep in range(1 if isval else 2):
                base.SEED=original_seed+rep*7919
                mask,regime=base.simulated_mask(name,times,v,q,e)
                rows,cols=base.sampled_targets(mask,times,1500 if isval else 750)
                X,bv,bq=features(p,pi,times,np.where(mask,np.nan,v),np.where(mask,np.nan,q),e,rows,cols,regime)
                rv=v[rows,cols]-bv;rq=q[rows,cols]/p.lanes[cols]-bq
                if isval:
                    for k,a in [('X',X),('yv',rv),('yq',rq),('bv',bv),('bq',bq),('panel',np.full(len(rows),pi)),('regime',regime[rows])]:val[k].append(a)
                else:xx.append(X);yv.append(rv);yq.append(rq)
            print(name,month,'done',flush=True)
    base.SEED=original_seed
    np.savez(OUT/'train.npz',X=np.concatenate(xx),yv=np.concatenate(yv),yq=np.concatenate(yq))
    np.savez(OUT/'valid.npz',**{k:np.concatenate(a) for k,a in val.items()})
    pd.DataFrame(audit).to_csv(OUT/'data_audit.csv',index=False)
    print('train',sum(len(a) for a in yv),'valid',sum(len(a) for a in val['yv']),'features',xx[0].shape[1],flush=True)
if __name__=='__main__':main()
