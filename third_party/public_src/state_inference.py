# Source: anubisyy, "CatBoost-LGBM-NNLS pipeline" (Kaggle notebook, 2026), kaggle.com/code/anubisyy/catboost-lgbm-nnls-pipeline-top-50-0-86517.
# Redistributed for reproduction with credit; see NOTICE. Changes by Homeshwar Nelakurthi: state_inference.py replaced by the
# notebook's tfb_source version (fixed 0.5/0.5 blend, no state_settings.json).
"""State residual ensemble; GPU prediction matches the original execution path."""
from pathlib import Path
import json,os
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from catboost.utils import get_gpu_device_count
from config import OUT,THREADS
from prepare_state import base,features

class StatePredictor:
    def __init__(self, assets):
        self.assets=Path(assets)
        self.settings={channel+'_mix': {'d8': 0.5, 'd10': 0.5} for channel in ('speed','flow')}
        self.task_type=os.environ.get('TFB_PREDICT_DEVICE', 'GPU' if get_gpu_device_count()>0 else 'CPU')
        self.models={}
        for channel in ('speed','flow'):
            for tag in self.settings[channel+'_mix']:
                model=CatBoostRegressor();model.load_model(str(self.assets/(channel+'_'+tag+'.cbm')))
                self.models[channel,tag]=model
        print('State inference device:',self.task_type,flush=True)
    def residual(self,X,channel):
        result=np.zeros(len(X))
        for tag,weight in self.settings[channel+'_mix'].items():
            model=self.models[channel,tag]
            for start in range(0,len(X),250000):
                batch=X[start:start+250000]
                pred=model.predict(batch,task_type=self.task_type,thread_count=THREADS)
                result[start:start+len(batch)]+=weight*pred
        return result

def predict_states(predictor):
    first=True
    for split in ('validation','private'):
        for pi,name in enumerate(base.all_panels()):
            panel=base.Panel(name,pi)
            frame=base.month_frame(name,'',split=split)
            times,v,q,e=base.matrices(frame,panel)
            target=pd.read_csv(base.RELEASE/'task1'/name/split/'sample_submission_state.csv',dtype={'station_id':str,'link_id':str})
            rows=times.get_indexer(target.timestamp);cols=pd.Index(panel.links).get_indexer(target.link_id)
            assert (rows>=0).all() and (cols>=0).all()
            regime=target.mask_regime.map({'R1':0,'R2':1,'R3':2}).to_numpy()
            X,bv,bq=features(panel,pi,times,v,q,e,rows,cols,regime)
            assert X.shape[1]==94
            target['speed_kmh']=np.clip(bv+predictor.residual(X,'speed'),1,180)
            target['flow_vph']=np.clip((bq+predictor.residual(X,'flow'))*panel.lanes[cols],0,150000)
            assert np.isfinite(target[['speed_kmh','flow_vph']].to_numpy()).all()
            columns=['panel','timestamp','station_id','link_id','mask_regime','speed_kmh','flow_vph']
            target[columns].to_csv(OUT/'state.csv',index=False,mode='w' if first else 'a',header=first)
            first=False;print(name,split,len(target),flush=True)
