"""Fast, single-pass seen-OOS reproducibility rebuild; no training or selection."""
from __future__ import annotations

import hashlib
import json
import sys
import time
import tomllib
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from pyarrow import feather

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cb_image_alpha.models.paper_cnn import PaperCNN

ROOT=Path(__file__).resolve().parents[1]
CFG=ROOT/'configs/paper/p4_reval_v1.toml'; OUT=ROOT/'artifacts/paper-v1/p4-reval-20260919-r1'
ZIP=ROOT/'artifacts/paper-v1/acquire-20260917T010443459965Z/recovery-1/img_data.zip'; INV=ROOT/'artifacts/paper-v1/p4-oos-inventory.json'
def sha(p): return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
def auc(y,p):
 pos=p[y==1]; neg=p[y==0]
 if len(pos)==0 or len(neg)==0:return None
 order=np.argsort(p,kind='mergesort'); ranks=np.empty(len(p)); ranks[order]=np.arange(1,len(p)+1); return float((ranks[y==1].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg)))
def metrics(y,p):
 pred=(p>=.5).astype(np.int64); cm=np.bincount(y*2+pred,minlength=4); tn,fp,fn,tp=cm
 return {'rows':len(y),'cross_entropy':float(-np.mean(y*np.log(np.clip(p,1e-7,1-1e-7))+(1-y)*np.log(np.clip(1-p,1e-7,1-1e-7)))),'roc_auc':auc(y,p),'accuracy':float((pred==y).mean()),'balanced_accuracy':float(((tp/(tp+fn)) if tp+fn else 0)+((tn/(tn+fp)) if tn+fp else 0))/2,'brier':float(np.mean((p-y)**2))}
def main():
 cfg=tomllib.loads(CFG.read_text()); inv=json.loads(INV.read_text())
 if cfg['status']!='IN_REVIEW' or sha(INV)!=cfg['oos_inventory_sha256'] or inv['data_member_count']!=38: raise RuntimeError('reval input gate')
 if not torch.cuda.is_available(): raise RuntimeError('CUDA required for performance rebuild')
 torch.set_num_threads(cfg['resources']['cpu_threads']); torch.cuda.set_per_process_memory_fraction(cfg['resources']['cuda_reserved_limit_bytes']/torch.cuda.get_device_properties(0).total_memory)
 OUT.mkdir(exist_ok=False); start=time.monotonic(); z=zipfile.ZipFile(ZIP); ids_parts=[]; y_parts=[]; preds=[[] for _ in range(5)]; exclusions=[]
 models=[]; norms=[]
 for m in range(5):
  model=PaperCNN().cuda().eval(); state=torch.load(ROOT/f'artifacts/paper-v1/p3-paper-training-v1/seal/member-{m}/state.pt',map_location='cuda',weights_only=False); model.load_state_dict(state['model']); models.append(model); norms.append(json.loads((ROOT/f'artifacts/paper-v1/p3-paper-training-v1/seal/member-{m}/normalization.json').read_text()))
 for year in range(2001,2020):
  ip=f'monthly_20d/20d_month_has_vb_[20]_ma_{year}_images.dat'; lp=f'monthly_20d/20d_month_has_vb_[20]_ma_{year}_labels_w_delay.feather'; raw=z.read(ip); tab=feather.read_table(BytesIO(z.read(lp))); n=tab.num_rows
  if len(raw)!=n*3840: raise RuntimeError('image row mismatch')
  ret=np.asarray(tab['Ret_month'].to_pylist(),dtype=np.float64); ew=np.asarray(tab['EWMA_vol'].to_pylist(),dtype=np.float64); keep=np.isfinite(ret)&(ret!=-99)&np.isfinite(ew)&(ew>0)
  arr=np.frombuffer(raw,dtype=np.uint8).reshape(n,64,60)[keep]
  if not np.all((arr==0)|(arr==255)) or np.any(np.count_nonzero(arr,axis=(1,2))==0): raise RuntimeError('invalid image')
  rows=np.flatnonzero(keep); ids_parts.append(year*1_000_000_000+rows); y_parts.append((ret[keep]>0).astype(np.int64)); exclusions.append({'year':year,'raw':n,'kept':int(keep.sum()),'excluded':int((~keep).sum())})
  for start_row in range(0,len(arr),cfg['performance']['batch_size']):
   raw_batch=torch.from_numpy(arr[start_row:start_row+cfg['performance']['batch_size']].copy()).cuda(non_blocking=False).unsqueeze(1)
   for m,(model,norm) in enumerate(zip(models,norms,strict=True)):
    x=raw_batch.float().div_(255.).sub_(float(norm['mean'])).div_(float(norm['std']))
    with torch.inference_mode(): preds[m].append(model(x).softmax(1)[:,1].cpu().numpy())
 del z,models
 ids=np.concatenate(ids_parts); y=np.concatenate(y_parts); prob=[np.concatenate(x) for x in preds]
 for m,p in enumerate(prob):
  if len(p)!=len(ids) or not np.isfinite(p).all(): raise RuntimeError('prediction integrity')
  np.savez_compressed(OUT/f'member-{m}-predictions.npz',sample_id=ids,label=y,probability=p)
 ensemble=np.mean(np.stack(prob),axis=0); result={'status':'COMPLETED_AWAITING_AUDIT','run_id':cfg['run_id'],'read_class':'already_seen_OOS_reassessment','rows':len(y),'common_rows':len(y),'exclusions':exclusions,'member_metrics':[metrics(y,p) for p in prob],'ensemble_metrics':metrics(y,ensemble),'elapsed_seconds':time.monotonic()-start,'protocol_sha256':sha(CFG),'inventory_sha256':sha(INV),'endpoint_verified':False,'strict_leakage_free_claim':False,'test_qualification':'none','old_results_guidance':False,'oos_reread_for_reassessment':True,'no_training':True,'no_tuning':True,'no_backtest':True}
 (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
if __name__=='__main__':main()