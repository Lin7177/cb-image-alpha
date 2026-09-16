"""One predeclared G7 diagnostic batch: existing development outputs only."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

from cb_image_alpha.artifacts import read_json, sha256, verify_files, write_json
from cb_image_alpha.data.encoding import decode_fixed_i128
from cb_image_alpha.evaluation.metrics import binary_metrics

PROJECT = Path(__file__).resolve().parents[1]
ORIGINAL = PROJECT / 'artifacts/runs/formal-20260912T083408279881Z'
COUNTS = {'train': 339682, 'validation': 123747}


def normalize_fingerprints(values: dict[str, str]) -> dict[str, str]:
    result = {}
    for name, fingerprint in values.items():
        key = name.replace(chr(92), '/')
        if key in result and result[key] != fingerprint:
            raise ValueError('Conflicting canonical path fingerprint')
        result[key] = fingerprint
    return result


def bins(y: np.ndarray, p: np.ndarray) -> list[dict]:
    assigned = np.minimum((p * 10).astype(int), 9)
    result = []
    for i in range(10):
        mask = assigned == i
        result.append({'lower': i/10, 'upper': (i+1)/10, 'right_closed': i == 9,
                       'count': int(mask.sum()), 'mean_probability': float(p[mask].mean()) if mask.any() else None,
                       'positive_rate': float(y[mask].mean()) if mask.any() else None})
    return result


def quantize(prices: np.ndarray, volumes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ma = np.array([prices[i-19:i+1, 3].mean() for i in range(19, 39)])
    values = np.column_stack((prices[-20:], ma)) / prices[19, 3]
    low, high = values.min(), values.max()
    if high == low or not np.isfinite(values).all() or np.any(volumes <= 0):
        raise ValueError('Invalid selected context')
    normalized = (values-low)/(high-low)
    # Preserve renderer floating-point order for coordinates.
    coords = np.rint((values-low)*(50/(high-low))).astype(int)
    relative_volume = volumes/volumes.max()
    heights = np.rint(volumes*(12/volumes.max())).astype(int)
    return normalized, coords, relative_volume, heights


def plot_lines(path: Path, title: str, labels: list[str], series: dict[str, list[float]],
               low: float, high: float) -> None:
    canvas = Image.new('RGB', (1200, 520), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((35, 12), title, fill='black')
    left, top, width, height = 70, 65, 1080, 340
    for step in range(6):
        value = low + (high-low)*step/5
        yy = top+height*(1-step/5)
        draw.line((left, yy, left+width, yy), fill='#dddddd')
        draw.text((5, yy-5), f'{value:.4f}', fill='black')
    colors = ['#2166ac', '#b2182b', '#228833']
    for k, (name, values) in enumerate(series.items()):
        points = [(left+width*i/max(1,len(labels)-1), top+height*(1-(v-low)/(high-low))) for i,v in enumerate(values)]
        for a,b in itertools.pairwise(points):
            if all(np.isfinite(v) for v in (*a,*b)):
                draw.line([a,b], fill=colors[k], width=3)
        for x,y in points:
            if np.isfinite(y):
                draw.ellipse((x-3,y-3,x+3,y+3), fill=colors[k])
        draw.text((80+k*300, 450), name, fill=colors[k])
    for i,label in enumerate(labels):
        draw.text((left+width*i/max(1,len(labels)-1)-15, 415), label, fill='black')
    canvas.save(path)


def check(out: Path) -> None:
    import psutil
    import torch

    psutil.Process().cpu_affinity(psutil.Process().cpu_affinity()[:4])
    pa.set_cpu_count(4)
    pa.set_io_thread_count(4)
    torch.set_num_threads(4)
    plan = read_json(out/'plan.json')
    verify_files(PROJECT, plan['code_sha256'])
    verify_files(ORIGINAL, plan['input_sha256'])
    frozen = read_json(ORIGINAL/'frozen.json')
    parent = Path(frozen['parent'])
    if sha256(parent/'samples.parquet') != frozen['parent_files']['samples.parquet']:
        raise ValueError('Parent index changed')
    stats = read_json(ORIGINAL/'normalization.json')
    training_rate = stats['positive_rate']
    groups, distributions = [], {}
    selected = None
    for split, count in COUNTS.items():
        columns = ['sample_id','instrument_id','context_start','window_start','window_end','label_end','label','image_row','shard_id','shard_row','p_up']
        table = pq.read_table(ORIGINAL/f'{split}-predictions.parquet', columns=columns, use_threads=False)
        parent_table = pq.read_table(parent/'samples.parquet', columns=['sample_id','image_row','label','window_end','label_end'], filters=[('split','=',split)], use_threads=False).sort_by([('image_row','ascending')])
        if len(table)!=count or not table.select(parent_table.column_names).equals(parent_table):
            raise ValueError('Parent alignment failed')
        ids = table['sample_id'].to_pylist()
        if len(set(ids)) != count:
            raise ValueError('Duplicate ID')
        y, p = table['label'].to_numpy(), table['p_up'].to_numpy()
        binary_metrics(y,p)
        if split == 'train' and float(y.mean()) != training_rate:
            raise ValueError('Training baseline mismatch')
        days = table['window_end'].to_pylist()
        end = date(2023,12,31) if split=='train' else date(2024,12,31)
        if max(days)>end or max(table['label_end'].to_pylist())>end:
            raise ValueError('Date boundary failed')
        keys = {'all':np.array(['all']*count), 'year':np.array([str(d.year) for d in days]),
                'quarter':np.array([f'{d.year}Q{(d.month-1)//3+1}' for d in days])}
        for grain, values in keys.items():
            for key in sorted(set(values)):
                mask = values == key
                yy, pp = y[mask], p[mask]
                report = binary_metrics(yy,pp)
                baseline = binary_metrics(yy,np.full(len(yy),training_rate))
                trading_days = len({days[i] for i in np.flatnonzero(mask)})
                report.update({'split':split,'grain':grain,'period':key,'positive_rate':float(yy.mean()),
                  'trading_days':trading_days,'descriptive_only':trading_days<20 or split=='train',
                  'constant_log_loss':baseline['log_loss'],'loss_difference':report['log_loss']-baseline['log_loss'],
                  'probability_mean':float(pp.mean()), 'probability_std':float(pp.std()),
                  'probability_bins':bins(yy,pp)})
                groups.append(report)
        distributions[split]=bins(y,p)
        if split=='train':
            selected=table.sort_by([('sample_id','ascending')]).slice(0,2048).drop(['p_up','label','label_end'])
            pq.write_table(selected,out/'selected-train.parquet')
            if plan.get('retry_selection_sha256') and sha256(out/'selected-train.parquet') != plan['retry_selection_sha256']:
                raise ValueError('Retry selected samples changed')
    if selected is None:
        raise ValueError('No selected Train rows')
    calendar_path = Path(frozen['source']).parent/'download/base/calendar.csv'
    expected = frozen['auxiliary_sha256'][str(calendar_path)]
    if sha256(calendar_path)!=expected:
        raise ValueError('Calendar changed')
    with calendar_path.open(encoding='utf-8-sig',newline='') as handle:
        calendar=[date.fromisoformat(r['trading_date']) for r in csv.DictReader(handle)]
    positions={day:i for i,day in enumerate(calendar)}
    by_instrument=defaultdict(list)
    for row in selected.to_pylist():
        by_instrument[row['instrument_id']].append(row)
    quant_rows, contexts, source_hashes = [], [], {}
    shard_cache={}
    for instrument, rows in sorted(by_instrument.items()):
        requested=set()
        for row in rows:
            i=positions[row['window_end']]
            days=calendar[i-38:i+1]
            if len(days)!=39 or days[0]!=row['context_start'] or days[19]!=row['window_start']:
                raise ValueError('Selected context calendar mismatch')
            requested.update(int(datetime(d.year,d.month,d.day,15,tzinfo=UTC).timestamp())*10**9 for d in days)
        folder=Path(frozen['source'])/'day01/data/bar'/f'{instrument}-1-DAY-LAST-EXTERNAL'
        paths=list(folder.glob('*.parquet'))
        if len(paths)!=1:
            raise ValueError('Source shape changed')
        path=paths[0]
        expected=frozen['source_files'][str(path.relative_to(Path(frozen['source'])))]
        if sha256(path)!=expected:
            raise ValueError('Source changed')
        source_hashes[str(path)]=expected
        # Predicate applied before decoding: only the union of authorized context dates.
        raw=pq.read_table(path, filters=[('ts_event','in',sorted(requested))], use_threads=False)
        lookup={int(r['ts_event']):r for r in raw.to_pylist()}
        if set(lookup)!=requested or len(raw)!=len(requested):
            raise ValueError('Raw context dates mismatch')
        if (raw.schema.metadata or {}).get(b'instrument_id',b'').decode()!=instrument:
            raise ValueError('Raw identity mismatch')
        for row in rows:
            i=positions[row['window_end']]
            times=[int(datetime(d.year,d.month,d.day,15,tzinfo=UTC).timestamp())*10**9 for d in calendar[i-38:i+1]]
            vals=np.array([[float(decode_fixed_i128(lookup[t][f])) for f in ('open','high','low','close','volume')] for t in times])
            u,q,v,h=quantize(vals[:,:4],vals[-20:,4])
            visible=np.maximum(1,h)
            scaled=quantize(vals[:,:4]*2,vals[-20:,4]*3)
            if not np.allclose(u,scaled[0],atol=1e-12,rtol=0) or not np.allclose(v,scaled[2],atol=1e-12,rtol=0):
                raise ValueError('Normalization invariance mismatch')
            pairs=list(itertools.combinations(range(5),2))
            newly_equal=sum(int(((q[:,a]==q[:,b])&(u[:,a]!=u[:,b])).sum()) for a,b in pairs)
            shard=int(row['shard_id'])
            if shard not in shard_cache:
                name=f'train/{shard:04d}.npy'
                if sha256(ORIGINAL/name)!=plan['input_sha256'][name]:
                    raise ValueError('Image input changed')
                shard_cache[shard]=np.load(ORIGINAL/name,mmap_mode='r',allow_pickle=False)
            image=shard_cache[shard][int(row['shard_row'])]
            if image.shape!=(64,60) or not np.all((image==0)|(image==255)):
                raise ValueError('Invalid stored image')
            observed=[int(np.count_nonzero(image[52:,3*j+1])) for j in range(20)]
            if observed!=visible.tolist():
                raise ValueError('Visible volume coordinates mismatch')
            for j in range(20):
                center=3*j+1
                for k,x in ((0,center-1),(1,center),(2,center),(3,center+1),(4,center)):
                    if image[50-int(q[j,k]),x]!=255:
                        raise ValueError('Price/MA coordinate mismatch')
            quant_rows.append({'sample_id':row['sample_id'],'instrument_id':instrument,'year':row['window_end'].year,
                'price_mae':float(np.abs(u-q/50).mean()),'price_max_error':float(np.abs(u-q/50).max()),
                'volume_mae':float(np.abs(v-visible/12).mean()),'volume_max_error':float(np.abs(v-visible/12).max()),
                'new_equal_y_pairs':newly_equal,'pair_opportunities':200,
                'collapsed_hl_days':int(((q[:,1]==q[:,2])&(u[:,1]!=u[:,2])).sum()),
                'volume_one_pixel_days':int((visible==1).sum()),'volume_rounded_zero_days':int((h==0).sum())})
            contexts.append({'sample_id':row['sample_id'],'instrument_id':instrument,'times':times,'ohlcv':vals.tolist()})
        if sha256(path)!=expected:
            raise ValueError('Source changed during read')
    pq.write_table(pa.Table.from_pylist(quant_rows),out/'quantization.parquet')
    write_json(out/'selected-contexts.json',contexts)
    write_json(out/'raw-inputs.json',{'sources':source_hashes,'calendar':str(calendar_path),'calendar_sha256':sha256(calendar_path),
                                   'scope':'only exact selected 39-day context unions decoded'})
    history=read_json(ORIGINAL/'history.json')
    seal=read_json(ORIGINAL/'seal.json')
    checkpoint=torch.load(ORIGINAL/'best.pt',map_location='cpu',weights_only=True)
    best=min(history,key=lambda r:r['validation_loss'])['epoch']
    if checkpoint['epoch']!=best or seal['best_epoch']!=best:
        raise ValueError('Checkpoint selection mismatch')
    if not all(torch.isfinite(t).all().item() for t in checkpoint['model'].values()):
        raise ValueError('Nonfinite checkpoint state')
    for name in ('config','normalization','frozen'):
        if checkpoint[name+'_sha256']!=sha256(ORIGINAL/(name+'.json')):
            raise ValueError('Checkpoint metadata mismatch')
    qsummary={'windows':len(quant_rows),'instruments':len(by_instrument),
              'year_counts':dict(Counter(str(r['year']) for r in quant_rows)),
              'mean_price_mae':float(np.mean([r['price_mae'] for r in quant_rows])),
              'max_price_error':max(r['price_max_error'] for r in quant_rows),
              'mean_volume_mae':float(np.mean([r['volume_mae'] for r in quant_rows])),
              'max_volume_error':max(r['volume_max_error'] for r in quant_rows),
              **{key:sum(r[key] for r in quant_rows) for key in ('new_equal_y_pairs','pair_opportunities','collapsed_hl_days','volume_one_pixel_days','volume_rounded_zero_days')},
              'display_days':len(quant_rows)*20,'normalization_invariance':'price x2 and volume x3 normalized values agree to atol1e-12',
              'scope':'fixed hash convenience sample; equal y is not necessarily pixel overlap; no population extrapolation'}
    decisions=pq.read_table(parent/'decisions.parquet', columns=['decision_date','candidate_split','reason'], filters=[('candidate_split','in',list(COUNTS))], use_threads=False)
    exclusion_counts=Counter(f"{r['decision_date'].year}|{r['candidate_split']}|{r['reason']}" for r in decisions.to_pylist())
    if sha256(parent/'decisions.parquet')!=frozen['parent_files']['decisions.parquet']:
        raise ValueError('Parent decisions changed')
    diagnostics={'exclusions_by_year_split_reason':dict(exclusion_counts),'scope':'development','protocol':'research-v2/G7','groups':groups,'probability_bins':distributions,
      'quantization':qsummary,'checkpoint':{'best_epoch':best,'final_epoch':history[-1]['epoch'],'model_tensors_finite':True,
      'new_gradient_computation':False,'gradient_evidence':'existing G3/G4 finite-gradient tests and G5 runtime guard; no saved per-step gradient trace',
      'history_loss_caveat':'train-mode changing weights/dropout versus fixed eval predictions are not directly comparable'},
      'training_performed':False,'inference_performed':False,'test_analysis_performed':False}
    write_json(out/'diagnostics.json',diagnostics)
    for grain in ('year','quarter'):
        chosen=[r for r in groups if r['grain']==grain]
        plot_lines(out/f'{grain}-loss.png',f'G7 {grain}: fixed eval loss; Train in-sample / 2024 selected validation',
          [r['period'] for r in chosen],{'CNN':[r['log_loss'] for r in chosen],'Train constant':[r['constant_log_loss'] for r in chosen]},
          min(min(r['log_loss'],r['constant_log_loss']) for r in chosen)-.01,max(max(r['log_loss'],r['constant_log_loss']) for r in chosen)+.01)
    canvas=Image.new('RGB',(900,650),'white')
    draw=ImageDraw.Draw(canvas)
    draw.text((40,15),'G7 Reliability: 10 fixed bins; x=mean p, y=observed positive fraction',fill='black')
    for step in range(11):
        x=70+700*step/10
        y=560-470*step/10
        draw.line((70,y,770,y),fill='#dddddd')
        draw.text((25,y-4),f'{step/10:.1f}',fill='black')
        draw.text((x,575),f'{step/10:.1f}',fill='black')
    draw.line((70,560,770,90),fill='#888888',width=2)
    for k,(split,entries) in enumerate(distributions.items()):
        color=['#2166ac','#b2182b'][k]
        points=[(70+700*b['mean_probability'],560-470*b['positive_rate']) for b in entries if b['count']]
        if len(points)>1:
            draw.line(points,fill=color,width=3)
        for x,y in points:
            draw.ellipse((x-4,y-4,x+4,y+4),fill=color)
        draw.text((90+k*300,610),split,fill=color)
    canvas.save(out/'reliability.png')
    plot_lines(out/'probability-distribution.png','Fixed-bin probability fractions; empty bins retained',
        [f'{(i+.5)/10:.2f}' for i in range(10)],
        {split:[b['count']/COUNTS[split] for b in entries] for split,entries in distributions.items()},0,1)
    verify_files(PROJECT,plan['code_sha256'])
    verify_files(ORIGINAL,plan['input_sha256'])


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=Path)
    parser.add_argument('--retry-of',type=Path,help='Explicitly approved single recovery of a failed batch')
    args=parser.parse_args()
    if args.worker:
        check(args.worker)
        return
    gate=PROJECT/'artifacts/research/G7'
    gate.mkdir(parents=True,exist_ok=True)
    previous = list(gate.glob('*/plan.json'))
    prior = None
    spent_seconds, spent_bytes = 0.0, 0
    if previous:
        if args.retry_of is None or len(previous) != 1:
            raise ValueError('G7 batch already exists: no unapproved or repeated recovery')
        prior = args.retry_of.resolve(strict=True)
        if prior != previous[0].parent.resolve() or read_json(prior/'manifest.json')['status'] != 'failed':
            raise ValueError('Recovery must name the only failed batch')
        spent_seconds = read_json(prior/'diagnostic-supervision.json')['seconds']
        spent_bytes = sum(p.stat().st_size for p in prior.rglob('*') if p.is_file())
    elif args.retry_of:
        raise ValueError('No failed batch to recover')
    out=gate/datetime.now(UTC).strftime('g7-%Y%m%dT%H%M%S%fZ')
    out.mkdir()
    frozen=read_json(ORIGINAL/'frozen.json')
    saved=read_json(ORIGINAL/'manifest.json')
    allowed=['frozen.json','config.json','normalization.json','history.json','seal.json','best.pt','training-summary.json',
             'train-predictions.parquet','validation-predictions.parquet','train/index.parquet','validation/index.parquet']
    allowed.extend(name for name in saved['artifacts_sha256'] if name.startswith(('train/','train\\')) and name.endswith('.npy'))
    canonical_hashes=normalize_fingerprints(saved['artifacts_sha256'])
    inputs={name.replace(chr(92),'/'):canonical_hashes[name.replace(chr(92),'/')] for name in allowed}
    if prior and inputs != normalize_fingerprints(read_json(prior/'plan.json')['input_sha256']):
        raise ValueError('Retry input scope changed')
    code={**frozen['code_sha256'],str(Path(__file__).relative_to(PROJECT)):sha256(Path(__file__))}
    write_json(out/'plan.json',{'protocol':'research-v2/G7','scope':'development','allowed_splits':list(COUNTS),
       'input_sha256':inputs,'code_sha256':code,'panels':['all/year/quarter metrics','fixed 10 bins distributions/reliability','2048 smallest Train sample_id quantization','existing checkpoint/history review'],
       'parent_files_sha256':frozen['parent_files'],'source_fingerprints_reference':'G5 frozen.json source_files; selected subset recorded in raw-inputs.json',
       'retry_of':str(prior) if prior else None,'prior_seconds':spent_seconds,'prior_bytes':spent_bytes,
       'retry_selection_sha256':sha256(prior/'selected-train.parquet') if prior else None,
       'recovery_authorization':'user reconfirmed G7 following explicit repair/retry question' if prior else None,
       'selection':2048,'batch_budget':2 if prior else 1,'training_runs':0,'inference_runs':0,'cpu_threads':4,'ram_bytes':4*2**30,'artifact_bytes':2**30,'seconds':1800})
    write_json(out/'manifest.json',{'status':'running','gate':'G7'})
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]='4'
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['PYTHONPATH']=str(PROJECT/'src')
    from run_formal import supervise
    print(json.dumps({'out':str(out),'event':'started'}),flush=True)
    try:
        supervise(out,'diagnostic',1800-spent_seconds,{'ram_limit_bytes':4*2**30,'artifact_limit_bytes':2**30-spent_bytes},
                  [sys.executable,str(Path(__file__).resolve()),'--worker',str(out)])
        write_json(out/'manifest.json',{'status':'completed','gate':'G7','acceptance':'pending independent audit',
          'artifacts_sha256':{str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'}})
        print(json.dumps({'out':str(out),'event':'completed'}),flush=True)
    except BaseException as exc:
        write_json(out/'manifest.json',{'status':'failed','gate':'G7','error':str(exc)})
        raise


if __name__=='__main__':
    main()
