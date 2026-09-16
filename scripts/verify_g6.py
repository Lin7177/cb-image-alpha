"""G6 read-only reconstruction against a sealed G5 run; no training or Test inference."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from cb_image_alpha.artifacts import read_json, sha256, verify_files, write_json
from cb_image_alpha.evaluation.metrics import compare_baselines
from cb_image_alpha.models.paper_cnn import PaperCNN
from cb_image_alpha.training.formal_data import ImageBatches, prepare_split
from cb_image_alpha.training.formal_engine import setup

ATOL, RTOL = 1e-7, 1e-6
PROJECT = Path(__file__).resolve().parents[1]


def check(original: Path, out: Path) -> None:
    manifest = read_json(original / 'manifest.json')
    frozen = read_json(original / 'frozen.json')
    config = read_json(original / 'config.json')
    if manifest['status'] != 'completed' or manifest['scope'] != 'formal':
        raise ValueError('Incomplete formal parent')
    verify_files(original, manifest['artifacts_sha256'])
    verify_files(PROJECT, frozen['code_sha256'])
    parent = Path(frozen['parent'])
    verify_files(parent, frozen['parent_files'])
    old = read_json(parent / 'manifest.json')
    for name in ('configs/data_policy.toml', 'configs/splits.toml',
                 'scripts/build_sample_index.py', 'src/cb_image_alpha/samples/index.py'):
        expected = old['inputs']['code_sha256'].get(name.replace('/', '\\'))
        expected = expected or old['inputs']['code_sha256'].get(name)
        if expected != sha256(PROJECT / name):
            raise ValueError(f'G2 semantic code/config changed: {name}')
    source = Path(frozen['source'])
    verify_files(source, frozen['source_files'])
    for name, expected in frozen['auxiliary_sha256'].items():
        if sha256(Path(name)) != expected:
            raise ValueError('Auxiliary changed')
    reader = PROJECT / 'artifacts/data_audit/reader-validation-20260911T062403608042Z.json'
    quality = PROJECT / 'artifacts/data_audit/quality-20260911T053105796931Z.json'
    if sha256(reader) != old['inputs']['reader_report_sha256'] or sha256(quality) != old['inputs']['quality_report_sha256']:
        raise ValueError('G1 evidence mismatch')
    command = [sys.executable, str(PROJECT / 'scripts/build_sample_index.py'),
               '--source', str(source), '--reader-report', str(reader), '--quality-report', str(quality)]
    with (out / 'index.stdout.log').open('w', encoding='utf-8') as stdout, (out / 'index.stderr.log').open('w', encoding='utf-8') as stderr:
        result = subprocess.run(command, cwd=PROJECT, stdout=stdout, stderr=stderr, timeout=900, check=False,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    write_json(out / 'index-command.json', {'command': command, 'exit_code': result.returncode})
    if result.returncode:
        raise ValueError('Index rebuild failed')
    rebuilt = Path(json.loads((out / 'index.stdout.log').read_text(encoding='utf-8'))['output'])
    exact = {}
    for name, expected in old['artifacts_sha256'].items():
        if sha256(rebuilt / name) != expected:
            raise ValueError(f'Index reconstruction mismatch: {name}')
        exact[name] = expected
    write_json(out / 'index-verification.json', {'status': 'passed', 'rebuilt': str(rebuilt), 'exact_sha256': exact})
    images = out / 'reconstruction'
    images.mkdir()
    write_json(images / 'frozen.json', frozen)
    write_json(images / 'config.json', config)
    image_checks = {}
    for split in ('train', 'validation'):
        prepare_split(images, split)
        saved = read_json(original / split / 'manifest.json')
        fresh = read_json(images / split / 'manifest.json')
        if fresh != saved:
            raise ValueError(f'Image manifest mismatch: {split}')
        image_checks[split] = {'count': saved['count'], 'all_shards_exact': True,
                               'index_and_labels_exact': True, 'shards': len(saved['shards'])}
    if read_json(images / 'normalization.json') != read_json(original / 'normalization.json'):
        raise ValueError('Normalization reconstruction mismatch')
    write_json(out / 'image-verification.json', image_checks)
    setup(config)
    checkpoint = torch.load(original / 'best.pt', map_location='cuda', weights_only=True)
    seal = read_json(original / 'seal.json')
    verify_files(original, seal['files'])
    for key, filename in (('config_sha256', 'config.json'), ('normalization_sha256', 'normalization.json'), ('frozen_sha256', 'frozen.json')):
        if checkpoint[key] != sha256(original / filename):
            raise ValueError('Checkpoint input mismatch')
    if checkpoint['epoch'] != seal['best_epoch']:
        raise ValueError('Best epoch mismatch')
    model = PaperCNN().cuda()
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    inference = {}
    for split in ('train', 'validation'):
        data = ImageBatches(images, split)
        probabilities = np.empty(data.length, dtype=np.float64)
        with torch.no_grad():
            for start in range(0, data.length, config['batch_size']):
                indices = np.arange(start, min(start + config['batch_size'], data.length))
                x, _ = data.batch(indices)
                probabilities[indices] = model(x.cuda()).softmax(1)[:, 1].cpu().numpy()
        saved = pq.read_table(original / f'{split}-predictions.parquet', columns=['sample_id', 'p_up'])
        index = pq.read_table(images / split / 'index.parquet', columns=['sample_id'])
        if not saved.select(['sample_id']).equals(index):
            raise ValueError('Recovery ID order mismatch')
        expected = saved['p_up'].to_numpy()
        if not np.isfinite(probabilities).all() or not np.allclose(probabilities, expected, atol=ATOL, rtol=RTOL):
            raise ValueError('Recovery inference exceeds frozen tolerance')
        np.save(out / f'{split}-restored-probabilities.npy', probabilities, allow_pickle=False)
        inference[split] = {'count': data.length, 'max_abs_difference': float(np.max(np.abs(probabilities-expected))), 'atol': ATOL, 'rtol': RTOL}
        del data
    metrics = read_json(original / 'metrics.json')
    metric_checks = {}
    for split in ('train', 'validation', 'test'):
        table = pq.read_table(original / f'{split}-predictions.parquet')
        columns = ['sample_id', 'image_row', 'label', 'decision_time']
        expected = pq.read_table(rebuilt / 'samples.parquet', columns=columns, filters=[('split', '=', split)]).sort_by([('image_row','ascending')])
        if not table.select(columns).equals(expected):
            raise ValueError('Saved prediction mapping mismatch')
        calculated = compare_baselines(table['label'].to_numpy(), table['p_up'].to_numpy(), metrics['train_rate'])
        if calculated != metrics['splits'][split]:
            raise ValueError('Saved metric mismatch')
        metric_checks[split] = {'existing_prediction_metrics_exact': True, 'count': len(table)}
    verify_files(original, manifest['artifacts_sha256'])
    verify_files(PROJECT, frozen['code_sha256'])
    verify_files(source, frozen['source_files'])
    write_json(out / 'verification.json', {'status':'passed', 'original':str(original),
        'index':exact,'images':image_checks,'inference':inference,'metrics':metric_checks,
        'test_inference_performed':False,'training_performed':False,
        'environment':{'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,
                       'gpu':torch.cuda.get_device_name(),'lock_sha256':sha256(PROJECT/'uv.lock')}})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--worker-output', type=Path)
    args = parser.parse_args()
    original = args.run.resolve(strict=True)
    if not original.is_relative_to((PROJECT/'artifacts/runs').resolve()):
        raise ValueError('Run outside project artifacts')
    if args.worker_output:
        output = args.worker_output.resolve(strict=True)
        if not output.is_relative_to((PROJECT/'artifacts').resolve()) or output == original:
            raise ValueError('Unsafe verification output')
        check(original, output)
        return
    out = PROJECT / 'artifacts' / datetime.now(UTC).strftime('g6-%Y%m%dT%H%M%S%fZ')
    out.mkdir(exist_ok=False)
    write_json(out/'manifest.json', {'status':'running','gate':'G6','original':str(original),
        'atol':ATOL,'rtol':RTOL,'verifier_sha256':sha256(Path(__file__))})
    from run_formal import supervise
    env = os.environ
    env['PYTHONPATH'] = str(PROJECT/'src')
    env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    print(json.dumps({'output':str(out),'event':'started'}),flush=True)
    try:
        supervise(out,'rebuild',1800,{'ram_limit_bytes':8589934592,'artifact_limit_bytes':8589934592},
                  [sys.executable,str(Path(__file__).resolve()),'--run',str(original),'--worker-output',str(out)])
        write_json(out/'manifest.json',{'status':'completed','gate':'G6','original':str(original),
            'atol':ATOL,'rtol':RTOL,'verifier_sha256':sha256(Path(__file__)),
            'artifacts_sha256':{str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file() and p != out/'manifest.json'}})
        print(json.dumps({'output':str(out),'event':'completed'}),flush=True)
    except BaseException as exc:
        write_json(out/'manifest.json',{'status':'failed','gate':'G6','error':str(exc)})
        raise


if __name__ == '__main__':
    main()
