"""Read-back reconciliation of persisted formal predictions, without model inference."""
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from cb_image_alpha.artifacts import read_json, write_json
from cb_image_alpha.evaluation.metrics import compare_baselines
from cb_image_alpha.training.formal_data import COUNTS, verify_split


def verify_results(run: Path) -> None:
    parent = Path(read_json(run / 'frozen.json')['parent'])
    saved = read_json(run / 'metrics.json')
    normalization = read_json(run / 'normalization.json')
    train_labels = np.load(run / 'train/labels.npy', allow_pickle=False)
    rate = float(train_labels.mean())
    if rate != saved['train_rate'] or rate != normalization['positive_rate']:
        raise ValueError('Train baseline rate mismatch')
    history = read_json(run / 'history.json')
    best = min(history, key=lambda row: row['validation_loss'])['epoch']
    if best != read_json(run / 'seal.json')['best_epoch']:
        raise ValueError('Best checkpoint selection mismatch')
    report = {}
    for split, count in COUNTS.items():
        verify_split(run, split)
        index = pq.read_table(run / split / 'index.parquet')
        predictions = pq.read_table(run / f'{split}-predictions.parquet')
        columns = ['sample_id', 'image_row', 'label', 'decision_time']
        expected = pq.read_table(parent / 'samples.parquet', columns=columns,
                                 filters=[('split', '=', split)]).sort_by([('image_row', 'ascending')])
        if len(predictions) != count or not predictions.select(columns).equals(expected):
            raise ValueError('Predictions differ from frozen parent')
        if not predictions.select(index.column_names).equals(index):
            raise ValueError('Prediction index mismatch')
        ids = predictions['sample_id'].to_pylist()
        if len(set(ids)) != count or any(v is None for v in ids):
            raise ValueError('Invalid sample IDs')
        labels = np.load(run / split / 'labels.npy', allow_pickle=False)
        if not np.array_equal(labels, predictions['label'].to_numpy()):
            raise ValueError('Label mismatch')
        p = predictions['p_up'].to_numpy()
        recalculated = compare_baselines(labels, p, saved['train_rate'])
        if recalculated != saved['splits'][split]:
            raise ValueError('Saved metrics mismatch')
        info = read_json(run / split / 'manifest.json')
        offset = 0
        white = 0
        for number, shard in enumerate(info['shards']):
            pixels = np.load(run / split / shard['file'], mmap_mode='r', allow_pickle=False)
            n = shard['count']
            if pixels.shape != (n, 64, 60) or pixels.dtype != np.uint8:
                raise ValueError('Shard shape mismatch')
            if split == 'train':
                if not np.all((pixels == 0) | (pixels == 255)):
                    raise ValueError('Nonbinary pixels')
                white += int(np.count_nonzero(pixels))
            rows = index.slice(offset, n)
            if not np.all(rows['shard_id'].to_numpy() == number):
                raise ValueError('Shard mapping mismatch')
            if not np.array_equal(rows['shard_row'].to_numpy(), np.arange(n)):
                raise ValueError('Shard row mismatch')
            offset += n
        if offset != count:
            raise ValueError('Shard total mismatch')
        if split == 'train':
            mean = white / (count * 3840)
            if mean != normalization['mean'] or float(np.sqrt(mean * (1-mean))) != normalization['std']:
                raise ValueError('Train normalization mismatch')
            if white != info['white_pixels']:
                raise ValueError('Pixel count mismatch')
        report[split] = {'count': count, 'unique_ids': count, 'parent_alignment': True,
                         'persisted_metrics_match': True, 'shard_mapping': True}
    write_json(run / 'result-verification.json', {'status': 'passed', 'splits': report})
