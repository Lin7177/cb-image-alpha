"""Read only the recovered development labels and preserve endpoint audit evidence."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from paper_data import LIMIT_SECONDS, sha256
from pyarrow import feather
from recover_paper_data import RUN


def main() -> None:
    started = time.monotonic()
    recovered = RUN / 'recovery-1'
    output = recovered / 'endpoint-audit.json'
    if output.exists():
        raise FileExistsError('Endpoint evidence already exists; no silent overwrite')
    parent = json.loads((recovered / 'manifest.json').read_text())
    supervision = json.loads((recovered / 'supervision.json').read_text())
    if parent['status'] != 'completed' or supervision['status'] != 'completed':
        raise ValueError('Recovery not completed')
    years = []
    for year in range(1993, 2001):
        path = recovered / 'monthly_20d' / f'20d_month_has_vb_[20]_ma_{year}_labels_w_delay.feather'
        table = feather.read_table(path)
        dates = np.array(table['Date'].to_pylist(), dtype='datetime64[D]')
        returns = np.asarray(table['Ret_month'].to_pylist(), dtype=np.float64)
        vol = np.asarray(table['EWMA_vol'].to_pylist(), dtype=np.float64)
        valid_return = np.isfinite(returns) & (returns != -99)
        valid_vol = np.isfinite(vol) & (vol > 0)
        tail = dates >= np.datetime64('2000-12-01')
        # Descriptive disjoint counts only, no sample index or protocol selection.
        counts = {
            'invalid_return': int((~valid_return).sum()),
            'invalid_vol_after_valid_return': int((valid_return & ~valid_vol).sum()),
            'tail_after_valid_return_and_vol': int((valid_return & valid_vol & tail).sum()),
            'remaining_descriptive_candidates': int((valid_return & valid_vol & ~tail).sum()),
        }
        if sum(counts.values()) != table.num_rows:
            raise ValueError('Counts do not reconcile')
        years.append({'year': year, 'rows': table.num_rows, 'input_sha256': sha256(path),
                      'fields': [{'name': f.name, 'type': str(f.type)} for f in table.schema],
                      'schema_metadata': {k.decode(): v.decode() for k,v in (table.schema.metadata or {}).items()},
                      'descriptive_exclusion_order': counts})
    elapsed = time.monotonic() - started
    cumulative = supervision['cumulative_seconds'] + elapsed
    if cumulative > LIMIT_SECONDS:
        raise TimeoutError('Cumulative acquisition and endpoint audit budget exceeded')
    report = {'status': 'completed', 'research_gate': 'BLOCKED',
              'verdict': 'ACTUAL_RET_MONTH_ENDPOINT_NOT_RELIABLY_VERIFIABLE',
              'actual_cross_oos_contamination': 'NOT_ESTABLISHED',
              'cross_oos_risk': 'CANNOT_EXCLUDE_FROM_AVAILABLE_EVIDENCE',
              'years': years, 'wall_seconds': elapsed, 'cumulative_recorded_seconds': cumulative,
              'source_manifest_sha256': sha256(recovered/'manifest.json'),
              'script_sha256': sha256(Path(__file__)),
              'reason': 'Only Date and return values are supplied; no realized return end date or underlying period-end return sequence is present. Official shift(-1) uses next available period-end, without adjacency requirement.',
              'limits': ['No OOS label/image decoded', 'No training or inference',
                         'No exact endpoint reconstructed', 'No new selection index',
                         'No claim that an actual cross-boundary label was observed']}
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'years':len(years),'rows':sum(x['rows'] for x in years),
                      'counts':{k:sum(y['descriptive_exclusion_order'][k] for y in years) for k in counts},
                      'cumulative_recorded_seconds':cumulative,'verdict':report['verdict']}))


if __name__ == '__main__':
    main()