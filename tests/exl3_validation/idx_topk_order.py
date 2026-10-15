# SPDX-License-Identifier: AGPL-3.0-only
"""Compare indexer top-k index buffers of two idx runs in the final prefill chunk.

Run on a Spark (or inside the reference image): idx_topk_order.py RUN_A RUN_B RANK.
For every valid token row of the chunk at computed_before=2048 on MLA layers 3
and 7: byte equality, selected-set equality, and how many positions differ.
"""
import glob
import json
import sys

import numpy as np
R = '/home/colum/idx-20261015/attempts/'
def find(run, rank, rid, chunk, layer, stage):
    for p in glob.glob(R + run + '/dump/ffn-L%d-rank%d-*-%s.json' % (layer, rank, stage)):
        m = json.load(open(p))
        cp = m.get('chunk_position') or {}
        if cp.get('request_id', '').startswith(rid) and cp.get('computed_before') == chunk:
            dt = {'i32-le': '<i4', 'bf16-le': '<u2', 'f32-le': '<f4'}[m['dtype']]
            a = np.fromfile(R + run + '/dump/' + m['file'], dtype=dt).reshape(m['shape'])
            return a, cp
    raise SystemExit('missing %s %s %s' % (run, stage, rid))
a_run, b_run, rank = sys.argv[1], sys.argv[2], int(sys.argv[3])
ra = json.load(open(R + a_run + '/rows.json')); rb = json.load(open(R + b_run + '/rows.json'))
out = []
for row in sorted(ra):
    for layer in (3, 7):
        A, cp = find(a_run, rank, ra[row][0], 2048, layer, 'attn-indexer_indexer_op')
        B, _ = find(b_run, rank, rb[row][0], 2048, layer, 'attn-indexer_indexer_op')
        n = cp['scheduled_tokens']
        rec = {'row': row, 'layer': layer, 'rank': rank, 'valid_rows': n, 'logical_after': cp['logical_after']}
        rec['stale_rows_equal'] = bool((A[n:] == B[n:]).all())
        per = []
        for t in range(n):
            x, y = A[t], B[t]
            xs, ys = x[x >= 0], y[y >= 0]
            per.append({'t': t, 'n_valid_idx': [int(xs.size), int(ys.size)],
                        'bytes_equal': bool((x == y).all()),
                        'same_set': bool(set(xs.tolist()) == set(ys.tolist())),
                        'positions_differing': int((x != y).sum()),
                        'set_symdiff': sorted(set(xs.tolist()) ^ set(ys.tolist()))[:12]})
        rec['tokens'] = per
        out.append(rec)
print(json.dumps(out))
