# SPDX-License-Identifier: AGPL-3.0-only
"""Quantify routed-expert FP32 differences and their absorption between two HSL runs.

Run on a Spark (or inside the reference image) with hsl_localize.py beside it:
    hsl_absorb.py RUN_A RUN_B RANK
Per request: MoE layers whose fused-kernel FP32 output differs, the first
post-reduce and FFN-input differences, any other stage differing before the
FFN input, and element/magnitude statistics at the first differing layer.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hsl_localize as hl
R = '/home/colum/hsl-20261015/attempts/'
a_name, b_name, rank = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows_a = json.load(open(R + a_name + '/rows.json')); rows_b = json.load(open(R + b_name + '/rows.json'))
A = hl.index(R + a_name + '/dump', rows_a); Bx = hl.index(R + b_name + '/dump', rows_b)
# file lookup: stage key -> bin path, via metadata scan (by request id + layer + stage)
def files(run, rows):
    rid = {v[0]: k for k, v in rows.items()}; out = {}
    fused = {}
    for p in glob.glob(R + run + '/dump/ffn-L*-rank%d-*.json' % rank):
        m = hl.STEM.match(os.path.basename(p))
        if not m: continue
        meta = json.load(open(p))
        if 'file' not in meta or 'chunk_position' not in meta: continue
        row = next(rid[x] for x in rid if meta['chunk_position']['request_id'].startswith(x))
        if m[4] in ('routed-weighted-f32', 'routed-output', 'post-tp-output', 'input'):
            out[(row, int(m[1]), m[4])] = (R + run + '/dump/' + meta['file'], meta['dtype'], meta['shape'])
    return out
FA, FB = files(a_name, rows_a), files(b_name, rows_b)
def load(f):
    path, dtype, shape = f
    raw = np.fromfile(path, dtype={'bf16-le': '<u2', 'f32-le': '<f4', 'f16-le': '<f2', 'i32-le': '<i4'}[dtype])
    if dtype == 'bf16-le':
        raw = (raw.astype(np.uint32) << 16).view(np.float32)
    return raw.reshape(shape).astype(np.float64)
summary = []
for row in sorted(rows_a):
    ta, tb = A[row, rank], Bx[row, rank]
    diff_layers = lambda st: [l for (l, s) in sorted(ta) if s == st and ta[(l, s)][1] != tb[(l, s)][1]]
    rw, ro, pt, fi = (diff_layers(s) for s in ('routed-weighted-f32', 'routed-output', 'post-tp-output', 'input'))
    first_in = fi[0] if fi else None
    # any non-routed difference before the first FFN-input difference?
    early = sorted({s for (l, s) in ta if ta[(l, s)][1] != tb[(l, s)][1]
                    and (first_in is None or l < first_in)} - {'routed-weighted-f32', 'routed-output'})
    l3 = rw[0]
    x, y = load(FA[(row, l3, 'routed-weighted-f32')]), load(FB[(row, l3, 'routed-weighted-f32')])
    d = np.abs(x - y); scale = np.abs(x).max()
    ulp = np.abs(x.astype(np.float32).view(np.int32).astype(np.int64) - y.astype(np.float32).view(np.int32).astype(np.int64))
    summary.append({'row': row, 'routed_f32_diff_layers': len(rw), 'routed_bf16_diff_layers': len(ro),
                    'post_tp_first_diff': pt[0] if pt else None, 'ffn_input_first_diff': first_in,
                    'non_routed_diffs_before_ffn_input': early,
                    'first_routed_f32_layer': l3, 'elements_differing': int((d > 0).sum()), 'elements': int(d.size),
                    'max_abs': float(d.max()), 'max_ulp_f32': int(ulp.max()), 'max_abs_value': float(scale)})
print(json.dumps(summary))
