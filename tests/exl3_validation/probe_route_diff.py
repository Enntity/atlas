# SPDX-License-Identifier: AGPL-3.0-only
"""Compare two probe attempts' router telemetry per request and layer.

Run on a Spark: probe_route_diff.py ATTEMPT_A ATTEMPT_B, with /tmp/ids-<attempt>.json
mapping collector rows to (response id, prompt length). Prints per-row first
differing (invocation, layer) per stage, then a strict check that top-k IDs
never differ where the same layer's gate logits are byte-identical.
"""
import json, pathlib, hashlib, sys, collections
def load(attempt, ids):
    root = pathlib.Path('/home/colum/probe-rm-20261014/attempts/%s/telemetry' % attempt)
    by_req = {}
    rid_to_row = {v[0]: k for k, v in ids.items()}
    out = collections.defaultdict(dict)
    for rec in root.glob('pid-*/flush-*/records.json'):
        d = json.loads(rec.read_text())
        for r in d['records']:
            if r['kind'] != 'tensor' or r['request'] is None: continue
            rid = next((x for x in rid_to_row if r['request'].startswith(x)), None)
            if rid is None: continue
            stage = r['name'].rsplit('-', 1)[0].replace('router-', '')
            layer, rank, inv, _ = r['ctx']
            h = hashlib.sha256((rec.parent / r['file']).read_bytes()).hexdigest()
            out[rid_to_row[rid]][(inv, layer, stage)] = h
    return out
a, b = sys.argv[1], sys.argv[2]
ia = json.load(open('/tmp/ids-%s.json' % a)); ib = json.load(open('/tmp/ids-%s.json' % b))
A, B = load(a, ia), load(b, ib)
res = {}
for row in sorted(ia):
    ka, kb = A[row], B[row]
    assert set(ka) == set(kb), row
    diff = sorted(k for k in ka if ka[k] != kb[k])
    first = {}
    for st in ('gate-logits', 'topk-ids', 'topk-weights'):
        ds = [k for k in diff if k[2] == st]
        first[st] = list(ds[0][:2]) if ds else None
    res[row] = {'keys': len(ka), 'differing': len(diff), 'first_by_stage': first,
                'ids_differ_count': sum(1 for k in diff if k[2] == 'topk-ids')}
print(json.dumps(res))
# --- strict check: ids differing where the same (inv, layer) gate logits are identical
strict = 0; lags = []
for row in sorted(ia):
    ka, kb = A[row], B[row]
    for (inv, layer, st) in ka:
        if st == 'topk-ids' and ka[(inv, layer, st)] != kb[(inv, layer, st)] \
                and ka[(inv, layer, 'gate-logits')] == kb[(inv, layer, 'gate-logits')]:
            strict += 1
    g = res[row]['first_by_stage']['gate-logits']; i = res[row]['first_by_stage']['topk-ids']
    if g and i: lags.append((i[0] - g[0], i[1] - g[1]))
print('STRICT ids-differ-with-identical-gate:', strict, 'lags (inv,layer):', sorted(set(lags)))
