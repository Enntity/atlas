#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Q2 diagnostic: stage-pair localization, candidate (cand-q2diag-ee5eaf0f-02)
vs reference (ref-q2diag-ctxv4diag-03), rows 0-7, chaos floor from reference
probe invocations 8-10. Diagnostic localization only; no gate claim.

Binding (from extended-plan.json, verified empirically):
- ffn/router/attn/kda/upstream/mhc-pre stages: invocation = request index.
- mhc-fused layers 1-3: invoke 2r = attn-side call, 2r+1 = FFN-side call.
- candidate invoke i == reference invoke i (same request).
"""
import argparse
import hashlib
import json
import pathlib

import numpy as np
from parity_protocol import load_artifact, validate_run_pair

ROWS = 8          # invocations 0-7 = requests (arithmetic prefixes 10-13, longer 36-39)
PROBES = (8, 9, 10)


def bf16_key(u16):
    u = u16.astype(np.uint32)
    return np.where((u & 0x8000) != 0, (~u) & 0xFFFF, u | 0x8000)


def metrics(a, b):
    if a.shape != b.shape:
        raise ValueError(f'mapped shape differs: {a.shape}, {b.shape}')
    if a.dtype != b.dtype or a.dtype not in (np.dtype('<u2'), np.dtype('<f4'), np.dtype('<i4')):
        raise ValueError('unsupported or mismatched metric dtype')
    if not a.size:
        raise ValueError('empty tensor')
    if a.dtype == np.dtype('<i4'):
        result = {'kind': 'ids', 'exact': bool(np.array_equal(a, b)),
                  'diff_count': int(np.count_nonzero(a != b))}
        if a.ndim == 2:
            result['membership_different_rows'] = int(np.count_nonzero(
                np.any(np.sort(a, axis=1) != np.sort(b, axis=1), axis=1)))
        return result
    finite_a = (a & 0x7f80) != 0x7f80 if a.dtype == np.dtype('<u2') else np.isfinite(a)
    finite_b = (b & 0x7f80) != 0x7f80 if b.dtype == np.dtype('<u2') else np.isfinite(b)
    if not finite_a.all() or not finite_b.all():
        raise ValueError('nonfinite internal tensor')
    if a.dtype == np.dtype('<f4'):
        d = np.abs(a.astype(np.float64) - b.astype(np.float64))
        return {'kind': 'f32', 'exact': a.tobytes() == b.tobytes(),
                'max_abs': float(d.max()), 'mean_abs': float(d.mean())}
    ka, kb = bf16_key(a.view(np.uint16)), bf16_key(b.view(np.uint16))
    du = np.abs(ka.astype(np.int64) - kb.astype(np.int64))
    return {'kind': 'bf16', 'exact': bool(np.all(du == 0)),
            'exact_pct': round(float((du == 0).mean() * 100), 4),
            'p50': float(np.percentile(du, 50)), 'p99': float(np.percentile(du, 99)),
            'max': int(du.max())}


def load(root):
    out = {}
    for f in sorted(root.glob('ffn-L*-rank*-*.json')):
        m = json.loads(f.read_text())
        if 'stage' not in m:
            continue
        stages = out.setdefault((int(m['layer']), int(m['invocation'])), {})
        if m['stage'] in stages:
            raise ValueError('duplicate stage binding')
        stages[m['stage']] = (m, f)
    return out


DTYPES = {'bf16-le': '<u2', 'f32-le': '<f4', 'i32-le': '<i4'}


def tensors(side):
    m, f = side
    rows = []
    for t in m.get('tensors', [m]):
        if t['dtype'] not in DTYPES:
            raise ValueError('unsupported capture dtype')
        name = t['file']
        if pathlib.Path(name).name != name:
            raise ValueError('capture filename must be local')
        path = f.parent / name
        if path.is_symlink() or not path.is_file():
            raise ValueError('capture must be a regular file')
        shape = t['shape']
        if not shape or any(type(d) is not int or d <= 0 for d in shape):
            raise ValueError('invalid capture shape')
        dtype = np.dtype(DTYPES[t['dtype']])
        if path.stat().st_size != int(np.prod(shape, dtype=object)) * dtype.itemsize:
            raise ValueError('capture byte count differs from shape')
        arr = np.fromfile(path, dtype=dtype).reshape(shape)
        rows.append((t.get('name'), arr))
    return rows


def kda_layout(reference, candidate_shape):
    # Source-attested singleton batch + contiguous head dimensions only.
    if reference.shape == candidate_shape:
        return reference
    if len(candidate_shape) == 3 and candidate_shape[-2:] == (32, 128):
        if reference.shape == (1, *candidate_shape):
            return reference.reshape(candidate_shape)
        raise ValueError('unbound KDA layout')
    if len(candidate_shape) != 2:
        raise ValueError('unbound KDA layout')
    m, width = candidate_shape
    if reference.shape == (1, m, 32, 128) and width == 4096:
        return reference.reshape(candidate_shape)
    if reference.shape == (1, m, 4096) and width == 4096:
        return reference.reshape(candidate_shape)
    raise ValueError('unbound KDA layout')


def producer_gate(get_inputs, ranks):
    if not ranks or len(set(ranks)) != len(ranks) or any(r not in (0, 1) for r in ranks):
        raise ValueError('explicit producer ranks required')
    checks = {}
    for rank in ranks:
        a, b = get_inputs(rank)
        checks[str(rank)] = metrics(a, b)
    return {'same_input': all(v['exact'] for v in checks.values()), 'ranks': checks,
            'scope': 'dynamic inputs only; weights/config/arithmetic remain unbound'}


def pair_list(r):
    ra, rf = 2 * r, 2 * r + 1
    pl = []
    for L in (0, 1, 2):
        pl.append((f'L{L}.ffn_input', L, 'input', None, r, 'input', None))
        pl.append((f'L{L}.gate_up', L, 'gate_up', ('gate', 'up'), r, 'gate-up', 'cat'))
        if L == 0:
            pl.append(('L0.incoming_hidden', 0, 'incoming_hidden', None, r, 'upstream-embedding', None))
            pl.append(('L0.activation', 0, 'activation', 'activation', r, 'activation', None))
            # layer-0 KDA / attention-side boundaries (pipeline order)
            pl.append(('L0.mhc_attn_normed', 0, 'attn_mhc_norm', 'normed', r, 'mhc-pre-layer-input', None))
            pl.append(('L0.kda_gated_norm', 0, 'gated_norm', 'gated', r, 'kda-gated-norm', None))
            pl.append(('L0.kda_oproj_input', 0, 'gated_norm', 'gated', r, 'kda-o-proj-input0', None))
            pl.append(('L0.kda_reduced', 0, 'attention_reduced_output', 'output', r, 'kda-o-proj', None))
            pl.append(('L0.kda_g1', 0, 'recurrent_inputs', 'g1', r, 'kda-g1', None))
            pl.append(('L0.kda_g2', 0, 'recurrent_inputs', 'g2', r, 'kda-g2', None))
            pl.append(('L0.kda_core_out', 0, 'recurrent_output', 'core', r, 'kda-core-output', None))
            pl.append(('L0.ffn_mhc_normed', 0, 'ffn_mhc_norm', 'normed', r, 'mhc-fused-layer-input', None))
            pl.append(('L0.out_after_tp', 0, 'output_after_tp', 'output', r, 'post-tp-output', None))
        pl.append((f'L{L}.dense_local', L, 'local_output', 'output', r, 'dense-local-down', None))
    pl.append(('L3.attn_input', 3, 'attn_pre', 'normed', ra, 'mhc-fused-layer-input', None))
    pl.append(('L3.ffn_input', 3, 'ffn_pre', 'normed', rf, 'mhc-fused-layer-input', None))
    pl.append(('L3.ffn_input_check', 3, 'input', None, r, 'input', None))
    pl.append(('L3.attn_block_out', 3, 'attn_post', 'block_output', r, 'attn-output', None))
    pl.append(('L3.attn_core_out', 3, 'proj_wo', 'input', r, 'attn-o_proj-input', None))
    pl.append(('L3.attn_core_in', 3, 'proj_wq_a', 'input', r, 'attn-input-hidden_states', None))
    pl.append(('L3.router_ids', 3, 'router', 'ids', r, 'route-ids', None))
    pl.append(('L3.router_weights', 3, 'router', 'weights', r, 'route-weights', None))
    pl.append(('L3.moE_out', 3, 'output_after_tp', 'output', r, 'post-tp-output', None))
    return pl


def analyze(root, output):
    QD = pathlib.Path(root)
    report = {'pairs': {}, 'chaos_floor': {}, 'notes': [], 'layout_probes': {}}
    plan_path = QD / 'extended-plan.json'
    plan = json.loads(plan_path.read_text())
    runs = {}
    manifests = {}
    for engine in ('candidate', 'reference'):
        attempt = plan['naming'][f'{engine}_attempt']
        if pathlib.Path(attempt).name != attempt:
            raise ValueError('attempt name must be local')
        artifact = QD / 'driver' / attempt / attempt / 'artifacts/short16-q2diag'
        manifest, _ = load_artifact(artifact)
        runs[engine] = manifest
        manifests[engine] = hashlib.sha256((artifact / 'manifest.json').read_bytes()).hexdigest()
    validate_run_pair(runs['reference'], runs['candidate'],
                      ['cases', 'tokens_sha256', 'request_policy'])
    if len(runs['candidate']['captures']) != ROWS:
        raise ValueError('diagnostic requires exactly eight bound request rows')
    report['run_identity'] = {
        'plan_sha256': hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        'manifest_sha256': manifests,
        'provenance': {engine: run['provenance'] for engine, run in runs.items()},
        'request_rows': runs['candidate']['captures'],
        'binding_scope': 'stage invocation mapping from extended plan; archival run identity validated',
    }
    inv = {}
    for rank in (0, 1):
        inv[('cand', rank)] = load(QD / f'dumps/cand-rank{rank}')
        inv[('ref', rank)] = load(QD / f'dumps/ref-rank{rank}/diagnostic')

    inventory = {}
    for (eng, rank), table in inv.items():
        stages = {}
        for (layer, _iv), sd in table.items():
            stages.setdefault(f'L{layer}', set()).update(sd)
        inventory[f'{eng}-rank{rank}'] = {k: sorted(v) for k, v in stages.items()}
    report['stage_inventory'] = inventory

    # ---- gate-up layout probe (row 0): which concat order matches the reference? ----
    for rank in (0, 1):
        cand, ref = inv[('cand', rank)], inv[('ref', rank)]
        cg = {n: a for n, a in tensors(cand[(0, 0)]['gate_up'])}
        rg = tensors(ref[(0, 0)]['gate-up'])[0][1]
        report['layout_probes'][f'rank{rank}.gate_up_order'] = {
            '[gate;up]': metrics(np.concatenate([cg['gate'], cg['up']], axis=1), rg),
            '[up;gate]': metrics(np.concatenate([cg['up'], cg['gate']], axis=1), rg)}

    # ---- pipeline pairs per request r ----

    for rank in (0, 1):
        cand, ref = inv[('cand', rank)], inv[('ref', rank)]
        for r in range(ROWS):
            for (pname, cl, cstage, ctensor, rinv, rstage, mode) in pair_list(r):
                key = f'rank{rank}.row{r}.{pname}'
                cside = cand.get((cl, r), {}).get(cstage)
                rside = ref.get((cl, rinv), {}).get(rstage)
                if cside is None or rside is None:
                    report['pairs'][key] = {'missing': [
                        cstage if cside is None else None,
                        f'{rstage}@{rinv}' if rside is None else None]}
                    continue
                try:
                    if mode == 'cat':
                        ct = {n: a for n, a in tensors(cside)}
                        rt = tensors(rside)[0][1]
                        stacked = np.concatenate([ct['gate'], ct['up']], axis=1)
                        report['pairs'][key] = {'metrics': metrics(stacked, rt),
                                                'shape': list(stacked.shape)}
                    else:
                        ct = dict(tensors(cside))
                        c_arr = ct[ctensor] if ctensor else list(ct.values())[0]
                        r_arr = tensors(rside)[0][1]
                        if pname in ('L0.kda_gated_norm', 'L0.kda_core_out', 'L0.kda_g1', 'L0.kda_g2'):
                            r_arr = kda_layout(r_arr, c_arr.shape)
                        report['pairs'][key] = {'metrics': metrics(c_arr, r_arr),
                                                'shape': list(c_arr.shape)}
                except Exception as exc:
                    report['pairs'][key] = {'error': f'{type(exc).__name__}: {exc}'}

    # A reduced output requires BOTH ranks' immediate producer inputs.
    report['producer_gates'] = {}
    for rank in (0, 1):
        for row in range(ROWS):
            for layer in (0, 1, 2):
                def dense_inputs(rr):
                    c = tensors(inv[('cand', rr)][(layer, row)]['input'])[0][1]
                    r = tensors(inv[('ref', rr)][(layer, row)]['input'])[0][1]
                    return c, r
                key = f'rank{rank}.row{row}.L{layer}.gate_up'
                report['producer_gates'][key] = producer_gate(dense_inputs, (rank,))
            def projection_inputs(rr):
                c = dict(tensors(inv[('cand', rr)][(0, row)]['gated_norm']))['gated']
                r = tensors(inv[('ref', rr)][(0, row)]['kda-o-proj-input0'])[0][1]
                return c, r
            key = f'rank{rank}.row{row}.L0.kda_reduced'
            report['producer_gates'][key] = producer_gate(projection_inputs, (0, 1))
    report['notes'].append('No same-input attribution from dense_local or MoE output: '
                           'activation/expert intermediate producer inputs are not bound.')

    # ---- chaos floor: parity-aware ----
    for rank in (0, 1):
        ref = inv[('ref', rank)]
        for layer in (0, 1, 2, 3):
            fams = {'direct': (PROBES, False)}
            if layer >= 1:
                fams['mhc-attn'] = (tuple(2 * p for p in PROBES), True)
                fams['mhc-ffn'] = (tuple(2 * p + 1 for p in PROBES), True)
            else:
                fams['mhc'] = (PROBES, True)
            for fam, (invs, mhc_only) in fams.items():
                stages = set()
                for iv in invs:
                    stages |= set(ref.get((layer, iv), {}))
                for stage in sorted(stages):
                    if mhc_only != stage.startswith('mhc-'):
                        continue
                    worst = None
                    for a, b in ((invs[0], invs[1]), (invs[0], invs[2]), (invs[1], invs[2])):
                        sa = ref.get((layer, a), {}).get(stage)
                        sb = ref.get((layer, b), {}).get(stage)
                        if not sa or not sb:
                            continue
                        try:
                            ta, tb = tensors(sa), tensors(sb)
                            for (na, aa), (nb, ab) in zip(ta, tb):
                                mm = metrics(aa, ab)
                                score = mm.get('max', 0) if mm['kind'] == 'bf16' else \
                                    (0 if mm['exact'] else 1)
                                if worst is None or score > worst[0]:
                                    worst = (score, na, mm)
                        except Exception as exc:
                            worst = (10 ** 9, stage, {'error': str(exc)})
                    if worst:
                        report['chaos_floor'][f'rank{rank}.L{layer}.{stage}'] = \
                            {'tensor': worst[1], **worst[2]}

    evidence = {}
    for base in (QD / 'dumps').glob('*rank*'):
        for path in sorted(base.rglob('*')):
            if path.is_file() and path.suffix in ('.json', '.bin'):
                evidence[str(path.relative_to(QD))] = hashlib.sha256(path.read_bytes()).hexdigest()
    report['evidence_sha256'] = evidence
    report['analyzer_sha256'] = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
    report['scope'] = 'offline diagnostic; frozen Q2 gate unchanged; no causal qualification'
    report['analysis_complete'] = all('metrics' in v for v in report['pairs'].values())
    with pathlib.Path(output).open('x') as target:
        json.dump(report, target, indent=1, allow_nan=False)
        target.write('\n')
    if not report['analysis_complete']:
        raise ValueError('diagnostic pairs contain missing stages or errors; inspect saved report')

    # ---- compact verdict summary ----
    import collections
    agg = collections.defaultdict(lambda: collections.Counter())
    for k, v in report['pairs'].items():
        pname = k.split('.', 2)[2]
        st = 'missing' if 'missing' in v else 'error' if 'error' in v else \
            'exact' if v['metrics'].get('exact') else 'diverges'
        agg[pname][st] += 1
    print('layout probes:', json.dumps(report['layout_probes'])[:400])
    for pname, c in sorted(agg.items()):
        print(f'{pname:24s} {dict(c)}')
    chaos_bad = {k: v for k, v in report['chaos_floor'].items()
                 if (v.get('kind') == 'bf16' and not v.get('exact'))
                 or (v.get('kind') == 'f32' and not v.get('exact'))}
    print(f"chaos floor: {len(report['chaos_floor'])} stages; nonzero: {len(chaos_bad)}")
    for k, v in chaos_bad.items():
        print('  CHAOS', k, v)
    # representative per-pair numbers for the verdict (rank0)
    for pname in sorted(agg):
        rows = [v for k, v in report['pairs'].items()
                if k.startswith('rank0.') and k.split('.', 2)[2] == pname and 'metrics' in v]
        if not rows:
            continue
        ex = rows[0]['metrics']
        nonexact = [r['metrics'] for r in rows if not r['metrics'].get('exact')]
        print(f'DETAIL {pname}: rows_exact={len(rows)-len(nonexact)}/{len(rows)} '
              f'first={ex} first_divergent={nonexact[0] if nonexact else None}')
    # per-row pattern for layer-0 stages (which requests diverge)
    for pname in ('L0.ffn_input', 'L0.gate_up', 'L0.mhc_attn_normed', 'L0.kda_gated_norm',
                  'L0.kda_oproj_input', 'L0.kda_reduced', 'L0.kda_core_out',
                  'L0.ffn_mhc_normed', 'L0.out_after_tp'):
        pat = []
        for r in range(ROWS):
            v = report['pairs'].get(f'rank0.row{r}.{pname}', {})
            m = v.get('metrics')
            pat.append('EXACT' if m and m.get('exact') else
                       (f"p99={m.get('p99')}" if m else v.get('error', 'missing')[:12]))
        print(f'ROWS {pname}: ' + ' | '.join(pat))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=pathlib.Path, required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    analyze(args.root, args.output)
