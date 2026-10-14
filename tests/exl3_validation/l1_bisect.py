#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""L1 escalation earliest-divergence bisection over paired engine captures (D3).

Compares candidate vs reference capture trees stage by stage in the parent
plan's bisection order (boundary streams -> L1 attention input -> L1 core ->
gated norm -> reduced o_proj -> FFN mHC input) and reports, per row and rank,
the earliest divergent stage. Diagnostic localization only; no gate claim.

Reuses analyze_localization's sidecar scan, metrics and KDA layout mapping
(metric SSOT, B3). Fail-closed: explicit layer/rank/rows only (layer within
the qualified {0,1} boundary), rank-bound sidecars, and a missing or errored
required pair fails the command AFTER the report is written, mirroring
analyze_localization's end-of-run refusal. The local (pre-reduce) o_proj has
no reference counterpart: the reference o_proj is a RowParallelLinear whose
reduce runs inside forward, so its hook output `kda-o-proj` is post-reduce
(spec D3 + assumption B13); C1 closed the local projection question from
native captures and C2's first measurement discharges its flag.
"""
import argparse
import json
import pathlib

from analyze_localization import kda_layout, load, metrics, tensors

SCHEMA = 'atlas-l1-bisect-report-v1'
CLAIM = 'diagnostic localization only; no gate claim'
POLICY_STAGE = 'mhc-pre-policy'

# (pair, candidate stage, candidate tensor, reference stage) in the
# bisection order of q2-norm-causality-plan.md "Fallback-suspect ordering".
REQUIRED_SPINE = (
    ('boundary_streams', 'incoming_highway', 'highway', 'mhc-pre-input-x'),
    ('boundary_streams_at_norm', 'attn_mhc_norm', 'highway', 'mhc-pre-input-x'),
    ('boundary_post', 'attn_mhc_norm', 'post', 'mhc-pre-post'),
    ('boundary_comb', 'attn_mhc_norm', 'comb', 'mhc-pre-comb'),
    ('attn_input', 'attn_mhc_norm', 'normed', 'mhc-pre-layer-input'),
    ('l1_core', 'recurrent_output', 'core', 'kda-core-output'),
    ('gated_norm', 'gated_norm', 'gated', 'kda-gated-norm'),
    ('reduced_oproj', 'attention_reduced_output', 'output', 'kda-o-proj'),
    ('ffn_mhc_input', 'ffn_mhc_norm', 'normed', 'mhc-fused-layer-input'),
)
DIAGNOSTIC_PAIRS = (
    ('oproj_input_identity', 'gated_norm', 'gated', 'kda-o-proj-input0'),
)


def _select(side, tensor_name):
    rows = tensors(side)
    if tensor_name is None:
        if len(rows) != 1:
            raise ValueError('single-tensor capture required')
        return rows[0][1]
    for name, array in rows:
        if name == tensor_name:
            return array
    raise ValueError('capture tensor not found: ' + tensor_name)


def _pair(sides, entry, layer, row):
    name, cstage, ctensor, rstage = entry
    candidate = sides['candidate'].get((layer, row), {}).get(cstage)
    reference = sides['reference'].get((layer, row), {}).get(rstage)
    if candidate is None or reference is None:
        return {'missing': [cstage if candidate is None else None,
                            rstage if reference is None else None]}
    try:
        c_arr = _select(candidate, ctensor)
        r_arr = kda_layout(_select(reference, None), c_arr.shape)
        if c_arr.dtype != r_arr.dtype:
            raise ValueError('capture dtype differs: %s vs %s'
                             % (c_arr.dtype.str, r_arr.dtype.str))
        return {'metrics': metrics(c_arr, r_arr)}
    except Exception as exc:  # recorded, never silently dropped
        return {'error': type(exc).__name__ + ': ' + str(exc)}


def _rank_checked(root, rank):
    stages = load(pathlib.Path(root))
    for (layer, invocation), entry in stages.items():
        for stage, (metadata, _) in entry.items():
            if metadata.get('rank') != rank or metadata.get('layer') != layer \
                    or metadata.get('invocation') != invocation:
                raise ValueError('sidecar identity mismatch: %s L%s inv%s %s'
                                 % (root, layer, invocation, stage))
    return stages


def _policy(root, layer, rank, row):
    path = pathlib.Path(root) / ('ffn-L%d-rank%d-%06d-%s.json'
                                % (layer, rank, row, POLICY_STAGE))
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def bisect(candidate_root, reference_root, layer, rank, rows):
    """Return the L1 bisection report (never partial-silent: missing or
    errored required pairs are recorded and `analysis_complete` is false).
    Raises only on sidecar identity mismatch, where no report is valid."""
    sides = {'candidate': _rank_checked(candidate_root, rank),
             'reference': _rank_checked(reference_root, rank)}
    report_rows, incomplete = [], []
    for row in rows:
        pairs = {}
        for entry in REQUIRED_SPINE + DIAGNOSTIC_PAIRS:
            pairs[entry[0]] = _pair(sides, entry, layer, row)
        earliest = None
        for name, _, _, _ in REQUIRED_SPINE:
            record = pairs[name]
            if 'missing' in record or 'error' in record:
                incomplete.append('row%d.%s' % (row, name))
            elif not record['metrics']['exact'] and earliest is None:
                earliest = name
        report_rows.append({
            'row': row,
            'pairs': pairs,
            'earliest_divergent_stage': earliest,
            'reference_policy': _policy(reference_root, layer, rank, row),
        })
    stages = {r['earliest_divergent_stage'] for r in report_rows}
    report = {
        'schema': SCHEMA,
        'claim': CLAIM,
        'rank': rank,
        'layer': layer,
        'rows': report_rows,
        'earliest_divergent_stage_by_row': {
            str(r['row']): r['earliest_divergent_stage'] for r in report_rows},
        'divergence_stages_mixed': len(stages - {None}) > 1,
        'incomplete_pairs': incomplete,
        'analysis_complete': not incomplete,
        'notes': [
            'local (pre-reduce) o_proj has no reference counterpart and is '
            'not bisectable: the reference o_proj is a RowParallelLinear and '
            'its hook output kda-o-proj is post-reduce (spec D3, assumption '
            'B13); a pre-reduce reference capture needs an owed overlay '
            'extension. C1 closed the local projection question from native '
            'captures',
        ],
    }
    return report


def validate_cli(candidate, reference, rank, layer, out):
    if not pathlib.Path(candidate).is_dir() or not pathlib.Path(reference).is_dir():
        raise ValueError('candidate and reference capture directories required')
    if rank not in (0, 1):
        raise ValueError('rank must be 0 or 1')
    if layer not in (0, 1):
        raise ValueError('layer must be within the qualified L0->L1 boundary')
    if pathlib.Path(out).exists():
        raise ValueError('output path must not exist')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--layer', type=int, required=True)
    parser.add_argument('--rows', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    rows = [int(row) for row in args.rows.split(',') if row != '']
    if not rows or len(set(rows)) != len(rows):
        raise ValueError('explicit unique row list required')
    validate_cli(args.candidate, args.reference, args.rank, args.layer, args.out)
    report = bisect(args.candidate, args.reference, args.layer, args.rank, rows)
    with open(args.out, 'x') as handle:
        handle.write(json.dumps(report, indent=2) + '\n')
    for row in report['rows']:
        print('row%d: earliest divergence %s' % (row['row'], row['earliest_divergent_stage']))
    print('analysis complete: %s' % report['analysis_complete'])
    if not report['analysis_complete']:
        raise ValueError('bisection pairs missing or errored: '
                         + '; '.join(report['incomplete_pairs'])
                         + '; inspect saved report')


if __name__ == '__main__':
    main()
