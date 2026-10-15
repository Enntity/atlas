#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Source-bound L0/L1 localization over fresh paired capture and collection trees.

Ordinary L1 uses fused attention mHC invocation 2r and FFN invocation 2r+1.
Its materialized incoming highway corresponds to the fused residual OUTPUT,
not to the deferred input branch. Diagnostic localization only; no gate claim.
"""
import argparse
import json
from pathlib import Path

from analyze_localization import kda_layout, load, metrics, tensors
from l1_binding import (candidate_boundary, policy_record, reference_context,
                        registered_requests, require, sha)

SCHEMA = 'atlas-l1-bisect-report-v2'
CLAIM = 'diagnostic localization only; no gate claim'


def _spine(layer, row):
    kind = 'fused' if layer == 1 else 'pre'
    attention = 2 * row if layer == 1 else row
    ffn = 2 * row + 1 if layer == 1 else row
    prefix = 'mhc-' + kind
    highway = prefix + ('-residual-output' if layer == 1 else '-input-x')
    return (
        ('boundary_streams', 'incoming_highway', 'highway', highway, attention),
        ('boundary_streams_at_norm', 'attn_mhc_norm', 'highway', highway, attention),
        ('boundary_post', 'attn_mhc_norm', 'post', prefix + '-post', attention),
        ('boundary_comb', 'attn_mhc_norm', 'comb', prefix + '-comb', attention),
        ('attn_input', 'attn_mhc_norm', 'normed', prefix + '-layer-input', attention),
        ('l1_core', 'recurrent_output', 'core', 'kda-core-output', row),
        ('gated_norm', 'gated_norm', 'gated', 'kda-gated-norm', row),
        ('reduced_oproj', 'attention_reduced_output', 'output', 'kda-o-proj', row),
        ('ffn_mhc_input', 'ffn_mhc_norm', 'normed', 'mhc-fused-layer-input', ffn),
    )


def _select(side, tensor_name):
    values = tensors(side)
    if tensor_name is None:
        require(len(values) == 1, 'single-tensor capture required')
        return values[0][1]
    for name, array in values:
        if name == tensor_name:
            return array
    raise ValueError('capture tensor not found: ' + tensor_name)


def _rank_checked(root, rank, engine):
    stages = load(Path(root))
    for (layer, invocation), entries in stages.items():
        for stage, (metadata, path) in entries.items():
            require(path.is_file() and not path.is_symlink(), 'regular sidecar required')
            owner = metadata.get('tp_rank') if engine == 'candidate' else metadata.get('rank')
            valid = type(owner) is int and owner == rank \
                and type(metadata.get('layer')) is int and metadata['layer'] == layer \
                and type(metadata.get('invocation')) is int and metadata['invocation'] == invocation
            if engine == 'candidate':
                valid = valid and metadata.get('schema') == 'atlas-exl3-ffn-v1' \
                    and type(metadata.get('tp_world')) is int and metadata['tp_world'] == 2
                if 'rank' in metadata:
                    valid = valid and metadata['rank'] == owner
            require(valid, 'sidecar identity mismatch: %s L%s inv%s %s' % (root, layer, invocation, stage))
    return stages


def _capture_hashes(sides, layer, rank, rows, reference_root):
    """Fingerprint only the selected spine/diagnostic tensors, never resident weights."""
    paths, missing = {}, []
    selected = set()
    for row in rows:
        entries = _spine(layer, row) + (('oproj_input_identity', 'gated_norm', 'gated', 'kda-o-proj-input0', row),)
        for _, cstage, _, rstage, invoke in entries:
            for engine, index, stage in [('candidate', row, cstage), ('reference', invoke, rstage)]:
                side = sides[engine].get((layer, index), {}).get(stage)
                if side is not None:
                    selected.add(side[1])
        for kind, invoke in [('fused' if layer else 'pre', 2 * row if layer else row),
                             ('fused', 2 * row + 1 if layer else row)]:
            path = Path(reference_root) / ('ffn-L%d-rank%d-%06d-mhc-%s-policy.json'
                                          % (layer, rank, invoke, kind))
            if path.is_file() and not path.is_symlink():
                paths[str(path)] = sha(path)
    for path in sorted(selected):
        paths[str(path)] = sha(path)
        metadata = json.loads(path.read_text())
        for tensor in metadata.get('tensors', [metadata]):
            filename = tensor.get('file')
            if not isinstance(filename, str) or Path(filename).name != filename:
                missing.append(str(path) + ': invalid payload filename')
                continue
            payload = path.parent / filename
            if payload.is_file() and not payload.is_symlink():
                paths[str(payload)] = sha(payload)
            else:
                missing.append(str(payload))
    return paths, missing


def _pair(sides, entry, layer, row, request, source):
    _, cstage, ctensor, rstage, rinvoke = entry
    candidate = sides['candidate'].get((layer, row), {}).get(cstage)
    reference = sides['reference'].get((layer, rinvoke), {}).get(rstage)
    record = {'candidate_stage': cstage, 'candidate_invocation': row,
              'reference_stage': rstage, 'reference_invocation': rinvoke}
    if candidate is None or reference is None:
        return dict(record, missing=[cstage if candidate is None else None,
                                     rstage if reference is None else None])
    try:
        reference_context(reference[0], request, source)
        c_arr = _select(candidate, ctensor)
        r_arr = kda_layout(_select(reference, None), c_arr.shape)
        require(c_arr.dtype == r_arr.dtype, 'capture dtype differs: %s vs %s' % (c_arr.dtype.str, r_arr.dtype.str))
        record['metrics'] = metrics(c_arr, r_arr)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        record['error'] = type(exc).__name__ + ': ' + str(exc)
    return record


def _policies(sides, reference_root, layer, rank, row, request, source):
    records, errors = {}, []
    for phase, stage, kind, invocation in (
        ('attention', 'attn_mhc_norm', 'fused' if layer == 1 else 'pre', 2 * row if layer == 1 else row),
        ('ffn', 'ffn_mhc_norm', 'fused', 2 * row + 1 if layer == 1 else row),
    ):
        records[phase] = None
        try:
            policy = policy_record(reference_root, kind, invocation, layer, rank, request['reference'], source)
            records[phase] = policy
            side = sides['candidate'].get((layer, row), {}).get(stage)
            require(side is not None, 'candidate boundary record missing: ' + stage)
            shape = _select(side, 'normed').shape
            binding = candidate_boundary(side[0], shape, request['candidate'], policy['policy'])
            highway_shape = _select(side, 'highway').shape
            require(len(highway_shape) == 3 and highway_shape[:2] == (shape[0], binding['hc_mult']),
                    'candidate highway stream shape differs from policy')
        except (KeyError, TypeError, ValueError, OSError) as exc:
            errors.append(phase + ': ' + type(exc).__name__ + ': ' + str(exc))
    return records, errors


def bisect(candidate_root, reference_root, layer, rank, rows, *, candidate_artifact,
           reference_artifact, binding_path):
    require(layer in (0, 1) and rank in (0, 1), 'explicit L0/L1 and TP rank required')
    require(rows and len(set(rows)) == len(rows) and all(type(row) is int and row >= 0 for row in rows),
            'explicit unique nonnegative row list required')
    requests, source, inputs = registered_requests(candidate_artifact, reference_artifact, binding_path, reference_root)
    require(all(row < len(requests['candidate']) for row in rows), 'request row outside registered collection')
    sides = {'candidate': _rank_checked(candidate_root, rank, 'candidate'),
             'reference': _rank_checked(reference_root, rank, 'reference')}
    report_rows, incomplete = [], []
    for row in rows:
        request = {engine: requests[engine][row] for engine in requests}
        policies, binding_errors = _policies(sides, reference_root, layer, rank, row, request, source)
        spine = _spine(layer, row)
        diagnostic = ('oproj_input_identity', 'gated_norm', 'gated', 'kda-o-proj-input0', row)
        pairs = {entry[0]: _pair(sides, entry, layer, row, request['reference'], source)
                 for entry in spine + (diagnostic,)}
        earliest, prefix_complete = None, not binding_errors
        for name, *_ in spine:
            record = pairs[name]
            if 'missing' in record or 'error' in record:
                incomplete.append('row%d.%s' % (row, name))
                prefix_complete = False
            elif not record['metrics']['exact'] and earliest is None and prefix_complete:
                earliest = name
        incomplete.extend('row%d.policy.%s' % (row, error) for error in binding_errors)
        report_rows.append({'row': row, 'request': dict(request['candidate'], reference_request=request['reference']),
                            'pairs': pairs, 'earliest_divergent_stage': earliest,
                            'attribution_blocked': earliest is None and not prefix_complete,
                            'binding_errors': binding_errors, 'reference_policy': policies})
    stages = {row['earliest_divergent_stage'] for row in report_rows} - {None}
    inputs['capture_file_sha256'], inputs['unhashed_payloads'] = _capture_hashes(sides, layer, rank, rows, reference_root)
    return {'schema': SCHEMA, 'claim': CLAIM, 'layer': layer, 'rank': rank, 'inputs': inputs,
            'rows': report_rows, 'analysis_complete': not incomplete, 'incomplete_pairs': incomplete,
            'earliest_divergent_stage_by_row': {str(row['row']): row['earliest_divergent_stage'] for row in report_rows},
            'divergence_stages_mixed': len(stages) > 1,
            'notes': ['L1 fused attention uses 2r, fused FFN uses 2r+1; incoming highway is materialized residual-output.',
                      'Local pre-reduce o_proj has no reference counterpart (spec B13); reduced o_proj remains required.',
                      'Collection manifests and hash-bound head provenance bind exact reference GPU request IDs; candidate boundaries bind submitted IDs.']}


def validate_cli(candidate, reference, rank, layer, out):
    require(Path(candidate).is_dir() and Path(reference).is_dir(), 'candidate and reference capture directories required')
    require(rank in (0, 1), 'rank must be 0 or 1')
    require(layer in (0, 1), 'layer must be within the qualified L0->L1 boundary')
    require(not Path(out).exists(), 'output path must not exist')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate', 'reference', 'candidate-artifact', 'reference-artifact', 'binding', 'rows', 'out'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--rank', required=True, type=int)
    parser.add_argument('--layer', required=True, type=int)
    args = parser.parse_args()
    rows = [int(value) for value in args.rows.split(',')]
    validate_cli(args.candidate, args.reference, args.rank, args.layer, args.out)
    report = bisect(args.candidate, args.reference, args.layer, args.rank, rows,
                    candidate_artifact=args.candidate_artifact, reference_artifact=args.reference_artifact,
                    binding_path=args.binding)
    with open(args.out, 'x') as destination:
        destination.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    for row in report['rows']:
        print('row%d: earliest divergence %s' % (row['row'], row['earliest_divergent_stage']))
    if not report['analysis_complete']:
        raise ValueError('bisection pairs or bindings missing or errored: ' + '; '.join(report['incomplete_pairs']))


if __name__ == '__main__':
    main()
