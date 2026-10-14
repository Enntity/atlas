# SPDX-License-Identifier: AGPL-3.0-only
"""C2 diagnostic scoring: the plan's fixed rules, implemented literally (D3).

Authority: docs/plans/q2-norm-causality-plan.md §"Experiment C2", "Diagnostic
scoring fixed before execution". The metric SSOT is
tests/exl3_validation/statistical_gate.py::row_metrics and its frozen-limit
envelope (statistical_gate.catastrophic) (docs/plans/c2-tooling-spec.md A4); the ref04-ref12
ensemble members and frozen limits are supplied inputs, never tooling choices
(A5). Every row of every supplied run is scored: there is no knob that drops a
reference, row or vocab subset. The five-token fingerprint is explanatory only
and can never change a verdict. This report never claims a Q2 gate pass.
"""
import argparse
import datetime
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from statistical_gate import (CATASTROPHE_FACTOR, FLOORS, catastrophic,
                              distance_table, load_run, row_metrics,
                              validate_runs)

SCHEMA = 'atlas-c2-diagnostic-score-v1'
FINGERPRINT_TOKENS = (2619, 8062, 29296, 220, 576)
KNOB_ENV = 'ATLAS_EXL3_NORM_SUBSTITUTE_DIR'
RECORD_ENV = 'ATLAS_EXL3_NORM_SUBSTITUTE_RECORD_DIR'
IDENTITY = ('cases', 'tokens_sha256', 'vocab', 'request_policy')
SIGNS = ('closer', 'farther', 'no-metric-change')
OUTPUT_ENV = ('ATLAS_EXL3_LOGITS_DIR', 'ATLAS_EXL3_UPSTREAM_DUMP_DIR',
              'ATLAS_EXL3_FFN_DUMP_DIR')


def _diagnostic_outputs(manifest, arm):
    """Bind fresh observer destinations to driver roles and recorded captures.

    This exempts only the three output destinations, not capture selectors or
    numerical policy. Paths are checked lexically without filesystem I/O.
    """
    environment = manifest['provenance']['environment']
    if not any(key in environment for key in OUTPUT_ENV):
        return {}
    paths = {}
    for key in OUTPUT_ENV:
        value = environment.get(key)
        if (not isinstance(value, str) or not value or not Path(value).is_absolute()
                or '..' in Path(value).parts or str(Path(value)) != value):
            raise ValueError('diagnostic output paths must be explicit canonical absolute strings')
        paths[key] = Path(value)
    logits, upstream, ffn = (paths[key] for key in OUTPUT_ENV)
    run_root = logits.parent
    if (logits.name != 'logits' or run_root.parent.name != 'runs'
            or not re.fullmatch(r'arm-' + arm + r'-repeat-[0-9]{2,}', run_root.name)
            or ffn != run_root / 'ffn'
            or upstream != run_root.parent.parent / 'gated_norm' / run_root.name):
        raise ValueError('diagnostic output paths do not match their driver roles')
    captures = manifest.get('captures')
    expected = sum(len(case['continuation_ids']) + 1 for case in manifest['cases'])
    if not isinstance(captures, list) or len(captures) != expected:
        raise ValueError('diagnostic output requires every recorded logits capture')
    for index, capture in enumerate(captures):
        if (not isinstance(capture, dict)
                or capture.get('capture') != str(logits / f'logits-{index:06d}.json')):
            raise ValueError('diagnostic output does not bind recorded logits captures')
    return {key: str(path) for key, path in paths.items()}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_arm_identity(arm_a, arm_b):
    """One-variable A/B: the arms must differ only in the substitution knob
    environment. Any other configuration difference refuses to score."""
    a, b = arm_a['manifests'][0], arm_b['manifests'][0]
    for field in IDENTITY:
        if a[field] != b[field]:
            raise ValueError(f'arm configurations differ in {field}')
    for field in ('serve_flags', 'engine_build', 'checkpoint_sha256', 'checkpoint_revision',
                  'tokenizer_sha256', 'config_sha256', 'execution'):
        if a['provenance'][field] != b['provenance'][field]:
            raise ValueError(f'arm configurations differ in provenance {field}')
    env_a, env_b = a['provenance']['environment'], b['provenance']['environment']
    if not isinstance(env_a, dict) or not isinstance(env_b, dict):
        raise ValueError('arm provenance environments must be explicit mappings')
    outputs_a = _diagnostic_outputs(a, 'a')
    outputs_b = _diagnostic_outputs(b, 'b')
    if bool(outputs_a) != bool(outputs_b):
        raise ValueError('diagnostic output paths must be bound on both arms')
    if outputs_a and (Path(outputs_a[OUTPUT_ENV[0]]).parents[2]
                      != Path(outputs_b[OUTPUT_ENV[0]]).parents[2]):
        raise ValueError('diagnostic output paths must belong to the same A/B attempt')
    knob_keys = {KNOB_ENV, RECORD_ENV, *outputs_a}
    stripped_a = {key: value for key, value in env_a.items() if key not in knob_keys}
    stripped_b = {key: value for key, value in env_b.items() if key not in knob_keys}
    if stripped_a != stripped_b:
        raise ValueError('mixed configurations between arms: environments differ beyond the knob')
    return {'arm_a_environment': env_a, 'arm_b_environment': env_b,
            'diagnostic_outputs': {'arm_a': outputs_a, 'arm_b': outputs_b}}


def _knob_state(environment):
    engaged = bool(environment.get(KNOB_ENV))
    return {'substitute_dir': environment.get(KNOB_ENV),
            'record_dir': environment.get(RECORD_ENV),
            'engaged': engaged}


def _mean(values):
    return float(np.mean(values)) if values else None


def _arm_row(run, references, key, row, case):
    """Per-row arm metrics against every supplied reference (row_metrics SSOT)."""
    per_reference = []
    for name, ref in zip(references[0], references[1]):
        metrics = row_metrics(run['rows'][key], ref['rows'][key], case)
        ce = metrics['ce_error'][row]
        per_reference.append({'reference': name,
                              'sym_kl': float(metrics['sym_kl'][row]),
                              'ce_error': float(ce) if np.isfinite(ce) else None,
                              'topk_miss': float(metrics['topk_miss'][row])})
    sym = [item['sym_kl'] for item in per_reference]
    ce_values = [item['ce_error'] for item in per_reference if item['ce_error'] is not None]
    topk = [item['topk_miss'] for item in per_reference]
    nearest = min(per_reference, key=lambda item: item['sym_kl'])
    return {'sym_kl_mean': _mean(sym),
            'sym_kl_nearest': nearest['sym_kl'],
            'sym_kl_nearest_reference': nearest['reference'],
            'ce_error_mean': _mean(ce_values),
            'topk_miss_mean': _mean(topk),
            'per_reference': per_reference}


def _catastrophic_rows(failures):
    rows = {}
    for failure in failures:
        rows.setdefault((tuple(failure['case']), failure['row']), {})[failure['metric']] = {
            'catastrophic': True, 'nearest': failure['nearest'], 'limit': failure['limit']}
    return rows


def _row_catastrophic(row_failures):
    return {metric: row_failures.get(metric, {'catastrophic': False}) for metric in FLOORS}


def _raw_bytes(run, key, row):
    return run['rows'][key][row].astype('<f4').tobytes()


def _fingerprint(arm_a, arm_b, references, key, row, vocab):
    runs = [('arm_a', arm_a), ('arm_b', arm_b)] + list(zip(references[0], references[1]))
    present = [token for token in FINGERPRINT_TOKENS if token < vocab]
    rows = {}
    for name, run in runs:
        values = {str(token): float(run['rows'][key][row][token]) for token in present}
        rows[name] = {'order': sorted(values, key=lambda token: -values[token]),
                      'logits': values}
    return rows


def score(reference_specs, reference_runs, arm_a, arm_b,
          arm_a_spec='arm-a', arm_b_spec='arm-b'):
    """Apply the fixed rules to one A run and one B run against every reference."""
    references = (list(reference_specs), list(reference_runs))
    validate_runs(references[1], [arm_a])
    validate_runs(references[1], [arm_b])
    configuration = validate_arm_identity(arm_a, arm_b)
    runs = references[1] + [arm_a, arm_b]
    table = distance_table(runs)
    ref_indices = list(range(len(references[1])))
    failures = {'arm_a': _catastrophic_rows(catastrophic(len(ref_indices), ref_indices, table)),
                'arm_b': _catastrophic_rows(
                    catastrophic(len(ref_indices) + 1, ref_indices, table))}
    vocab = arm_a['manifests'][0]['vocab']
    rows, signs, byte_equal_count = [], [], 0
    fingerprint_rows = []
    for key in sorted(arm_a['cases']):
        case = arm_a['cases'][key]
        for row in range(len(case['continuation_ids']) + 1):
            entry = {'suite': key[0], 'case': key[1], 'row': row}
            entry['arm_a'] = _arm_row(arm_a, references, key, row, case)
            entry['arm_b'] = _arm_row(arm_b, references, key, row, case)
            delta = entry['arm_b']['sym_kl_mean'] - entry['arm_a']['sym_kl_mean']
            sign = ('closer' if delta < 0 else 'farther' if delta > 0 else 'no-metric-change')
            signs.append(sign)
            entry['primary_delta_sym_kl'] = delta
            entry['delta_sign'] = sign
            raw_a, raw_b = _raw_bytes(arm_a, key, row), _raw_bytes(arm_b, key, row)
            equal = raw_a == raw_b
            byte_equal_count += equal
            entry['raw_logit_bytes'] = {
                'arm_a_sha256': hashlib.sha256(raw_a).hexdigest(),
                'arm_b_sha256': hashlib.sha256(raw_b).hexdigest(),
                'a_b_byte_equal': equal}
            entry['catastrophic'] = {
                'arm_a': _row_catastrophic(failures['arm_a'].get((key, row), {})),
                'arm_b': _row_catastrophic(failures['arm_b'].get((key, row), {}))}
            rows.append(entry)
            fingerprint_rows.append({'suite': key[0], 'case': key[1], 'row': row,
                                     'runs': _fingerprint(arm_a, arm_b, references,
                                                         key, row, vocab)})
    distinct = sorted(set(signs))
    report = {
        'schema': SCHEMA,
        'scoring_rules': {
            'authority': 'docs/plans/q2-norm-causality-plan.md#experiment-c2',
            'metric_source': 'tests/exl3_validation/statistical_gate.py::row_metrics',
            'catastrophic_source': 'tests/exl3_validation/statistical_gate.py::catastrophic',
            'primary_row_delta': 'mean_sym_kl(B, refs) - mean_sym_kl(A, refs); negative = closer',
            'catastrophic_limits': {'sym_kl': FLOORS['sym_kl'], 'ce_error': FLOORS['ce_error'],
                                    'catastrophe_factor': CATASTROPHE_FACTOR},
            'aggregation': {'sym_kl': 'mean and nearest over every supplied reference',
                            'ce_error': 'mean over every supplied reference where defined',
                            'topk_miss': 'mean over every supplied reference'},
            'subset_selection': 'none: every supplied reference, row and vocab entry is scored'},
        'inputs': {'references': references[0], 'arm_a': arm_a_spec,
                   'arm_b': arm_b_spec, 'reference_count': len(references[1])},
        'rows': rows,
        'row_delta_summary': {sign: signs.count(sign) for sign in SIGNS},
        'row_sign_result': 'mixed' if len(distinct) > 1 else distinct[0],
        'raw_logit_byte_equality': {
            'rows_total': len(rows), 'rows_byte_equal': byte_equal_count,
            'all_rows_equal': byte_equal_count == len(rows)},
        'five_token_fingerprint': {
            'role': 'explanatory-only', 'verdict_influence': 'none',
            'tokens': list(FINGERPRINT_TOKENS),
            'missing_tokens': [token for token in FINGERPRINT_TOKENS if token >= vocab],
            'rows': fingerprint_rows},
        'arm_configuration': {
            'diagnostic_outputs': configuration['diagnostic_outputs'],
            'knob': {'arm_a': _knob_state(configuration['arm_a_environment']),
                     'arm_b': _knob_state(configuration['arm_b_environment'])}},
        'harness_sha256': {'c2_score.py': _sha(__file__),
                           'statistical_gate.py': _sha(Path(__file__).with_name('statistical_gate.py'))},
        'evaluated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'q2_gate': {'pass_claimed': False,
                    'statement': 'diagnostic slice only; this report never claims a Q2 gate pass'}}
    knob = report['arm_configuration']['knob']
    if knob['arm_a']['engaged'] or not knob['arm_b']['engaged']:
        report['arm_configuration']['attribution'] = {
            'status': 'unresolved',
            'reason': 'knob engagement mismatch: arm A must be off and arm B engaged'}
    else:
        report['arm_configuration']['attribution'] = {
            'status': 'one-variable-substitution-observed',
            'reason': 'B-A deltas are attributable to the substituted norm values only '
                      'once the driver receipt confirms byte-determinism'}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', action='append', required=True,
                        help='one reference run: comma-joined artifact dirs, one per suite')
    parser.add_argument('--arm-a', required=True, help='knob-off run (same suite joining)')
    parser.add_argument('--arm-b', required=True, help='knob-on run (same suite joining)')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if Path(args.out).exists():
        raise ValueError('report destination must be new')
    references = [load_run(spec) for spec in args.reference]
    report = score(args.reference, references, load_run(args.arm_a), load_run(args.arm_b),
                   args.arm_a, args.arm_b)
    Path(args.out).write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({'row_sign_result': report['row_sign_result'],
                      'rows': len(report['rows']), 'report': str(args.out)}))


if __name__ == '__main__':
    main()
