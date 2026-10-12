# SPDX-License-Identifier: AGPL-3.0-only
"""Gated-normalization arithmetic ablation harness for the Atlas Q2 diagnostic.

Replays captured runtime gated-norm fixtures through the attested embedded PTX
and single-feature candidate-kernel variants (gated_norm_variants.VARIANTS),
comparing every run against the candidate target and the reference-engine
target to attribute a <=1-ULP cross-engine difference per arithmetic feature.
Fail-closed throughout: explicit evidence paths only, immutable output
directories, anchors and engagement controls evaluated after all evidence is
written.
"""
import argparse
import hashlib
import json
import re
import struct
from pathlib import Path

import numpy as np

import analyze_localization
import gated_norm_variants
import native_gated_norm_replay
import native_ptx
from analyze_localization import bf16_key, metrics
from gated_norm_variants import KERNEL_ENTRY, VARIANTS, derive, frozen_body
from native_gated_norm_replay import (KDA_CONTRACT, device_properties,
                                      perturbations, replay, validate_operands)
from native_ptx import load_module, sha, validate_binding

FROZEN_KDA_SHA256 = 'bf2149c381d6c4283930d2bc69f6c5ddb4d53f7755de954de9f9762d179af9bd'
CONTROL_RUNS = ('attested', 'v0-copy', 'v-all')
ANCHOR_RUNS = ('attested', 'v0-copy')
# Engagement-control names written by native_gated_norm_replay.perturbations().
REQUIRED_CONTROLS = ('input', 'gate', 'weight')
MAX_COORDINATES = 64
_ROW_DIR = re.compile(r'rank(\d+)-row(\d+)')
FIXTURE_SCHEMA = 'norm-replay-fixtures-v1'
# Every file a fixture.json must attest; reference-* operands bind cross-engine equality.
FIXTURE_ARRAYS = ('core', 'gate', 'weight', 'expected', 'reference-core',
                  'reference-gate', 'reference-weight', 'reference-expected')


def compare_bf16(output, target):
    """Compare BF16 bit patterns; ULP distances use the monotonic key ordering."""
    if output.ndim != 3 or output.shape != target.shape:
        raise ValueError('comparison requires matching [tokens, heads, dim] tensors')
    if output.dtype.str != '<u2' or target.dtype.str != '<u2':
        raise ValueError('contiguous little-endian uint16 BF16 bits required')
    summary = metrics(output, target)
    keys_output, keys_target = bf16_key(output), bf16_key(target)
    distance = np.abs(keys_output.astype(np.int64) - keys_target.astype(np.int64))
    flat = int(np.argmax(distance))
    unequal = np.argwhere(output != target)
    result = {'exact': bool(summary['exact']),
              'unequal_elements': int(unequal.shape[0]),
              'max_ulp': int(summary['max']),
              'signed_ulp_at_max': (int(keys_output.reshape(-1)[flat])
                                    - int(keys_target.reshape(-1)[flat])),
              'coordinates': [[int(index) for index in row]
                              for row in unequal[:MAX_COORDINATES]]}
    if unequal.shape[0] > MAX_COORDINATES:
        result['coordinates_truncated'] = True
    return result


def build_summary(rank, rows, harness_sha256):
    """Aggregate per-row run records across rows; interpretation is blocked
    unless every replay anchor passes and every required engagement control
    is present and engaged on every control run."""
    if not rows:
        raise ValueError('at least one row record required')
    run_order = ('attested',) + tuple(VARIANTS)
    aggregates = {}
    for name in run_order:
        records = []
        for row in rows:
            runs = row.get('runs')
            if not isinstance(runs, dict) or set(runs) != set(run_order):
                raise ValueError('each row must record exactly the attested run and every variant')
            records.append(runs[name])
        aggregates[name] = {
            'rows_exact_vs_candidate': sum(1 for r in records if r['vs_candidate']['exact']),
            'rows_exact_vs_reference': sum(1 for r in records if r['vs_reference']['exact']),
            'total_unequal_vs_reference': sum(r['vs_reference']['unequal_elements'] for r in records),
            'max_ulp_vs_reference': max(r['vs_reference']['max_ulp'] for r in records)}
    anchors = {name: all(row['runs'][name]['self_reproduction'] for row in rows)
               for name in ANCHOR_RUNS}
    controls_engaged = True
    summary_rows = []
    for row in rows:
        run_records = {}
        for name in run_order:
            record = row['runs'][name]
            summary_record = {'self_reproduction': record['self_reproduction'],
                              'vs_candidate': record['vs_candidate'],
                              'vs_reference': record['vs_reference']}
            if name in CONTROL_RUNS:
                controls = record.get('controls')
                summary_record['controls'] = controls
                controls_engaged = controls_engaged and all(
                    isinstance(controls, dict)
                    and isinstance(controls.get(control), dict)
                    and controls[control].get('engaged') is True
                    for control in REQUIRED_CONTROLS)
            run_records[name] = summary_record
        summary_rows.append({'row': row['row'], 'tokens': row['tokens'],
                             'runs': run_records})
    return {
        'schema': 'atlas-gated-norm-ablation-summary-v1',
        'rank': rank,
        'runs': aggregates,
        'rows': summary_rows,
        'interpretation_blocked': not (all(anchors.values()) and controls_engaged),
        'anchors': anchors,
        'harness_sha256': harness_sha256}


def enforce_summary_gate(summary, failures):
    """Fail the command whenever the saved summary blocks interpretation or
    any anchor/engagement-control check failed; keeps exit status and summary
    agreement even if the failures bookkeeping drifts."""
    if summary.get('interpretation_blocked') or failures:
        raise RuntimeError('ablation anchors or engagement controls failed: '
                           + ('; '.join(failures) or 'saved summary blocks interpretation'))


def validate_cli(fixtures, rank, out):
    if rank not in (0, 1):
        raise ValueError('rank must be 0 or 1')
    if not Path(fixtures).is_dir():
        raise ValueError('fixtures directory required')
    if Path(out).exists():
        raise ValueError('output directory must not exist')


def row_dirs(fixtures, rank):
    rows = []
    for path in Path(fixtures).glob('rank' + str(rank) + '-row*'):
        match = _ROW_DIR.fullmatch(path.name)
        if (match and int(match.group(1)) == rank and path.is_dir()
                and (path / 'fixture.json').is_file()):
            rows.append((int(match.group(2)), path))
    if not rows:
        raise ValueError('fixture rows with fixture.json required for rank ' + str(rank))
    return sorted(rows)


def _f32_bits(value):
    """Effective little-endian f32 bit pattern of a recorded positive epsilon."""
    if type(value) not in (int, float):
        raise ValueError('fixture epsilon must be numeric')
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError('fixture epsilon overflows f32') from error
    if not np.isfinite(number) or number <= 0:
        raise ValueError('fixture epsilon must be finite and positive')
    try:
        raw = struct.pack('<f', number)
    except OverflowError as error:
        raise ValueError('fixture epsilon overflows f32') from error
    bits, = struct.unpack('<I', raw)
    if bits == 0:
        raise ValueError('fixture epsilon underflows f32')
    return bits


def validate_fixture(row_dir, row, rank):
    """Fail closed on fixture schema, row identity, recorded file hashes,
    actual cross-engine operand equality and effective epsilon bits before
    any replay arithmetic or attribution."""
    fixture = json.loads((row_dir / 'fixture.json').read_text())
    if not isinstance(fixture, dict) or fixture.get('schema') != FIXTURE_SCHEMA:
        raise ValueError('fixture schema mismatch: ' + str(row_dir))
    if type(fixture.get('rank')) is not int or fixture['rank'] != rank:
        raise ValueError('fixture rank mismatch: ' + str(row_dir))
    if type(fixture.get('invocation')) is not int or fixture['invocation'] != row:
        raise ValueError('fixture invocation/row identity mismatch: ' + str(row_dir))
    recorded = fixture.get('fixtures')
    if not isinstance(recorded, dict) or set(recorded) != set(FIXTURE_ARRAYS):
        raise ValueError('fixture file manifest mismatch: ' + str(row_dir))
    arrays = {}
    for name in FIXTURE_ARRAYS:
        entry = recorded[name]
        if (not isinstance(entry, dict) or entry.get('file') != name + '.npy'
                or entry.get('dtype') != 'bf16-le'):
            raise ValueError('fixture file identity mismatch: ' + name + ': ' + str(row_dir))
        path = row_dir / entry['file']
        if sha(path) != entry.get('sha256'):
            raise ValueError('fixture file hash mismatch: ' + name + ': ' + str(row_dir))
        value = np.load(path, allow_pickle=False)
        if (not isinstance(entry.get('shape'), list) or value.dtype.str != '<u2'
                or list(value.shape) != entry['shape']):
            raise ValueError('fixture file geometry mismatch: ' + name + ': ' + str(row_dir))
        arrays[name] = value
    if type(fixture.get('tokens')) is not int or fixture['tokens'] != arrays['core'].shape[0]:
        raise ValueError('fixture token identity mismatch: ' + str(row_dir))
    operands = fixture.get('equal_operands')
    if not isinstance(operands, dict):
        raise ValueError('fixture operand equality record missing: ' + str(row_dir))
    for key in ('core', 'gate', 'weight'):
        if operands.get(key) is not True:
            raise ValueError('recorded cross-engine operand inequality: ' + key
                             + ': ' + str(row_dir))
        if not np.array_equal(arrays[key], arrays['reference-' + key]):
            raise ValueError('actual cross-engine operand mismatch: ' + key
                             + ': ' + str(row_dir))
    candidate_bits = _f32_bits(fixture.get('candidate_epsilon_f32'))
    reference_bits = _f32_bits(fixture.get('reference_epsilon'))
    if (type(fixture.get('candidate_epsilon_f32_bits')) is not int
            or fixture['candidate_epsilon_f32_bits'] != candidate_bits
            or type(fixture.get('reference_epsilon_f32_bits')) is not int
            or fixture['reference_epsilon_f32_bits'] != reference_bits):
        raise ValueError('recorded epsilon bits mismatch: ' + str(row_dir))
    if candidate_bits != reference_bits or fixture.get('epsilon_f32_equal') is not True:
        raise ValueError('cross-engine epsilon bits differ: ' + str(row_dir))
    return fixture, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--rank', type=int, required=True)
    for name in ('common', 'ptx-dir', 'ptx-binding', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    validate_cli(args.fixtures, args.rank, args.out)
    rows = row_dirs(args.fixtures, args.rank)
    binding = validate_binding(args.ptx_dir, args.ptx_binding, args.common, KDA_CONTRACT)
    if not re.search(r'(?:^|@)sha256:[0-9a-f]{64}$', binding['image']):
        raise ValueError('immutable sha256 image identity required')
    ptx_path = args.ptx_dir / binding['modules']['kda']['file']
    signature = re.search(r'\.entry\s+' + KERNEL_ENTRY + r'\s*\((.*?)\)', ptx_path.read_text(), re.S)
    if not signature or re.findall(r'\.param\s+\.(\w+)', signature[1]) != (
            ['u64'] * 4 + ['u32', 'u32', 'f32']):
        raise ValueError('runtime gated norm argument types differ from production ABI')
    frozen_path = args.common / 'kda.cu'
    if sha(frozen_path) != FROZEN_KDA_SHA256:
        raise ValueError('frozen kda.cu differs from the Q2 ablation kernel source')
    frozen = frozen_body(frozen_path)
    derived = {name: derive(name, frozen_path) for name in VARIANTS}
    harness = {'gated_norm_ablation.py': sha(__file__),
               'gated_norm_variants.py': sha(Path(gated_norm_variants.__file__)),
               'native_gated_norm_replay.py': sha(Path(native_gated_norm_replay.__file__)),
               'native_ptx.py': sha(Path(native_ptx.__file__)),
               'analyze_localization.py': sha(Path(analyze_localization.__file__))}
    # Immutable output directories preserve failures as well as successes.
    args.out.mkdir(parents=True, exist_ok=False)
    import cupy as cp
    kernels = {'attested': load_module(cp, args.common, 'kda', args.ptx_dir, binding).get_function(
        KERNEL_ENTRY)}
    options = ('--std=c++17', '--fmad=false', '-I' + str(args.common))
    for name in VARIANTS:
        kernels[name] = cp.RawModule(code=derived[name][0], options=options).get_function(
            KERNEL_ENTRY)
    run_order = ('attested',) + tuple(VARIANTS)
    row_records, failures = [], []
    for row, row_dir in rows:
        fixture, arrays = validate_fixture(row_dir, row, args.rank)
        core, gate, weight = arrays['core'], arrays['gate'], arrays['weight']
        expected, reference = arrays['expected'], arrays['reference-expected']
        grid, block, epsilon = validate_operands(
            core, gate, weight, fixture['candidate_epsilon_f32'])
        for name, target in (('expected', expected), ('reference-expected', reference)):
            if target.shape != core.shape or target.dtype.str != '<u2':
                raise ValueError('captured BF16 target must match core geometry: ' + name)
        out_dir = args.out / ('rank' + str(args.rank) + '-row' + str(row))
        out_dir.mkdir()
        run_records = []
        for run in run_order:
            output = replay(cp, kernels[run], core, gate, weight, epsilon)
            np.save(out_dir / (run + '.npy'), output)
            record = {'run': run,
                      'kind': 'attested' if run == 'attested' else 'variant',
                      'output_file': run + '.npy',
                      'self_reproduction': output.tobytes() == expected.tobytes(),
                      'vs_candidate': compare_bf16(output, expected),
                      'vs_reference': compare_bf16(output, reference)}
            if run == 'attested':
                record['ptx_sha256'] = sha(ptx_path)
            else:
                record['variant_source_sha256'] = derived[run][2]
                record['variant_unified_diff'] = derived[run][1]
            if run in CONTROL_RUNS:
                record['controls'] = {}
                for name, operands in perturbations(core, gate, weight).items():
                    perturbed = replay(cp, kernels[run], *operands, epsilon)
                    engaged = bool(np.any(perturbed != output))
                    record['controls'][name] = {
                        'engaged': engaged,
                        'unequal_elements': int(np.count_nonzero(perturbed != output))}
                    if not engaged:
                        failures.append(out_dir.name + ': ' + run + ' control ' + name
                                        + ' did not engage')
            if run in ANCHOR_RUNS and not record['self_reproduction']:
                failures.append(out_dir.name + ': ' + run + ' self-reproduction anchor failed')
            run_records.append(record)
        receipt = {
            'schema': 'atlas-gated-norm-ablation-v1',
            'rank': args.rank, 'row': row, 'tokens': int(core.shape[0]),
            'epsilon_f32': float(epsilon),
            'epsilon_f32_bits': struct.unpack('<I', struct.pack('<f', float(epsilon)))[0],
            'grid': list(grid), 'block': list(block),
            'frozen_kda_sha256': sha(frozen_path),
            'frozen_body_sha256': hashlib.sha256(frozen.encode('utf-8')).hexdigest(),
            'inputs': {name: {'file': str(path), 'sha256': sha(path)} for name, path in
                       zip(('fixture.json', 'core.npy', 'gate.npy', 'weight.npy',
                            'expected.npy', 'reference-expected.npy'),
                           (row_dir / 'fixture.json', row_dir / 'core.npy', row_dir / 'gate.npy',
                            row_dir / 'weight.npy', row_dir / 'expected.npy',
                            row_dir / 'reference-expected.npy'))},
            'ptx_binding_sha256': sha(args.ptx_binding),
            'binding': binding,
            'runs': run_records,
            'device': device_properties(
                cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)),
            'runtime_version': cp.cuda.runtime.runtimeGetVersion(),
            'driver_version': cp.cuda.runtime.driverGetVersion(),
            'harness_sha256': harness}
        (out_dir / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        row_records.append({'row': row, 'tokens': int(core.shape[0]),
                            'runs': {record['run']: record for record in run_records}})
    summary = build_summary(args.rank, row_records, harness)
    (args.out / 'ablation-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    enforce_summary_gate(summary, failures)


if __name__ == '__main__':
    main()
