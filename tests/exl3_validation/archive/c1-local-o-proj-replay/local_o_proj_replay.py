# SPDX-License-Identifier: AGPL-3.0-only
# ARCHIVED 2026-10-12 — DO NOT USE. The C1 BF16 common-space replay premise
# is invalid (see README.md in this directory). Importing or running this
# module fails unless ATLAS_ALLOW_ARCHIVED_C1=1 is set for forensic reading.
import os as _os
if _os.environ.get("ATLAS_ALLOW_ARCHIVED_C1") != "1":
    raise ImportError("archived C1 o_proj replay harness: premise invalid; see README.md")
"""Experiment C1: local o_proj replay before TP for the Atlas Q2 diagnostic.

Replays each engine's layer-0 local (pre-TP) output projection from bound
resident weights over three bound gated-norm input variants
(candidate-original, reference-exact, v24-corrected). Each engine must first
reproduce its OWN captured local projection output byte-exactly (self-anchor)
before any variant replay runs; cross-engine comparison happens only at
byte-identical operands. Emits the per-element propagation matrix (input
delta -> output delta with coordinates and signed BF16 ULP ordering via
gated_norm_ablation.compare_bf16) plus per-row binding receipts.

Fail-closed throughout: fixtures are validated through
gated_norm_ablation.validate_fixture, binding rows through validate_binding
(schema/rank/layer/invocation/engine/token identity, projection config, file
hashes, the additive resident sub-record, geometry, operand equality), immutable output directories, and the
summary gate refuses interpretation on any failed anchor or binding check.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import gated_norm_ablation
from gated_norm_ablation import compare_bf16, row_dirs, validate_fixture
from native_ptx import sha

BINDING_SCHEMA = 'atlas-local-o-proj-binding-v1'
MATRIX_SCHEMA = 'atlas-local-o-proj-propagation-matrix-v1'
ENGINES = ('candidate', 'reference')
VARIANTS = ('candidate-original', 'reference-exact', 'v24-corrected')
# Each engine anchors against its own captured gated-norm output first.
OWN_ORIGINAL = {'candidate': 'candidate-original', 'reference': 'reference-exact'}
BINDING_ARRAYS = ('weight', 'own-output', 'candidate-original',
                  'reference-exact', 'v24-corrected')
WEIGHT_LAYOUTS = ('out-in', 'in-out')
PROJECTION_DTYPES = {'input_dtype': 'bf16', 'weight_dtype': 'bf16',
                     'accumulator_dtype': 'f32', 'output_dtype': 'bf16'}


def compare_2d(output, target):
    """Per-element BF16 comparison of [tokens, features] projection tensors;
    coordinates are reported as [token, 0, feature]."""
    if (output.ndim != 2 or target.ndim != 2 or output.shape != target.shape
            or output.dtype.str != '<u2' or target.dtype.str != '<u2'):
        raise ValueError('2-D BF16 comparison requires matching [tokens, features]')
    return compare_bf16(output.reshape(output.shape[0], 1, output.shape[1]),
                        target.reshape(target.shape[0], 1, target.shape[1]))


def bf16_to_f32(bits):
    if bits.dtype.str != '<u2' or not bits.flags.c_contiguous:
        raise ValueError('contiguous little-endian uint16 BF16 bits required')
    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)


def f32_to_bf16(values):
    """Round finite f32 values to BF16 bits, round-to-nearest-even."""
    if not np.isfinite(values).all():
        raise RuntimeError('projection produced nonfinite f32 output')
    words = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounded = words + np.uint32(0x7fff) + ((words >> np.uint32(16)) & np.uint32(1))
    bits = (rounded >> np.uint32(16)).astype(np.uint16)
    if np.any((bits & 0x7f80) == 0x7f80):
        raise RuntimeError('projection produced nonfinite BF16 output')
    return bits


def _check_geometry(layout, inp, weight):
    if layout not in WEIGHT_LAYOUTS:
        raise ValueError('unsupported weight_layout: ' + str(layout))
    if inp.ndim != 2 or weight.ndim != 2 or 0 in weight.shape:
        raise ValueError('projection requires nonempty 2-D input and weight')
    width = weight.shape[1] if layout == 'out-in' else weight.shape[0]
    if width != inp.shape[1]:
        raise ValueError('projection input width differs from weight geometry')
    return inp.shape[0], weight.shape[0] if layout == 'out-in' else weight.shape[1]


def project_numpy(layout, inp, weight):
    """CPU-verifiable projection: exact BF16->F32 operand upcast, F32 GEMM,
    round-to-nearest-even back to BF16 bits."""
    _check_geometry(layout, inp, weight)
    x, w = bf16_to_f32(inp), bf16_to_f32(weight)
    return f32_to_bf16(x @ w.T if layout == 'out-in' else x @ w)


def project_cupy(layout, inp, weight):
    """GPU projection: same operand/bit contract as project_numpy with the
    F32 GEMM executed on the device."""
    import cupy as cp
    _check_geometry(layout, inp, weight)
    x, w = cp.asarray(bf16_to_f32(inp)), cp.asarray(bf16_to_f32(weight))
    output = x @ w.T if layout == 'out-in' else x @ w
    cp.cuda.get_current_stream().synchronize()
    return f32_to_bf16(cp.asnumpy(output))


def make_backend(name):
    if name == 'numpy':
        return project_numpy
    if name == 'cupy':
        return project_cupy
    raise ValueError('unsupported backend: ' + str(name))


def validate_binding(row_dir, rank, row, engine, arrays):
    """Fail closed on binding schema, row identity, projection config, file
    hashes, geometry and fixture operand equality before any replay."""
    row_dir = Path(row_dir)
    path = row_dir / 'binding.json'
    if not path.is_file():
        raise ValueError('binding record missing: ' + str(row_dir))
    binding = json.loads(path.read_text())
    if not isinstance(binding, dict) or binding.get('schema') != BINDING_SCHEMA:
        raise ValueError('binding schema mismatch: ' + str(row_dir))
    tokens = arrays['expected'].shape[0]
    for key, expected, label, kind in (
            ('rank', rank, 'rank', int), ('layer', 0, 'layer', int),
            ('invocation', row, 'invocation', int), ('tokens', tokens, 'token', int),
            ('engine', engine, 'engine', str)):
        if type(binding.get(key)) is not kind or binding[key] != expected:
            raise ValueError('binding ' + label + ' mismatch: ' + str(row_dir))
    projection = binding.get('projection')
    if not isinstance(projection, dict):
        raise ValueError('projection config missing: ' + str(row_dir))
    module = projection.get('module')
    if not isinstance(module, str) or not module:
        raise ValueError('projection module missing: ' + str(row_dir))
    if projection.get('tp_pre_reduce') is not True:
        raise ValueError('projection must bind the local pre-TP output '
                         '(tp_pre_reduce=true): ' + str(row_dir))
    layout = projection.get('weight_layout')
    if layout not in WEIGHT_LAYOUTS:
        raise ValueError('projection weight_layout mismatch: ' + str(row_dir))
    for key, expected in PROJECTION_DTYPES.items():
        if (type(projection.get(key)) is not str
                or projection.get(key) != expected):
            raise ValueError('projection ' + key + ' mismatch: ' + str(row_dir))
    recorded = binding.get('files')
    if not isinstance(recorded, dict) or set(recorded) != set(BINDING_ARRAYS):
        raise ValueError('binding file manifest mismatch: ' + str(row_dir))
    files, hashes = {}, {}
    for name in BINDING_ARRAYS:
        entry = recorded[name]
        if (not isinstance(entry, dict) or entry.get('file') != name + '.npy'
                or entry.get('dtype') != 'bf16-le'):
            raise ValueError('binding file identity mismatch: ' + name
                             + ': ' + str(row_dir))
        file_path = row_dir / entry['file']
        if not file_path.is_file():
            raise ValueError('binding file missing: ' + name + ': ' + str(row_dir))
        digest = sha(file_path)
        if digest != entry.get('sha256'):
            raise ValueError('binding file hash mismatch: ' + name + ': ' + str(row_dir))
        value = np.load(file_path, allow_pickle=False)
        if (not isinstance(entry.get('shape'), list) or value.dtype.str != '<u2'
                or not value.flags.c_contiguous
                or list(value.shape) != entry['shape']):
            raise ValueError('binding file geometry mismatch: ' + name
                             + ': ' + str(row_dir))
        if np.any((value & 0x7f80) == 0x7f80):
            raise ValueError('finite BF16 required: ' + name + ': ' + str(row_dir))
        files[name] = value
        hashes[name] = digest
    resident = binding.get('resident')
    if resident is not None:
        # C1 resident sub-record (additive): identity of the bytes the engine
        # actually keeps resident, hashed against a file carried in the row.
        if not isinstance(resident, dict):
            raise ValueError('binding resident must be an object: ' + str(row_dir))
        for key in ('form', 'file', 'dtype', 'sha256'):
            if not isinstance(resident.get(key), str) or not resident.get(key):
                raise ValueError('binding resident ' + key + ' missing: ' + str(row_dir))
        recipe = resident.get('recipe')
        if (not isinstance(recipe, dict)
                or not isinstance(recipe.get('name'), str) or not recipe.get('name')
                or not isinstance(recipe.get('inputs'), dict)):
            raise ValueError('binding resident recipe incomplete: ' + str(row_dir))
        resident_path = row_dir / resident['file']
        if not resident_path.is_file():
            raise ValueError('binding resident file missing: ' + str(row_dir))
        if sha(resident_path) != resident['sha256']:
            raise ValueError('binding resident hash mismatch: ' + str(row_dir))
    weight, own_output = files['weight'], files['own-output']
    if weight.ndim != 2 or 0 in weight.shape:
        raise ValueError('binding weight must be a nonempty matrix: ' + str(row_dir))
    width = int(np.prod(arrays['expected'].shape[1:]))
    in_features = weight.shape[1] if layout == 'out-in' else weight.shape[0]
    out_features = weight.shape[0] if layout == 'out-in' else weight.shape[1]
    if in_features != width:
        raise ValueError('weight geometry mismatch with input width: ' + str(row_dir))
    if own_output.shape != (tokens, out_features):
        raise ValueError('own-output geometry mismatch: ' + str(row_dir))
    for name, fixture_key in (('candidate-original', 'expected'),
                              ('reference-exact', 'reference-expected')):
        if files[name].shape != arrays[fixture_key].shape:
            raise ValueError(name + ' input geometry mismatch: ' + str(row_dir))
        if not np.array_equal(files[name], arrays[fixture_key]):
            raise ValueError(name + ' input mismatch with norm fixture: '
                             + str(row_dir))
    if files['v24-corrected'].shape != arrays['expected'].shape:
        raise ValueError('v24-corrected input geometry mismatch: ' + str(row_dir))
    return {'manifest': binding, 'files': files, 'sha256': hashes,
            'binding_sha256': sha(path)}


def prepare_row(rank, row, fixture_row_dir, candidate_row_dir, reference_row_dir):
    """Validate the norm fixture row and both engine binding rows together;
    the shared input variants must be byte-identical across engines."""
    fixture, arrays = validate_fixture(Path(fixture_row_dir), row, rank)
    bindings = {engine: validate_binding(Path(directory), rank, row, engine, arrays)
                for engine, directory in (('candidate', candidate_row_dir),
                                          ('reference', reference_row_dir))}
    if not np.array_equal(bindings['candidate']['files']['v24-corrected'],
                          bindings['reference']['files']['v24-corrected']):
        raise ValueError('cross-engine v24-corrected input mismatch: '
                         f'rank {rank} row {row}')
    return {'rank': rank, 'row': row, 'tokens': fixture['tokens'],
            'arrays': arrays, 'bindings': bindings}


def anchor_row(ctx, project):
    """Replay each engine's own original gated-norm output through its bound
    projection; byte-exact reproduction of the engine's OWN captured local
    projection output is required (self-reproduction anchor)."""
    records, outputs, failures = {}, {}, []
    for engine in ENGINES:
        binding = ctx['bindings'][engine]
        variant = OWN_ORIGINAL[engine]
        layout = binding['manifest']['projection']['weight_layout']
        inp = binding['files'][variant].reshape(ctx['tokens'], -1)
        output = project(layout, inp, binding['files']['weight'])
        expected = binding['files']['own-output']
        if output.shape != expected.shape:
            raise ValueError('replayed projection geometry differs from capture: '
                             + engine)
        exact = output.tobytes() == expected.tobytes()
        records[engine] = {
            'variant': variant,
            'self_reproduction': exact,
            'vs_own_output': compare_2d(output, expected),
            'weight_sha256': binding['sha256']['weight'],
            'binding': {'binding_sha256': binding['binding_sha256'],
                        'files': dict(binding['sha256'])}}
        outputs[engine] = output
        if not exact:
            failures.append(f"rank{ctx['rank']}-row{ctx['row']}: {engine} "
                            'self-reproduction anchor failed')
    return records, outputs, failures


def enforce_anchor_gate(failures):
    """Variant replays stay gated behind clean anchors and bindings."""
    if failures:
        raise RuntimeError('C1 self-reproduction anchors failed: '
                           + '; '.join(failures))


def propagation_row(ctx, anchor_records, anchor_outputs, project):
    """Run the remaining input variants per engine and build the per-element
    propagation entries plus the identical-operand cross comparison."""
    tokens = ctx['tokens']
    engines, outputs = {}, {}
    for engine in ENGINES:
        binding = ctx['bindings'][engine]
        layout = binding['manifest']['projection']['weight_layout']
        weight = binding['files']['weight']
        own = OWN_ORIGINAL[engine]
        own_input, own_output = binding['files'][own], binding['files']['own-output']
        variants, engine_outputs = {}, {}
        for variant in VARIANTS:
            input_bits = binding['files'][variant]
            if variant == own:
                output = anchor_outputs[engine]
            else:
                output = project(layout, input_bits.reshape(tokens, -1), weight)
            engine_outputs[variant] = output
            variants[variant] = {
                'input_delta': compare_bf16(input_bits, own_input),
                'output_delta': compare_2d(output, own_output),
                'output_sha256': hashlib.sha256(output.tobytes()).hexdigest()}
        outputs[engine] = engine_outputs
        engines[engine] = {'own_original_variant': own, 'variants': variants,
                           'weight_sha256': binding['sha256']['weight']}
    cross = {}
    for variant in VARIANTS:
        entry = compare_2d(outputs['candidate'][variant],
                           outputs['reference'][variant])
        # validate_binding/prepare_row guarantee identical input bytes here.
        entry['identical_operands'] = True
        cross[variant] = entry
    weight_shas = {engine: engines[engine]['weight_sha256'] for engine in ENGINES}
    return {'row': ctx['row'], 'tokens': tokens,
            'anchors': {engine: anchor_records[engine]['self_reproduction']
                        for engine in ENGINES},
            'weight_sha256': weight_shas,
            'weights_cross_engine_equal': len(set(weight_shas.values())) == 1,
            'engines': engines, 'cross': cross}


def build_matrix(rank, rows, anchors, failures, weight_sha256, harness):
    """Aggregate the propagation matrix; interpretation stays blocked on any
    failure, failed anchor, inconsistent resident weight, or absent rows."""
    consistent = {engine: len(set(weight_sha256.get(engine, []))) <= 1
                  for engine in ENGINES}
    blocked = (bool(failures) or not rows
               or any(anchors.get(engine) is not True for engine in ENGINES)
               or not all(consistent.values()))
    return {'schema': MATRIX_SCHEMA, 'rank': rank, 'anchors': dict(anchors),
            'weights_rank_consistent': consistent, 'rows': rows,
            'interpretation_blocked': blocked, 'harness_sha256': harness}


def enforce_matrix_gate(summary, failures):
    """Fail the command whenever the saved matrix blocks interpretation or
    any anchor/binding check failed."""
    if summary.get('interpretation_blocked') or failures:
        raise RuntimeError('C1 anchors or binding gates failed: '
                           + ('; '.join(failures) or 'summary blocks interpretation'))


def execute(rank, ctxs, project, emit=None, harness=None):
    """Two phases: self-reproduction anchors for every row first — any failure
    returns a blocked matrix before a single variant replay — then the three
    input variants per engine and the identical-operand cross comparison."""
    emit = emit or (lambda kind, payload: None)
    harness = harness if harness is not None else {}
    failures = []
    weight_sha256 = {engine: [ctx['bindings'][engine]['sha256']['weight']
                              for ctx in ctxs]
                     for engine in ENGINES}
    for engine in ENGINES:
        if len(set(weight_sha256[engine])) > 1:
            failures.append(f'rank{rank}: resident {engine} o_proj weight '
                            'differs across rows')
    anchors = {engine: True for engine in ENGINES}
    anchor_records = {}
    for ctx in ctxs:
        records, outputs, row_failures = anchor_row(ctx, project)
        failures.extend(row_failures)
        anchor_records[ctx['row']] = (records, outputs)
        for engine in ENGINES:
            anchors[engine] = anchors[engine] and records[engine]['self_reproduction']
        emit('anchor-receipt', {'rank': rank, 'row': ctx['row'],
                                'anchors': records, 'harness_sha256': harness})
    try:
        enforce_anchor_gate(failures)
    except RuntimeError:
        return build_matrix(rank, [], anchors, failures, weight_sha256, harness), failures
    rows = []
    for ctx in ctxs:
        records, outputs = anchor_records[ctx['row']]
        record = propagation_row(ctx, records, outputs, project)
        rows.append(record)
        emit('row-receipt', record)
    return build_matrix(rank, rows, anchors, failures, weight_sha256, harness), failures


def validate_cli(fixtures, rank, candidate_binding, reference_binding, out):
    if rank not in (0, 1):
        raise ValueError('rank must be 0 or 1')
    for label, path in (('fixtures', fixtures), ('candidate binding', candidate_binding),
                        ('reference binding', reference_binding)):
        if not Path(path).is_dir():
            raise ValueError(label + ' directory required')
    if Path(out).exists():
        raise ValueError('output directory must not exist')


def harness_sha256():
    return {'local_o_proj_replay.py': sha(__file__),
            'gated_norm_ablation.py': sha(Path(gated_norm_ablation.__file__))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('fixtures', 'candidate-binding', 'reference-binding', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--backend', choices=('numpy', 'cupy'), required=True)
    args = parser.parse_args()
    validate_cli(args.fixtures, args.rank, args.candidate_binding,
                 args.reference_binding, args.out)
    rows = row_dirs(args.fixtures, args.rank)
    ctxs = [prepare_row(args.rank, row, fixture_dir,
                        args.candidate_binding / fixture_dir.name,
                        args.reference_binding / fixture_dir.name)
            for row, fixture_dir in rows]
    project = make_backend(args.backend)
    harness = harness_sha256()
    # Immutable output directories preserve failures as well as successes.
    args.out.mkdir(parents=True, exist_ok=False)

    def emit(kind, payload):
        directory = args.out / f'rank{args.rank}-row{payload["row"]}'
        directory.mkdir(exist_ok=True)
        (directory / (kind + '.json')).write_text(
            json.dumps(payload, indent=2) + '\n')

    summary, failures = execute(args.rank, ctxs, project, emit=emit, harness=harness)
    (args.out / 'propagation-matrix.json').write_text(
        json.dumps(summary, indent=2) + '\n')
    enforce_matrix_gate(summary, failures)


if __name__ == '__main__':
    main()
