# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only tests for the experiment C1 local o_proj replay harness."""
import json
import struct

import numpy as np
import pytest

import local_o_proj_replay
from local_o_proj_replay import (BINDING_SCHEMA, ENGINES, VARIANTS,
                                 build_matrix, enforce_anchor_gate,
                                 enforce_matrix_gate, execute, prepare_row,
                                 project_numpy, validate_binding)
from native_ptx import sha


def _fixture_bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def _write_norm_row(row_dir, rank, row, tokens=2, reference_low=0x3f81):
    """Synthesize one norm-replay-fixtures-v1 row whose reference gated-norm
    output differs from the candidate's at exactly one coordinate."""
    row_dir.mkdir(parents=True)
    rng = np.random.default_rng(7000 + 100 * rank + row)
    arrays = {
        'core': rng.integers(1, 0x7800, (tokens, 32, 128)).astype('<u2'),
        'gate': rng.integers(1, 0x7800, (tokens, 4096)).astype('<u2'),
        'weight': rng.integers(1, 0x7800, (128,)).astype('<u2'),
        'expected': np.full((tokens, 32, 128), 0x3f80, dtype='<u2'),
        'reference-expected': np.full((tokens, 32, 128), 0x3f80, dtype='<u2')}
    arrays['reference-expected'][0, 0, 0] = np.uint16(reference_low)
    for name in ('core', 'gate', 'weight'):
        arrays['reference-' + name] = arrays[name].copy()
    manifest = {
        'schema': 'norm-replay-fixtures-v1', 'rank': rank, 'invocation': row,
        'tokens': tokens, 'candidate_epsilon_f32': 1e-5,
        'candidate_epsilon_f32_bits': _fixture_bits(1e-5),
        'reference_epsilon': 1e-5, 'reference_epsilon_f32_bits': _fixture_bits(1e-5),
        'epsilon_f32_equal': True,
        'equal_operands': {'core': True, 'gate': True, 'weight': True},
        'fixtures': {}}
    for name, value in arrays.items():
        path = row_dir / (name + '.npy')
        np.save(path, value, allow_pickle=False)
        manifest['fixtures'][name] = {'file': name + '.npy',
                                      'shape': list(value.shape), 'dtype': 'bf16-le',
                                      'sha256': sha(path)}
    (row_dir / 'fixture.json').write_text(json.dumps(manifest))
    return arrays


def _own_original(engine):
    return local_o_proj_replay.OWN_ORIGINAL[engine]


def _write_binding(row_dir, rank, row, engine, arrays, *, out_features=2,
                   layout='out-in', project=None, own_output=None, v24=None,
                   weight=None, **overrides):
    """Synthesize one atlas-local-o-proj-binding-v1 row bound to a fixture."""
    row_dir.mkdir(parents=True, exist_ok=True)
    tokens = arrays['expected'].shape[0]
    width = int(np.prod(arrays['expected'].shape[1:]))
    if weight is None:
        # Resident weight: constant per (rank, engine), independent of the row;
        # candidate and reference bindings get independent default weights and
        # byte-identical cross-engine weights are tested explicitly.
        rng = np.random.default_rng(9000 + rank + (0 if engine == 'candidate' else 1))
        weight = rng.integers(1, 0x7800,
                              (out_features, width) if layout == 'out-in'
                              else (width, out_features)).astype('<u2')
    project = project or (lambda lay, inp, w: project_numpy(lay, inp, w))
    if own_output is None:
        original = arrays['expected' if engine == 'candidate'
                          else 'reference-expected'].reshape(tokens, width)
        own_output = project(layout, original, weight)
    if v24 is None:
        v24 = arrays['reference-expected'].copy()
    manifest = {
        'schema': BINDING_SCHEMA, 'engine': engine, 'rank': rank, 'layer': 0,
        'invocation': row, 'tokens': tokens,
        'projection': {'module': 'model.layers.0.self_attn.o_proj',
                       'tp_pre_reduce': True, 'weight_layout': layout,
                       'input_dtype': 'bf16', 'weight_dtype': 'bf16',
                       'accumulator_dtype': 'f32', 'output_dtype': 'bf16'},
        'files': {}}
    manifest.update(overrides)
    files = {'weight': weight, 'own-output': own_output,
             'candidate-original': arrays['expected'].copy(),
             'reference-exact': arrays['reference-expected'].copy(),
             'v24-corrected': v24}
    for name, value in files.items():
        path = row_dir / (name + '.npy')
        np.save(path, value, allow_pickle=False)
        manifest['files'][name] = {'file': name + '.npy', 'shape': list(value.shape),
                                   'dtype': 'bf16-le', 'sha256': sha(path)}
    (row_dir / 'binding.json').write_text(json.dumps(manifest))
    return manifest


def _pair(tmp_path, rank=0, row=0, *, tokens=2, reference_low=0x3f81,
          **binding_kwargs):
    fixtures = tmp_path / 'fixtures' / f'rank{rank}-row{row}'
    arrays = _write_norm_row(fixtures, rank, row, tokens=tokens,
                             reference_low=reference_low)
    cand = tmp_path / 'bind-candidate' / f'rank{rank}-row{row}'
    ref = tmp_path / 'bind-reference' / f'rank{rank}-row{row}'
    _write_binding(cand, rank, row, 'candidate', arrays, **binding_kwargs)
    _write_binding(ref, rank, row, 'reference', arrays, **binding_kwargs)
    return fixtures, cand, ref, arrays


def test_validate_binding_accepts_both_engines(tmp_path):
    fixtures, cand, ref, arrays = _pair(tmp_path)
    for engine, path in (('candidate', cand), ('reference', ref)):
        bound = validate_binding(path, 0, 0, engine, arrays)
        assert bound['manifest']['schema'] == BINDING_SCHEMA
        assert bound['manifest']['engine'] == engine
        assert set(bound['files']) == set(local_o_proj_replay.BINDING_ARRAYS)
        assert bound['files']['weight'].shape == (2, 4096)
        assert bound['files']['own-output'].shape == (2, 2)
        assert len(bound['binding_sha256']) == 64
        assert set(bound['sha256']) == set(local_o_proj_replay.BINDING_ARRAYS)


def test_validate_binding_rejects_missing_binding(tmp_path):
    fixtures, cand, ref, arrays = _pair(tmp_path)
    (cand / 'binding.json').unlink()
    with pytest.raises(ValueError, match='binding record missing'):
        validate_binding(cand, 0, 0, 'candidate', arrays)
    with pytest.raises(ValueError, match='binding record missing'):
        validate_binding(tmp_path / 'no-such-dir', 0, 0, 'candidate', arrays)
    (cand / 'binding.json').write_text(
        json.dumps({'schema': BINDING_SCHEMA, 'engine': 'candidate', 'rank': 0,
                    'layer': 0, 'invocation': 0, 'tokens': 2,
                    'projection': {'module': 'm', 'tp_pre_reduce': True,
                                   'weight_layout': 'out-in',
                                   'input_dtype': 'bf16', 'weight_dtype': 'bf16',
                                   'accumulator_dtype': 'f32', 'output_dtype': 'bf16'},
                    'files': {}}))
    with pytest.raises(ValueError, match='file manifest mismatch'):
        validate_binding(cand, 0, 0, 'candidate', arrays)


def test_validate_binding_rejects_identity_mismatch(tmp_path):
    fixtures, cand, ref, arrays = _pair(tmp_path)
    path = cand / 'binding.json'
    base = json.loads(path.read_text())
    cases = (
        ({'schema': 'atlas-local-o-proj-binding-v2'}, 'schema mismatch'),
        ({'rank': 1}, 'rank mismatch'),
        ({'layer': 3}, 'layer mismatch'),
        ({'invocation': 5}, 'invocation mismatch'),
        ({'engine': 'reference'}, 'engine mismatch'),
        ({'tokens': 3}, 'token mismatch'))
    for changes, match in cases:
        manifest = dict(base, **changes)
        path.write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match=match):
            validate_binding(cand, 0, 0, 'candidate', arrays)
    path.write_text(json.dumps(base))


def test_validate_binding_rejects_config_mismatch(tmp_path):
    fixtures, cand, ref, arrays = _pair(tmp_path)
    path = cand / 'binding.json'
    base = json.loads(path.read_text())
    projection = dict(base['projection'])
    cases = (
        (dict(projection, tp_pre_reduce=False), 'pre-TP'),
        (dict(projection, weight_layout='row-major'), 'weight_layout'),
        (dict(projection, accumulator_dtype='f16'), 'accumulator_dtype'),
        (dict(projection, input_dtype='f32'), 'input_dtype'),
        (dict(projection, output_dtype='f32'), 'output_dtype'),
        (dict(projection, weight_dtype='e4m3'), 'weight_dtype'),
        (dict(projection, module=''), 'module'))
    for value, match in cases:
        path.write_text(json.dumps(dict(base, projection=value)))
        with pytest.raises(ValueError, match=match):
            validate_binding(cand, 0, 0, 'candidate', arrays)
    path.write_text(json.dumps(dict(base, projection=None)))
    with pytest.raises(ValueError, match='projection config'):
        validate_binding(cand, 0, 0, 'candidate', arrays)
    path.write_text(json.dumps(base))


def test_validate_binding_rejects_file_and_operand_mismatch(tmp_path):
    fixtures, cand, ref, arrays = _pair(tmp_path)
    path = cand / 'binding.json'
    base = json.loads(path.read_text())

    # Tampered weight bytes after the manifest recorded the hash.
    weight = cand / 'weight.npy'
    original_weight = weight.read_bytes()
    payload = bytearray(original_weight)
    payload[-1] ^= 0x01
    weight.write_bytes(bytes(payload))
    with pytest.raises(ValueError, match='file hash mismatch'):
        validate_binding(cand, 0, 0, 'candidate', arrays)
    weight.write_bytes((tmp_path / 'fixtures' / 'rank0-row0' / 'core.npy').read_bytes())
    with pytest.raises(ValueError, match='file hash mismatch'):
        validate_binding(cand, 0, 0, 'candidate', arrays)
    weight.write_bytes(original_weight)

    # Wrong recorded file identity (rename entry).
    manifest = dict(base)
    manifest['files'] = dict(base['files'])
    manifest['files']['weight'] = dict(base['files']['weight'], file='other.npy')
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='file identity mismatch'):
        validate_binding(cand, 0, 0, 'candidate', arrays)
    path.write_text(json.dumps(base))

    # Recorded shape lying about the stored geometry.
    manifest = dict(base)
    manifest['files'] = dict(base['files'])
    manifest['files']['own-output'] = dict(base['files']['own-output'], shape=[2, 3])
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='geometry mismatch'):
        validate_binding(cand, 0, 0, 'candidate', arrays)
    path.write_text(json.dumps(base))

    # Weight geometry inconsistent with the declared layout and input width.
    bad = cand / 'bad-layout'
    arrays_bad = arrays
    _write_binding(bad, 0, 0, 'candidate', arrays_bad, out_features=3,
                   layout='in-out')
    bad_manifest = json.loads((bad / 'binding.json').read_text())
    bad_manifest['projection']['weight_layout'] = 'out-in'
    (bad / 'binding.json').write_text(json.dumps(bad_manifest))
    with pytest.raises(ValueError, match='weight geometry'):
        validate_binding(bad, 0, 0, 'candidate', arrays_bad)

    # own-output width does not match the weight's output dimension.
    bad2 = cand / 'bad-own'
    bad2.mkdir(exist_ok=True)
    np.save(bad2 / 'own-output.npy',
            np.full((2, 5), 0x3f80, dtype='<u2'), allow_pickle=False)
    manifest = json.loads(path.read_text())
    manifest = dict(manifest)
    manifest['files'] = dict(manifest['files'])
    manifest['files']['own-output'] = dict(manifest['files']['own-output'],
                                           shape=[2, 5],
                                           sha256=sha(bad2 / 'own-output.npy'))
    (bad2 / 'binding.json').write_text(json.dumps(manifest))
    for name in local_o_proj_replay.BINDING_ARRAYS:
        if name != 'own-output':
            data = (cand / (name + '.npy')).read_bytes()
            (bad2 / (name + '.npy')).write_bytes(data)
    with pytest.raises(ValueError, match='own-output geometry'):
        validate_binding(bad2, 0, 0, 'candidate', arrays)

    # The recorded candidate-original must equal the fixture's candidate output.
    changed = arrays['expected'].copy()
    changed[0, 0, 1] = np.uint16(0x3f81)
    np.save(cand / 'candidate-original.npy', changed, allow_pickle=False)
    manifest = json.loads(path.read_text())
    manifest['files'] = dict(manifest['files'])
    manifest['files']['candidate-original'] = dict(
        manifest['files']['candidate-original'], sha256=sha(cand / 'candidate-original.npy'))
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='candidate-original input mismatch'):
        validate_binding(cand, 0, 0, 'candidate', arrays)
    path.write_text(json.dumps(base))

    # Nonfinite BF16 bits are rejected.
    hostile = np.full((2, 2), 0x7f80, dtype='<u2')
    np.save(cand / 'own-output.npy', hostile, allow_pickle=False)
    manifest = json.loads(path.read_text())
    manifest['files'] = dict(manifest['files'])
    manifest['files']['own-output'] = dict(manifest['files']['own-output'],
                                           sha256=sha(cand / 'own-output.npy'))
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='finite BF16'):
        validate_binding(cand, 0, 0, 'candidate', arrays)
    path.write_text(json.dumps(base))


def test_prepare_row_rejects_cross_engine_v24_mismatch(tmp_path):
    fixtures, cand, ref, arrays = _pair(tmp_path)
    prepare_row(0, 0, fixtures, cand, ref)  # identical v24 copies accepted
    diverged = arrays['reference-expected'].copy()
    diverged[0, 1, 1] = np.uint16(0x3f81)
    np.save(ref / 'v24-corrected.npy', diverged, allow_pickle=False)
    manifest = json.loads((ref / 'binding.json').read_text())
    manifest['files']['v24-corrected'] = dict(
        manifest['files']['v24-corrected'], sha256=sha(ref / 'v24-corrected.npy'))
    (ref / 'binding.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='cross-engine v24-corrected input mismatch'):
        prepare_row(0, 0, fixtures, cand, ref)


def test_prepare_row_rejects_wrong_fixture_identity(tmp_path):
    fixtures, cand, ref, arrays = _pair(tmp_path, row=0)
    with pytest.raises(ValueError, match='invocation/row identity mismatch'):
        prepare_row(0, 1, fixtures, cand, ref)


class _CountingProject:
    """Wraps a projection backend and counts replays per call."""

    def __init__(self, inner=project_numpy):
        self.inner = inner
        self.calls = []

    def __call__(self, layout, inp, weight):
        self.calls.append(inp.shape)
        return self.inner(layout, inp, weight)


def _execute_pair(tmp_path, rank=0, row=0, *, project=None, events=None, **kwargs):
    runner = project if project is not None else _CountingProject()
    fixtures, cand, ref, arrays = _pair(tmp_path, rank, row, project=runner, **kwargs)
    ctx = prepare_row(rank, row, fixtures, cand, ref)
    if hasattr(runner, 'calls'):
        runner.calls.clear()  # count only the replays executed by execute()
    sink = events if events is not None else []
    summary, failures = execute(rank, [ctx], runner,
                                emit=lambda kind, payload: sink.append((kind, payload)))
    return summary, failures, runner, sink, arrays, ctx


def test_anchor_pass_emits_binding_receipts_and_matrix(tmp_path):
    summary, failures, project, sink, arrays, ctx = _execute_pair(tmp_path)
    assert failures == []
    assert summary['schema'] == local_o_proj_replay.MATRIX_SCHEMA
    assert summary['anchors'] == {'candidate': True, 'reference': True}
    assert summary['interpretation_blocked'] is False
    assert summary['weights_rank_consistent'] == {'candidate': True, 'reference': True}
    # Two anchor replays (one per engine) then four variant replays.
    assert len(project.calls) == 6

    kinds = [kind for kind, _ in sink]
    assert kinds == ['anchor-receipt', 'row-receipt']
    receipt = sink[0][1]
    assert receipt['rank'] == 0 and receipt['row'] == 0
    for engine in ENGINES:
        record = receipt['anchors'][engine]
        assert record['self_reproduction'] is True
        assert record['vs_own_output']['exact'] is True
        assert len(record['binding']['binding_sha256']) == 64
        assert set(record['binding']['files']) == set(local_o_proj_replay.BINDING_ARRAYS)

    row_record = sink[1][1]
    assert set(row_record['engines']) == set(ENGINES)
    for engine in ENGINES:
        assert set(row_record['engines'][engine]['variants']) == set(VARIANTS)
    assert set(row_record['cross']) == set(VARIANTS)
    for variant in VARIANTS:
        assert row_record['cross'][variant]['identical_operands'] is True
    assert row_record['weights_cross_engine_equal'] is False  # distinct seeds
    assert summary['rows'] == [row_record]


def test_self_anchor_failure_gates_variants_and_fails_closed(tmp_path):
    fixtures, cand, ref, arrays = _pair(tmp_path)
    # Capture a reference own-output that the replay cannot reproduce.
    wrong = np.full((2, 2), 0x3f81, dtype='<u2')
    np.save(ref / 'own-output.npy', wrong, allow_pickle=False)
    manifest = json.loads((ref / 'binding.json').read_text())
    manifest['files']['own-output'] = dict(manifest['files']['own-output'],
                                           sha256=sha(ref / 'own-output.npy'))
    (ref / 'binding.json').write_text(json.dumps(manifest))
    ctx = prepare_row(0, 0, fixtures, cand, ref)
    events = []
    project = _CountingProject()
    summary, failures = execute(0, [ctx], project,
                                emit=lambda kind, payload: events.append(kind))
    assert any('self-reproduction anchor failed' in f for f in failures)
    assert summary['anchors'] == {'candidate': True, 'reference': False}
    assert summary['interpretation_blocked'] is True
    # Gated behind the anchors: only the two anchor replays ran, no row receipt.
    assert len(project.calls) == 2
    assert events == ['anchor-receipt']
    with pytest.raises(RuntimeError, match='self-reproduction anchor failed'):
        enforce_matrix_gate(summary, failures)


def test_anchor_gate_raises_before_variants(tmp_path):
    with pytest.raises(RuntimeError, match='anchor'):
        enforce_anchor_gate(['rank0-row0: reference self-reproduction anchor failed'])
    enforce_anchor_gate([])  # clean pass


def test_variant_propagation_math(tmp_path):
    # Passthrough projection: input ULP deltas map 1:1 to output deltas.
    def passthrough(layout, inp, weight):
        assert layout == 'out-in'
        return inp[:, :weight.shape[0]].copy()
    summary, failures, project, sink, arrays, ctx = _execute_pair(
        tmp_path, project=passthrough, reference_low=0x3f81)
    assert failures == [] and summary['interpretation_blocked'] is False
    row_record = summary['rows'][0]
    cand = row_record['engines']['candidate']['variants']
    # Own original: zero input delta and zero output delta by construction.
    assert cand['candidate-original']['input_delta']['exact'] is True
    assert cand['candidate-original']['output_delta']['exact'] is True
    # reference-exact input is +1 ULP at one coordinate; output follows.
    assert cand['reference-exact']['input_delta'] == {
        'exact': False, 'unequal_elements': 1, 'max_ulp': 1,
        'signed_ulp_at_max': 1, 'coordinates': [[0, 0, 0]]}
    assert cand['reference-exact']['output_delta'] == {
        'exact': False, 'unequal_elements': 1, 'max_ulp': 1,
        'signed_ulp_at_max': 1, 'coordinates': [[0, 0, 0]]}
    ref = row_record['engines']['reference']['variants']
    # For the reference engine the reference-exact variant is its own original.
    assert ref['reference-exact']['input_delta']['exact'] is True
    assert ref['reference-exact']['output_delta']['exact'] is True
    assert ref['candidate-original']['input_delta']['unequal_elements'] == 1
    # v24-corrected equals reference-exact bytes on these fixtures.
    assert cand['v24-corrected'] == cand['reference-exact']
    # The passthrough double ignores weights, so cross outputs at identical
    # operands are exact for every variant; the weight-sensitive cross is
    # covered by test_cross_exact_when_engines_share_weight.
    for variant in VARIANTS:
        assert row_record['cross'][variant]['identical_operands'] is True
        assert row_record['cross'][variant]['exact'] is True


def test_ulp_sign_convention_negative_delta(tmp_path):
    def passthrough(layout, inp, weight):
        return inp[:, :weight.shape[0]].copy()
    summary, failures, project, sink, arrays, ctx = _execute_pair(
        tmp_path, project=passthrough, reference_low=0x3f7f)  # -1 ULP vs 1.0
    assert failures == []
    variants = summary['rows'][0]['engines']['candidate']['variants']
    # Signed ULP is oriented variant-minus-own-original on both deltas.
    assert variants['reference-exact']['input_delta']['signed_ulp_at_max'] == -1
    assert variants['reference-exact']['output_delta']['signed_ulp_at_max'] == -1
    assert variants['reference-exact']['input_delta']['max_ulp'] == 1
    assert variants['reference-exact']['output_delta']['coordinates'] == [[0, 0, 0]]


def test_cross_exact_when_engines_share_weight(tmp_path):
    shared = np.arange(2 * 4096, dtype=np.uint16).astype('<u2') % 0x7800 + 1
    shared = shared.reshape(2, 4096).astype('<u2')
    summary, failures, project, sink, arrays, ctx = _execute_pair(
        tmp_path, weight=shared)
    assert failures == []
    row_record = summary['rows'][0]
    assert row_record['weights_cross_engine_equal'] is True
    for variant in VARIANTS:
        entry = row_record['cross'][variant]
        assert entry['identical_operands'] is True
        assert entry['exact'] is True  # same input bytes + same weight bits


def test_execute_blocks_on_resident_weight_inconsistency(tmp_path):
    fixtures0, cand0, ref0, arrays0 = _pair(tmp_path, row=0)
    fixtures1, cand1, ref1, arrays1 = _pair(tmp_path, row=1)
    other = np.full((2, 4096), 0x3f81, dtype='<u2')
    np.save(cand1 / 'weight.npy', other, allow_pickle=False)
    manifest = json.loads((cand1 / 'binding.json').read_text())
    manifest['files']['weight'] = dict(manifest['files']['weight'],
                                       sha256=sha(cand1 / 'weight.npy'))
    (cand1 / 'binding.json').write_text(json.dumps(manifest))
    # own-output must still match the replay of the *other* weight for the
    # anchor phase to be reached; consistency is checked before anchors run.
    ctxs = [prepare_row(0, 0, fixtures0, cand0, ref0),
            prepare_row(0, 1, fixtures1, cand1, ref1)]
    project = _CountingProject()
    summary, failures = execute(0, ctxs, project, emit=lambda *args: None)
    assert any('weight differs across rows' in f for f in failures)
    assert summary['interpretation_blocked'] is True
    assert summary['weights_rank_consistent']['candidate'] is False
    with pytest.raises(RuntimeError, match='weight differs across rows'):
        enforce_matrix_gate(summary, failures)


def test_build_matrix_requires_complete_rows_and_anchors():
    harness = {'local_o_proj_replay.py': 'a'}
    weights = {'candidate': ['aa'], 'reference': ['bb']}
    summary = build_matrix(0, [], {'candidate': False, 'reference': True},
                           ['rank0-row0: candidate self-reproduction anchor failed'],
                           weights, harness)
    assert summary['interpretation_blocked'] is True
    assert summary['anchors'] == {'candidate': False, 'reference': True}
    assert summary['rows'] == []
    assert summary['harness_sha256'] == harness
    assert summary['weights_rank_consistent'] == {'candidate': True, 'reference': True}
    # No rows is never interpretable, even with clean anchors.
    clean = build_matrix(0, [], {'candidate': True, 'reference': True}, [],
                         weights, harness)
    assert clean['interpretation_blocked'] is True
    with pytest.raises(RuntimeError):
        enforce_matrix_gate(summary, [])
    with pytest.raises(RuntimeError):
        enforce_matrix_gate(clean, [])
    enforce_matrix_gate({'interpretation_blocked': False}, [])


def _attach_resident(row_dir, binding):
    """Extend a synthesized row with a valid C1 resident sub-record."""
    payload = b'resident-shard-bytes'
    (row_dir / 'resident.bin').write_bytes(payload)
    binding = dict(binding)
    binding['resident'] = {
        'form': 'exl3-trellis-mul1', 'file': 'resident.bin',
        'dtype': 'exl3-shard-pack-v1', 'sha256': sha(row_dir / 'resident.bin'),
        'recipe': {'name': 'exl3-mul1-reference-reconstruct-rne-bf16',
                   'inputs': {'resident_sha256': sha(row_dir / 'resident.bin')}}}
    (row_dir / 'binding.json').write_text(json.dumps(binding))
    return binding


def test_validate_binding_accepts_resident_sub_record(tmp_path):
    fixtures, cand, ref, arrays = _pair(tmp_path)
    # Rows without the additive sub-record stay valid (legacy capture form).
    validate_binding(cand, 0, 0, 'candidate', arrays)
    binding = json.loads((cand / 'binding.json').read_text())
    _attach_resident(cand, binding)
    validated = validate_binding(cand, 0, 0, 'candidate', arrays)
    assert validated['manifest']['resident']['form'] == 'exl3-trellis-mul1'
    validate_binding(ref, 0, 0, 'reference', arrays)


@pytest.mark.parametrize('mutation', ['drop-form', 'drop-recipe', 'drop-file',
                                      'hash-mismatch', 'not-an-object'])
def test_validate_binding_rejects_incomplete_resident(tmp_path, mutation):
    fixtures, cand, ref, arrays = _pair(tmp_path)
    binding = json.loads((cand / 'binding.json').read_text())
    _attach_resident(cand, binding)
    binding = json.loads((cand / 'binding.json').read_text())
    if mutation == 'drop-form':
        del binding['resident']['form']
    elif mutation == 'drop-recipe':
        del binding['resident']['recipe']
    elif mutation == 'drop-file':
        del binding['resident']['file']
    elif mutation == 'hash-mismatch':
        binding['resident']['sha256'] = '0' * 64
    else:
        binding['resident'] = 'resident.bin'
    (cand / 'binding.json').write_text(json.dumps(binding))
    with pytest.raises(ValueError):
        validate_binding(cand, 0, 0, 'candidate', arrays)


def test_validate_binding_rejects_missing_resident_file(tmp_path):
    fixtures, cand, ref, arrays = _pair(tmp_path)
    binding = json.loads((cand / 'binding.json').read_text())
    _attach_resident(cand, binding)
    (cand / 'resident.bin').unlink()
    with pytest.raises(ValueError):
        validate_binding(cand, 0, 0, 'candidate', arrays)
