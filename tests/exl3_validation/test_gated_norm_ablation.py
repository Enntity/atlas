# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only tests for the gated-normalization arithmetic ablation harness."""
import hashlib
import json
import struct
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

import gated_norm_ablation
from gated_norm_ablation import (FROZEN_KDA_SHA256, build_summary, compare_bf16,
                                 validate_cli)
from gated_norm_variants import (KERNEL_ENTRY, VARIANTS, derive, frozen_body,
                                 render_variant)
from native_ptx import sha
from test_variant_compile import COMPILER, _fake_nvcc, _manifest

KDA_CU = Path(__file__).resolve().parents[2] / 'kernels' / 'gb10' / 'common' / 'kda.cu'

_COMPILE = {'schema': 'atlas-variant-compile-nvrtc-v1', 'lane': 'nvrtc',
            'toolchain': 'NVRTC', 'options': ['--std=c++17', '--fmad=false'],
            'variants': {}}


def _changed_lines(diff):
    removed, added = [], []
    for line in diff.splitlines():
        if line.startswith('---') or line.startswith('+++'):
            continue
        if line.startswith('-'):
            removed.append(line[1:])
        elif line.startswith('+'):
            added.append(line[1:])
    return removed, added


def test_v0_copy_diff_is_empty_and_body_verbatim():
    frozen = frozen_body(KDA_CU)
    source, diff, _ = derive('v0-copy', KDA_CU)
    assert diff == ''
    assert frozen in source
    assert VARIANTS['v0-copy'](frozen) == frozen


def test_frozen_kernel_matches_ablation_contract():
    assert sha(KDA_CU) == FROZEN_KDA_SHA256
    assert frozen_body(KDA_CU).startswith(
        'extern "C" __global__ void ' + KERNEL_ENTRY + '(')
    assert frozen_body(KDA_CU).endswith('}')


def test_frozen_body_requires_exactly_one_entry(tmp_path):
    source = KDA_CU.read_text()
    doubled = tmp_path / 'kda.cu'
    doubled.write_text(source + '\n' + source)
    with pytest.raises(ValueError):
        frozen_body(doubled)
    absent = tmp_path / 'other.cu'
    absent.write_text('extern "C" __global__ void other_kernel() {}\n')
    with pytest.raises(ValueError):
        frozen_body(absent)


def test_variant_diffs_touch_only_documented_regions():
    frozen = frozen_body(KDA_CU)
    for name, transform in VARIANTS.items():
        source, diff, _ = derive(name, KDA_CU)
        if name == 'v0-copy':
            assert diff == ''
            continue
        removed, added = _changed_lines(diff)
        assert set(removed) <= set(transform.old.splitlines()), name
        assert set(added) <= set(transform.new.splitlines()), name
        assert removed and added, name
        assert frozen.replace(transform.old, transform.new, 1) in source, name


def test_variant_diffs_change_one_documented_feature():
    _, diff, _ = derive('v1-reduction-order', KDA_CU)
    removed, added = _changed_lines(diff)
    assert any('for (unsigned int i = 1; i < dim; ++i) sum += red[i];' in line
               for line in added)
    assert not any('asm' in line for line in removed + added)

    _, diff, _ = derive('v2-rsqrt-sqrt-div', KDA_CU)
    removed, added = _changed_lines(diff)
    assert any('sqrt.approx.ftz.f32' in line for line in added)
    assert not any('expf' in line or 'sigmoid' in line for line in removed + added)

    _, diff, _ = derive('v3-sigmoid-exp', KDA_CU)
    removed, added = _changed_lines(diff)
    assert any('ex2.approx.f32' in line for line in added)
    assert any('/ (1.0f + e);' in line for line in added)  # division kept
    assert any('/ (1.0f + expf(-g))' in line for line in removed)

    _, diff, _ = derive('v4-sigmoid-reciprocal-mul', KDA_CU)
    removed, added = _changed_lines(diff)
    assert any('expf(-g)' in line for line in added)  # expf kept
    assert any('* sig;' in line for line in added)  # divide -> reciprocal multiply
    assert any('/ (1.0f + expf(-g))' in line for line in removed)

    _, diff, _ = derive('v-all', KDA_CU)
    _, added = _changed_lines(diff)
    assert any('sum += red[i]' in line for line in added)  # v1 reduction
    assert any('sqrt.approx.ftz.f32' in line for line in added)  # v2 rsqrt
    assert any('ex2.approx.f32' in line for line in added)  # v3 sigmoid exp
    assert any('* sig;' in line for line in added)  # v4 reciprocal multiply

    _, diff, _ = derive('v24-minimal', KDA_CU)
    removed, added = _changed_lines(diff)
    assert any('sqrt.approx.ftz.f32' in line for line in added)  # v2 rsqrt
    assert any('* sig;' in line for line in added)  # v4 reciprocal multiply
    assert any('expf(-g)' in line for line in added)  # expf kept (no v3)
    assert not any('sum += red[i]' in line for line in added)  # no v1 reduction
    assert not any('ex2.approx.f32' in line for line in added)  # no v3 sigmoid exp


def test_transforms_fail_closed_on_mutated_snippets():
    frozen = frozen_body(KDA_CU)
    for name, transform in VARIANTS.items():
        if name == 'v0-copy':
            continue
        mutated = frozen.replace(transform.old, transform.old.replace(' ', '\t', 1), 1)
        assert mutated != frozen
        with pytest.raises(ValueError):
            transform(mutated)
    renamed = frozen.replace('rsqrtf', 'rsqrt')
    for name in ('v2-rsqrt-sqrt-div', 'v24-minimal', 'v-all'):
        with pytest.raises(ValueError):
            VARIANTS[name](renamed)


def test_derivation_is_deterministic():
    for name in VARIANTS:
        assert derive(name, KDA_CU) == derive(name, KDA_CU)


def test_rendered_wrapper_header_and_includes():
    frozen = frozen_body(KDA_CU)
    for name in VARIANTS:
        lines = render_variant(name, VARIANTS[name](frozen)).splitlines()
        assert lines[0] == '// SPDX-License-Identifier: AGPL-3.0-only'
        assert name in lines[1] and 'gated_norm_variants.py' in lines[1]
        assert '#include "atlas_pdl.cuh"' in lines
        assert '#include <cuda_bf16.h>' in lines
        assert any(KERNEL_ENTRY + '(' in line and line.startswith('extern "C"')
                   for line in lines)


def test_compare_bf16_ulp_signed_and_coordinates():
    # 1-ULP up, 1-ULP down, sign-crossing (-0.0 vs +0.0), 2-ULP max.
    target = np.array([[[0x3f80, 0x3f80, 0x0000], [0x3f80, 0x4000, 0x3f80]]], dtype='<u2')
    output = np.array([[[0x3f81, 0x3f7f, 0x8000], [0x3f82, 0x4000, 0x3f80]]], dtype='<u2')
    assert compare_bf16(output, target) == {
        'exact': False, 'unequal_elements': 4, 'max_ulp': 2, 'signed_ulp_at_max': 2,
        'coordinates': [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0]]}
    assert compare_bf16(target, target) == {
        'exact': True, 'unequal_elements': 0, 'max_ulp': 0, 'signed_ulp_at_max': 0,
        'coordinates': []}


def test_compare_bf16_signed_at_first_max_element():
    target = np.array([[[0x3f80, 0x3f80]]], dtype='<u2')
    output = np.array([[[0x3f7d, 0x3f83]]], dtype='<u2')
    result = compare_bf16(output, target)
    assert result['max_ulp'] == 3
    assert result['signed_ulp_at_max'] == -3  # first of two max-distance elements
    assert result['coordinates'] == [[0, 0, 0], [0, 0, 1]]


def test_compare_bf16_caps_coordinates():
    target = np.zeros((2, 32, 128), dtype='<u2')
    output = target.copy()
    output.reshape(-1)[:70] = 0x3f80
    result = compare_bf16(output, target)
    assert result['exact'] is False
    assert result['unequal_elements'] == 70
    assert len(result['coordinates']) == 64
    assert result['coordinates_truncated'] is True
    assert result['coordinates'][0] == [0, 0, 0]


def _run_record(self_reproduction=True, exact_reference=True, unequal_reference=0,
                max_ulp_reference=0):
    # Engagement controls as written by the runner's control loop for the
    # attested/v0-copy/v-all/v24-minimal runs (native_gated_norm_replay.perturbations).
    return {
        'self_reproduction': self_reproduction,
        'vs_candidate': {'exact': True, 'unequal_elements': 0, 'max_ulp': 0,
                         'signed_ulp_at_max': 0, 'coordinates': []},
        'vs_reference': {'exact': exact_reference, 'unequal_elements': unequal_reference,
                         'max_ulp': max_ulp_reference, 'signed_ulp_at_max': 0,
                         'coordinates': []},
        'controls': {name: {'engaged': True, 'unequal_elements': 5}
                     for name in ('input', 'gate', 'weight')}}


def _synthetic_rows(v0_copy_ok=True, attested_ok=True):
    rows = []
    for row in (0, 1):
        runs = {name: _run_record() for name in ('attested',) + tuple(VARIANTS)}
        runs['v-all'] = _run_record(exact_reference=False, unequal_reference=3 + row,
                                    max_ulp_reference=1 + row)
        runs['v0-copy'] = _run_record(self_reproduction=v0_copy_ok or row == 0)
        runs['attested'] = _run_record(self_reproduction=attested_ok)
        rows.append({'row': row, 'tokens': 10, 'runs': runs})
    return rows


def test_summary_aggregates_and_interpretation_blocked():
    harness = {'gated_norm_ablation.py': 'a', 'gated_norm_variants.py': 'b'}
    summary = build_summary(0, _synthetic_rows(), harness, _COMPILE)
    assert summary['schema'] == 'atlas-gated-norm-ablation-summary-v2'
    assert summary['runs']['attested'] == {
        'rows_exact_vs_candidate': 2, 'rows_exact_vs_reference': 2,
        'total_unequal_vs_reference': 0, 'max_ulp_vs_reference': 0}
    assert summary['runs']['v-all'] == {
        'rows_exact_vs_candidate': 2, 'rows_exact_vs_reference': 0,
        'total_unequal_vs_reference': 7, 'max_ulp_vs_reference': 2}
    assert summary['rows'][1]['runs']['v-all']['vs_reference']['unequal_elements'] == 4
    assert summary['interpretation_blocked'] is False
    assert summary['anchors'] == {'attested': True, 'v0-copy': True}
    assert summary['harness_sha256'] == harness

    blocked = build_summary(1, _synthetic_rows(v0_copy_ok=False), harness, _COMPILE)
    assert blocked['interpretation_blocked'] is True
    assert blocked['anchors'] == {'attested': True, 'v0-copy': False}
    assert blocked['runs']['v-all']['rows_exact_vs_candidate'] == 2  # runs not skipped

    attested_blocked = build_summary(1, _synthetic_rows(attested_ok=False), harness, _COMPILE)
    assert attested_blocked['interpretation_blocked'] is True
    assert attested_blocked['anchors'] == {'attested': False, 'v0-copy': True}


def test_summary_retains_engagement_controls():
    summary = build_summary(0, _synthetic_rows(), {'gated_norm_ablation.py': 'a'}, _COMPILE)
    for name in ('attested', 'v0-copy', 'v-all', 'v24-minimal'):
        controls = summary['rows'][0]['runs'][name]['controls']
        assert set(controls) == {'input', 'gate', 'weight'}
        assert all(record['engaged'] for record in controls.values())
        assert summary['rows'][0]['runs'][name]['controls_engaged'] is True
    assert 'controls' not in summary['rows'][0]['runs']['v1-reduction-order']


def test_summary_blocks_on_one_failed_control_with_passing_anchors():
    harness = {'gated_norm_ablation.py': 'a'}
    rows = _synthetic_rows()
    rows[1]['runs']['v-all']['controls']['gate']['engaged'] = False
    summary = build_summary(0, rows, harness, _COMPILE)
    assert all(summary['anchors'].values())  # every replay anchor passes
    assert summary['interpretation_blocked'] is True  # one failed control blocks
    assert summary['rows'][1]['runs']['v-all']['controls']['gate'] == {
        'engaged': False, 'unequal_elements': 5}
    assert summary['runs']['v-all']['rows_exact_vs_candidate'] == 2  # runs not skipped


def test_summary_blocks_on_absent_required_control():
    harness = {}
    rows = _synthetic_rows()
    del rows[0]['runs']['attested']['controls']['weight']
    summary = build_summary(0, rows, harness, _COMPILE)
    assert all(summary['anchors'].values())
    assert summary['interpretation_blocked'] is True
    assert 'weight' not in summary['rows'][0]['runs']['attested']['controls']

    rows = _synthetic_rows()
    del rows[1]['runs']['v0-copy']['controls']
    summary = build_summary(0, rows, harness, _COMPILE)
    assert summary['interpretation_blocked'] is True
    assert summary['rows'][1]['runs']['v0-copy']['controls'] is None


def test_summary_gate_fails_command_when_blocked():
    harness = {}
    gated_norm_ablation.enforce_summary_gate(
        build_summary(0, _synthetic_rows(), harness, _COMPILE), [])  # clean run exits 0
    rows = _synthetic_rows()
    rows[1]['runs']['v-all']['controls']['gate']['engaged'] = False
    blocked = build_summary(0, rows, harness, _COMPILE)
    with pytest.raises(RuntimeError):
        gated_norm_ablation.enforce_summary_gate(blocked, [])
    with pytest.raises(RuntimeError):
        gated_norm_ablation.enforce_summary_gate(
            blocked, ['rank0-row1: v-all control gate did not engage'])


def test_summary_requires_complete_run_records():
    rows = _synthetic_rows()
    del rows[0]['runs']['v-all']
    with pytest.raises(ValueError):
        build_summary(0, rows, {}, _COMPILE)
    with pytest.raises(ValueError):
        build_summary(0, [], {}, _COMPILE)


def test_summary_requires_explicit_variant_compile_record():
    for record in ({}, None, {'lane': ''}):
        with pytest.raises(ValueError, match='variant compile record'):
            build_summary(0, _synthetic_rows(), {}, record)


def test_summary_blocks_when_v24_minimal_controls_do_not_engage():
    rows = _synthetic_rows()
    rows[0]['runs']['v24-minimal']['controls']['gate']['engaged'] = False
    summary = build_summary(0, rows, {}, _COMPILE)
    assert all(summary['anchors'].values())  # replay anchors still pass
    assert summary['interpretation_blocked'] is True
    assert summary['rows'][0]['runs']['v24-minimal']['controls_engaged'] is False

    rows = _synthetic_rows()
    del rows[1]['runs']['v24-minimal']['controls']
    summary = build_summary(0, rows, {}, _COMPILE)
    assert summary['interpretation_blocked'] is True
    with pytest.raises(RuntimeError):
        gated_norm_ablation.enforce_summary_gate(summary, [])


def test_validate_cli_rejects_bad_rank_and_existing_out(tmp_path):
    fixtures = tmp_path / 'fixtures'
    fixtures.mkdir()
    for rank in (2, -1, 3):
        with pytest.raises(ValueError):
            validate_cli(fixtures, rank, tmp_path / 'out')
    out = tmp_path / 'out'
    out.mkdir()
    with pytest.raises(ValueError):
        validate_cli(fixtures, 0, out)
    with pytest.raises(ValueError):
        validate_cli(tmp_path / 'missing', 0, tmp_path / 'fresh')
    validate_cli(fixtures, 0, tmp_path / 'fresh')
    validate_cli(fixtures, 1, tmp_path / 'fresh')


def _fixture_bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def _write_fixture_row(row_dir, rank=1, row=0, tokens=2, schema='norm-replay-fixtures-v1'):
    """Synthesize one norm-replay-fixtures-v1 row that satisfies the runner."""
    row_dir.mkdir(parents=True)
    rng = np.random.default_rng(1000 * rank + row)
    arrays = {
        'core': rng.integers(0, 0x7800, (tokens, 32, 128)).astype('<u2'),
        'gate': rng.integers(0, 0x7800, (tokens, 4096)).astype('<u2'),
        'weight': rng.integers(0, 0x7800, (128,)).astype('<u2'),
        'expected': rng.integers(0, 0x7800, (tokens, 32, 128)).astype('<u2'),
        'reference-expected': rng.integers(0, 0x7800, (tokens, 32, 128)).astype('<u2')}
    for name in ('core', 'gate', 'weight'):
        arrays['reference-' + name] = arrays[name].copy()
    manifest = {
        'schema': schema, 'rank': rank, 'invocation': row, 'tokens': tokens,
        'candidate_epsilon_f32': 1e-5, 'candidate_epsilon_f32_bits': _fixture_bits(1e-5),
        'reference_epsilon': 1e-5, 'reference_epsilon_f32_bits': _fixture_bits(1e-5),
        'epsilon_f32_equal': True,
        'equal_operands': {'core': True, 'gate': True, 'weight': True},
        'fixtures': {}}
    for name, value in arrays.items():
        path = row_dir / (name + '.npy')
        np.save(path, value, allow_pickle=False)
        manifest['fixtures'][name] = {'file': name + '.npy', 'shape': list(value.shape),
                                      'dtype': 'bf16-le', 'sha256': sha(path)}
    (row_dir / 'fixture.json').write_text(json.dumps(manifest))
    return manifest


def _rewrite_manifest(row_dir, manifest):
    (row_dir / 'fixture.json').write_text(json.dumps(manifest))


def test_validate_fixture_accepts_valid_row(tmp_path):
    row_dir = tmp_path / 'rank1-row0'
    _write_fixture_row(row_dir, rank=1, row=0, tokens=2)
    fixture, arrays = gated_norm_ablation.validate_fixture(row_dir, 0, 1)
    assert fixture['schema'] == 'norm-replay-fixtures-v1'
    assert fixture['rank'] == 1 and fixture['invocation'] == 0
    assert arrays['core'].shape == (2, 32, 128)
    assert np.array_equal(arrays['core'], arrays['reference-core'])
    other = tmp_path / 'rank0-row0'
    _write_fixture_row(other, rank=0, row=0)
    gated_norm_ablation.validate_fixture(other, 0, 0)


def test_validate_fixture_rejects_wrong_schema(tmp_path):
    row_dir = tmp_path / 'rank1-row0'
    _write_fixture_row(row_dir, schema='norm-replay-fixtures-v2')
    with pytest.raises(ValueError, match='fixture schema mismatch'):
        gated_norm_ablation.validate_fixture(row_dir, 0, 1)


def test_validate_fixture_rejects_tampered_file_hash(tmp_path):
    row_dir = tmp_path / 'rank1-row0'
    _write_fixture_row(row_dir)
    path = row_dir / 'core.npy'
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 0x01  # tamper with the file after the manifest recorded its hash
    path.write_bytes(bytes(payload))
    with pytest.raises(ValueError, match='fixture file hash mismatch'):
        gated_norm_ablation.validate_fixture(row_dir, 0, 1)

    row_dir = tmp_path / 'rank1-row1'
    manifest = _write_fixture_row(row_dir, row=1)
    manifest['fixtures']['gate']['sha256'] = '0' * 64  # tamper with the recorded hash
    _rewrite_manifest(row_dir, manifest)
    with pytest.raises(ValueError, match='fixture file hash mismatch'):
        gated_norm_ablation.validate_fixture(row_dir, 1, 1)


def test_validate_fixture_rejects_wrong_rank(tmp_path):
    row_dir = tmp_path / 'rank1-row0'
    _write_fixture_row(row_dir, rank=0, row=0)  # rank-0 fixture inside a rank-1 row
    with pytest.raises(ValueError, match='fixture rank mismatch'):
        gated_norm_ablation.validate_fixture(row_dir, 0, 1)


def test_validate_fixture_rejects_wrong_invocation(tmp_path):
    row_dir = tmp_path / 'rank1-row0'
    _write_fixture_row(row_dir, rank=1, row=3)
    with pytest.raises(ValueError, match='invocation/row identity mismatch'):
        gated_norm_ablation.validate_fixture(row_dir, 0, 1)


def test_validate_fixture_rejects_unequal_cross_engine_operands(tmp_path):
    row_dir = tmp_path / 'rank1-row0'
    manifest = _write_fixture_row(row_dir)
    manifest['equal_operands']['weight'] = False  # recorded inequality
    _rewrite_manifest(row_dir, manifest)
    with pytest.raises(ValueError, match='recorded cross-engine operand inequality'):
        gated_norm_ablation.validate_fixture(row_dir, 0, 1)

    row_dir = tmp_path / 'rank1-row1'
    manifest = _write_fixture_row(row_dir, row=1)
    changed = np.load(row_dir / 'reference-weight.npy', allow_pickle=False)
    changed ^= np.uint16(0x8000)
    np.save(row_dir / 'reference-weight.npy', changed, allow_pickle=False)
    manifest['fixtures']['reference-weight']['sha256'] = sha(row_dir / 'reference-weight.npy')
    _rewrite_manifest(row_dir, manifest)  # record claims equality, bytes do not
    with pytest.raises(ValueError, match='actual cross-engine operand mismatch'):
        gated_norm_ablation.validate_fixture(row_dir, 1, 1)


def test_validate_fixture_rejects_changed_epsilon_bits(tmp_path):
    row_dir = tmp_path / 'rank1-row0'
    manifest = _write_fixture_row(row_dir)
    manifest['candidate_epsilon_f32_bits'] ^= 1  # recorded bits vs effective epsilon
    _rewrite_manifest(row_dir, manifest)
    with pytest.raises(ValueError, match='recorded epsilon bits mismatch'):
        gated_norm_ablation.validate_fixture(row_dir, 0, 1)

    row_dir = tmp_path / 'rank1-row1'
    manifest = _write_fixture_row(row_dir, row=1)
    manifest['reference_epsilon'] = 1e-4  # internally consistent but cross-engine unequal
    manifest['reference_epsilon_f32_bits'] = _fixture_bits(1e-4)
    _rewrite_manifest(row_dir, manifest)
    with pytest.raises(ValueError, match='cross-engine epsilon bits differ'):
        gated_norm_ablation.validate_fixture(row_dir, 1, 1)

    row_dir = tmp_path / 'rank1-row2'
    manifest = _write_fixture_row(row_dir, row=2)
    manifest['epsilon_f32_equal'] = False
    _rewrite_manifest(row_dir, manifest)
    with pytest.raises(ValueError, match='cross-engine epsilon bits differ'):
        gated_norm_ablation.validate_fixture(row_dir, 2, 1)


def test_validate_fixture_rejects_reordered_or_mismatched_rows(tmp_path):
    row0 = tmp_path / 'rank1-row0'
    row1 = tmp_path / 'rank1-row1'
    _write_fixture_row(row0, rank=1, row=0)
    _write_fixture_row(row1, rank=1, row=1)
    swapped = (row0 / 'fixture.json').read_text()
    (row0 / 'fixture.json').write_text((row1 / 'fixture.json').read_text())
    (row1 / 'fixture.json').write_text(swapped)
    for row_dir, row in ((row0, 0), (row1, 1)):
        with pytest.raises(ValueError, match='invocation/row identity mismatch'):
            gated_norm_ablation.validate_fixture(row_dir, row, 1)

    row_dir = tmp_path / 'rank1-row2'
    manifest = _write_fixture_row(row_dir, row=2)
    manifest['tokens'] = 99  # row identity binds the recorded token count to the core
    _rewrite_manifest(row_dir, manifest)
    with pytest.raises(ValueError, match='token identity mismatch'):
        gated_norm_ablation.validate_fixture(row_dir, 2, 1)


# --- main() wiring, driven end to end with GPU-dependent pieces stubbed ---

def _stub_main(monkeypatch, tmp_path, production=False):
    """Drive main() on one valid synthetic fixture row with every GPU-, PTX-
    and frozen-source-dependent call stubbed. Returns (out_dir, fixture_dir,
    calls) where calls records replay/enforce invocations."""
    fixtures = tmp_path / 'fixtures'
    _write_fixture_row(fixtures / 'rank1-row0', rank=1, row=0, tokens=2)
    ptx_dir = tmp_path / 'ptx'
    ptx_dir.mkdir()
    (ptx_dir / 'kda.ptx').write_text(
        '.target sm_121f\n.entry ' + KERNEL_ENTRY + '( .param .u64 a, .param .u64 b, '
        '.param .u64 c, .param .u64 d, .param .u32 e, .param .u32 f, .param .f32 g )')
    (tmp_path / 'binding.json').write_text('{}')
    out = tmp_path / 'out'
    calls = {'replay': 0, 'enforce': None}
    cupy = MagicMock()
    cupy.cuda.runtime.runtimeGetVersion.return_value = 13000
    cupy.cuda.runtime.driverGetVersion.return_value = 13000
    monkeypatch.setitem(sys.modules, 'cupy', cupy)
    monkeypatch.setattr(gated_norm_ablation, 'validate_binding', lambda *a: {
        'image': 'img@sha256:' + '0' * 64,
        'modules': {'kda': {'file': 'kda.ptx', 'sha256': 'a' * 64}}})
    real_sha = gated_norm_ablation.sha
    frozen_kda = tmp_path / 'kda.cu'
    monkeypatch.setattr(gated_norm_ablation, 'sha', lambda path: (
        FROZEN_KDA_SHA256 if Path(path) == frozen_kda else real_sha(path)))
    monkeypatch.setattr(gated_norm_ablation, 'frozen_body', lambda path: 'body')
    monkeypatch.setattr(gated_norm_ablation, 'derive',
                        lambda name, path: ('code', 'diff', 'source-sha'))
    monkeypatch.setattr(gated_norm_ablation, 'load_module', lambda *a: MagicMock())
    monkeypatch.setattr(gated_norm_ablation, 'validate_operands',
                        lambda core, gate, weight, eps: ((1, 1, 1), (128, 1, 1), eps))
    monkeypatch.setattr(gated_norm_ablation, 'device_properties', lambda props: {})
    monkeypatch.setattr(gated_norm_ablation, 'perturbations', lambda c, g, w: {})

    def replay(cp, kernel, core, gate, weight, eps):
        calls['replay'] += 1
        return np.load(fixtures / 'rank1-row0' / 'expected.npy')
    monkeypatch.setattr(gated_norm_ablation, 'replay', replay)

    def enforce(summary, failures):
        calls['enforce'] = (summary, list(failures))
    monkeypatch.setattr(gated_norm_ablation, 'enforce_summary_gate', enforce)
    argv = ['gated_norm_ablation.py', '--fixtures', str(fixtures), '--rank', '1',
            '--common', str(tmp_path), '--ptx-dir', str(ptx_dir),
            '--ptx-binding', str(tmp_path / 'binding.json'), '--out', str(out)]
    if production:
        script, _ = _fake_nvcc(tmp_path)
        argv += ['--variant-compile', 'nvcc-production',
                 '--build-manifest', str(_manifest(tmp_path)),
                 '--nvcc', str(script)]
    else:
        argv += ['--variant-compile', 'nvrtc']
    monkeypatch.setattr(sys, 'argv', argv)
    return out, fixtures, calls


def test_main_validates_fixture_before_any_replay(monkeypatch, tmp_path):
    out, _, calls = _stub_main(monkeypatch, tmp_path)

    def reject(row_dir, row, rank):
        raise ValueError('fixture rejected by validator')
    monkeypatch.setattr(gated_norm_ablation, 'validate_fixture', reject)
    with pytest.raises(ValueError, match='fixture rejected by validator'):
        gated_norm_ablation.main()
    assert calls['replay'] == 0
    assert calls['enforce'] is None


def test_main_passes_saved_summary_to_enforce_summary_gate(monkeypatch, tmp_path):
    out, _, calls = _stub_main(monkeypatch, tmp_path)
    gated_norm_ablation.main()
    assert calls['replay'] == 1 + len(VARIANTS)
    summary, failures = calls['enforce']
    saved = json.loads((out / 'ablation-summary.json').read_text())
    assert summary == saved
    assert saved['variant_compile']['lane'] == 'nvrtc'
    # Perturbations are stubbed to none, so control runs carry no engaged
    # controls and the summary must block interpretation.
    assert saved['interpretation_blocked'] is True


def test_main_writes_failure_receipt_and_reraises_on_crash(monkeypatch, tmp_path):
    out, _, calls = _stub_main(monkeypatch, tmp_path)

    def crash(*args):
        raise RuntimeError('simulated replay crash')
    monkeypatch.setattr(gated_norm_ablation, 'replay', crash)
    with pytest.raises(RuntimeError, match='simulated replay crash'):
        gated_norm_ablation.main()
    failure = json.loads((out / 'rank1-row0' / 'failure.json').read_text())
    assert failure['exception_type'] == 'RuntimeError'
    assert failure['message'] == 'simulated replay crash'
    assert failure['row'] == 0
    assert failure['utc_time'].endswith('Z')
    assert calls['enforce'] is None


def test_main_nvcc_production_lane_compiles_and_records(monkeypatch, tmp_path):
    out, _, calls = _stub_main(monkeypatch, tmp_path, production=True)
    gated_norm_ablation.main()
    summary, _ = calls['enforce']
    record = summary['variant_compile']
    assert record['lane'] == 'nvcc-production'
    assert record['arch'] == 'sm_121f'
    assert record['compiler'] == COMPILER
    assert record['effective_flags'] == [
        '--fmad=false', '-DTQ_PLUS_SIGNS', '--expt-relaxed-constexpr']
    for name in VARIANTS:
        ptx = out / 'variant-ptx' / (name + '.ptx')
        assert ptx.is_file()
        assert (out / 'variant-src' / (name + '.cu')).read_text() == 'code'
        assert record['variants'][name]['ptx_sha256'] == hashlib.sha256(
            ptx.read_bytes()).hexdigest()
    receipt = json.loads((out / 'rank1-row0' / 'receipt.json').read_text())
    assert receipt['variant_compile'] == record
    assert receipt['runs'][1]['variant_compile'] == 'nvcc-production'
    assert (tmp_path / 'nvcc-args.txt').read_text().splitlines()[:3] == [
        '--ptx', '-arch=sm_121f', '-O3']


# --- Item 2: negative validator branch tests ---

def _synthetic_fixture_dir(tmp_path, rank=0, row=0, bad_key=None, bad_file=None,
                           bad_dtype=False, bad_shape=False, bad_hash=False,
                           bad_epsilon_bits=False, bad_epsilon_underflow=False):
    d = tmp_path / ('row_' + str(bad_key) if bad_key else 'row'); d.mkdir()
    # build minimal arrays
    core = np.zeros((2, 32, 128), dtype=np.uint16)
    gate = np.zeros((2, 4096), dtype=np.uint16)
    weight = np.zeros((128,), dtype=np.uint16)
    exp = np.zeros((2, 32, 128), dtype=np.uint16)
    ref_c = core.copy(); ref_g = gate.copy(); ref_w = weight.copy(); ref_e = exp.copy()
    for name, arr in (('core', core), ('gate', gate), ('weight', weight),
                      ('expected', exp), ('reference-core', ref_c),
                      ('reference-gate', ref_g), ('reference-weight', ref_w),
                      ('reference-expected', ref_e)):
        path = d / (name + '.npy')
        np.save(path, arr)
    fixture = {
        'schema': 'norm-replay-fixtures-v1', 'rank': rank, 'invocation': row,
        'tokens': 2, 'candidate_epsilon_f32': 1e-3, 'reference_epsilon': 1e-3,
        'epsilon_f32_equal': True,
        'candidate_epsilon_f32_bits': 0x3c23d70a,
        'reference_epsilon_f32_bits': 0x3c23d70a,
        'equal_operands': {'core': True, 'gate': True, 'weight': True},
        'fixtures': {}
    }
    for name in ('core', 'gate', 'weight', 'expected',
                 'reference-core', 'reference-gate', 'reference-weight', 'reference-expected'):
        fixture['fixtures'][name] = {
            'file': name + '.npy', 'dtype': 'bf16-le',
            'sha256': gated_norm_ablation.sha(d / (name + '.npy')),
            'shape': list(np.load(d / (name + '.npy')).shape)}
    if bad_key == 'keyset':
        fixture['fixtures']['extra'] = fixture['fixtures']['core'].copy()
        del fixture['fixtures']['expected']
    elif bad_key == 'file_identity':
        fixture['fixtures']['core']['file'] = 'wrong.npy'
    elif bad_key == 'dtype':
        fixture['fixtures']['core']['dtype'] = 'f32-le'
    elif bad_key == 'hash':
        fixture['fixtures']['core']['sha256'] = '0'*64
    elif bad_key == 'shape':
        fixture['fixtures']['core']['shape'] = [99, 32, 128]
    elif bad_key == 'operands_unequal':
        fixture['equal_operands']['core'] = False
    elif bad_key == 'epsilon_bits':
        fixture['candidate_epsilon_f32_bits'] = 0
    elif bad_key == 'epsilon_underflow':
        fixture['candidate_epsilon_f32'] = 0.0
    return d, fixture


def test_validate_fixture_manifest_keyset_mismatch(tmp_path):
    d, fixture = _synthetic_fixture_dir(tmp_path, bad_key='keyset')
    (d / 'fixture.json').write_text(json.dumps(fixture))
    with pytest.raises(ValueError, match='fixture file manifest mismatch'):
        gated_norm_ablation.validate_fixture(d, 0, 0)


def test_validate_fixture_file_identity_dtype_mismatch(tmp_path):
    for bad in ('file_identity', 'dtype'):
        d, fixture = _synthetic_fixture_dir(tmp_path, bad_key=bad)
        (d / 'fixture.json').write_text(json.dumps(fixture))
        with pytest.raises(ValueError, match='fixture file identity mismatch'):
            gated_norm_ablation.validate_fixture(d, 0, 0)


def test_validate_fixture_shape_record_mismatch(tmp_path):
    d, fixture = _synthetic_fixture_dir(tmp_path, bad_key='shape')
    (d / 'fixture.json').write_text(json.dumps(fixture))
    with pytest.raises(ValueError, match='fixture file geometry mismatch'):
        gated_norm_ablation.validate_fixture(d, 0, 0)


def test_f32_bits_rejects_non_numeric():
    with pytest.raises(ValueError, match='fixture epsilon must be numeric'):
        gated_norm_ablation._f32_bits('hello')


def test_f32_bits_rejects_non_positive():
    with pytest.raises(ValueError, match='fixture epsilon must be finite and positive'):
        gated_norm_ablation._f32_bits(-1.0)
    with pytest.raises(ValueError, match='fixture epsilon must be finite and positive'):
        gated_norm_ablation._f32_bits(0.0)


def test_f32_bits_rejects_underflow_to_zero():
    # Smallest positive subnormal that packs to bits==0 after f32 rounding
    with pytest.raises(ValueError, match='fixture epsilon underflows f32'):
        gated_norm_ablation._f32_bits(1e-46)


def test_f32_bits_rejects_overflow():
    with pytest.raises(ValueError, match='fixture epsilon overflows f32'):
        gated_norm_ablation._f32_bits(1e40)

def test_summary_controls_engaged_is_per_variant():
    summary = build_summary(0, _synthetic_rows(), {}, _COMPILE)
    for row in summary['rows']:
        for name, record in row['runs'].items():
            assert record['controls_engaged'] is (
                name in ('attested', 'v0-copy', 'v-all', 'v24-minimal'))
    rows = _synthetic_rows()
    rows[0]['runs']['v-all']['controls']['gate']['engaged'] = False
    summary = build_summary(0, rows, {}, _COMPILE)
    assert summary['rows'][0]['runs']['v-all']['controls_engaged'] is False
    assert summary['rows'][0]['runs']['attested']['controls_engaged'] is True


def test_compare_bf16_matches_analyzer_metrics_on_edge_values():
    a = np.array([0x0000, 0x8000, 0x3c00, 0xbc00, 0x0001, 0x8001], dtype='<u2').reshape(1, 1, 6)
    b = np.array([0x8000, 0x0000, 0xbc00, 0x3c00, 0x8001, 0x0001], dtype='<u2').reshape(1, 1, 6)
    # compare_bf16 locates the max with its own distance line and reports the
    # magnitude from analyze_localization.metrics; the two must agree.
    for pair in ((a, b), (b, a), (a[..., :2], b[..., :2]), (a[..., 4:], b[..., 4:])):
        ours = compare_bf16(*pair)
        assert abs(ours['signed_ulp_at_max']) == ours['max_ulp']
        assert ours['max_ulp'] == gated_norm_ablation.metrics(*pair)['max']


def test_nvrtc_options_unchanged():
    from native_ptx import NVRTC_OPTIONS
    assert NVRTC_OPTIONS == ('--std=c++17', '--fmad=false')
