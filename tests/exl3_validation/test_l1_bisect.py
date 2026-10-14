# SPDX-License-Identifier: AGPL-3.0-only
"""CPU tests for the L1 earliest-divergence bisection driver (D3)."""
import json
from pathlib import Path

import numpy as np
import pytest

import l1_bisect
from l1_bisect import bisect

RANK = 0
LAYER = 1


def _arr(shape, seed):
    # finite positive BF16 bit patterns only (metrics rejects nonfinite)
    rng = np.random.default_rng(seed)
    return rng.integers(0, 0x7F00, size=shape, dtype=np.uint16).astype('<u2')


def _write_candidate(root, rows):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for row, stages in rows.items():
        for stage, tensors in stages.items():
            stem = 'ffn-L%d-rank%d-%06d-%s' % (LAYER, RANK, row, stage)
            entries = []
            for name, array in tensors.items():
                file = stem + '-' + name + '.bin'
                array.tofile(root / file)
                entries.append({'name': name, 'dtype': 'bf16-le',
                                'shape': list(array.shape), 'file': file})
            (root / (stem + '.json')).write_text(json.dumps(
                {'stage': stage, 'layer': LAYER, 'rank': RANK,
                 'invocation': row, 'tensors': entries}))


def _write_reference(root, rows, drop=()):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for row, stages in rows.items():
        for stage, array in stages.items():
            if (row, stage) in drop:
                continue
            stem = 'ffn-L%d-rank%d-%06d-%s' % (LAYER, RANK, row, stage)
            file = stem + '.bin'
            array.tofile(root / file)
            (root / (stem + '.json')).write_text(json.dumps(
                {'stage': stage, 'layer': LAYER, 'rank': RANK,
                 'invocation': row, 'dtype': 'bf16-le',
                 'shape': list(array.shape), 'file': file}))


def _drop_reference(root, row, stage):
    stem = 'ffn-L%d-rank%d-%06d-%s' % (LAYER, RANK, row, stage)
    (Path(root) / (stem + '.json')).unlink()
    (Path(root) / (stem + '.bin')).unlink()


def _trees(tmp_path, seed=7, rows=(0, 1)):
    candidate, reference = {}, {}
    for row in rows:
        core = _arr((2, 32, 128), seed + 40 + row)
        shared = {
            'incoming_highway': {'highway': _arr((3, 2, 2), seed + row)},
            'attn_mhc_norm': {
                'normed': _arr((3, 2), seed + 10 + row),
                'highway': _arr((3, 2, 2), seed + row),
                'post': _arr((3, 2), seed + 20 + row),
                'comb': _arr((3, 2, 2), seed + 30 + row)},
            'recurrent_output': {'core': core},
            'gated_norm': {'gated': _arr((3, 4), seed + 50 + row)},
            'attention_reduced_output': {'output': _arr((3, 2), seed + 60 + row)},
            'ffn_mhc_norm': {'normed': _arr((3, 2), seed + 70 + row)},
        }
        candidate[row] = shared
        reference[row] = {
            'mhc-pre-input-x': shared['incoming_highway']['highway'].copy(),
            'mhc-pre-post': shared['attn_mhc_norm']['post'].copy(),
            'mhc-pre-comb': shared['attn_mhc_norm']['comb'].copy(),
            'mhc-pre-layer-input': shared['attn_mhc_norm']['normed'].copy(),
            # reference emits the singleton-batch KDA core layout
            'kda-core-output': core[None].copy(),
            'kda-gated-norm': shared['gated_norm']['gated'].copy(),
            'kda-o-proj': shared['attention_reduced_output']['output'].copy(),
            'mhc-fused-layer-input': shared['ffn_mhc_norm']['normed'].copy(),
            'kda-o-proj-input0': shared['gated_norm']['gated'].copy(),
        }
    _write_candidate(tmp_path / 'candidate', candidate)
    _write_reference(tmp_path / 'reference', reference)
    return candidate, reference


def test_identical_trees_are_exact_and_complete(tmp_path):
    _trees(tmp_path)
    report = bisect(tmp_path / 'candidate', tmp_path / 'reference',
                    LAYER, RANK, [0, 1])
    assert report['schema'] == 'atlas-l1-bisect-report-v1'
    assert report['claim'].endswith('no gate claim')
    assert report['analysis_complete'] is True
    assert report['incomplete_pairs'] == []
    assert report['divergence_stages_mixed'] is False
    for row in report['rows']:
        assert row['earliest_divergent_stage'] is None
        for name, record in row['pairs'].items():
            assert record['metrics']['exact'] is True


def test_locates_earliest_divergent_stage(tmp_path):
    candidate, reference = _trees(tmp_path)
    for row in (0, 1):
        candidate[row]['gated_norm']['gated'][1, 2] ^= 1  # one BF16 ULP
    _write_candidate(tmp_path / 'candidate', candidate)
    report = bisect(tmp_path / 'candidate', tmp_path / 'reference',
                    LAYER, RANK, [0, 1])
    for row in report['rows']:
        assert row['earliest_divergent_stage'] == 'gated_norm'
        assert row['pairs']['l1_core']['metrics']['exact'] is True
        assert row['pairs']['gated_norm']['metrics']['exact'] is False
        assert row['pairs']['gated_norm']['metrics']['max'] == 1
        # later pairs exact again: the earliest stage still wins
        assert row['pairs']['ffn_mhc_input']['metrics']['exact'] is True


def test_mixed_divergence_stages_reported_as_mixed(tmp_path):
    candidate, reference = _trees(tmp_path)
    candidate[0]['recurrent_output']['core'][0, 0, 0] ^= 1
    candidate[1]['ffn_mhc_norm']['normed'][2, 1] ^= 1
    _write_candidate(tmp_path / 'candidate', candidate)
    report = bisect(tmp_path / 'candidate', tmp_path / 'reference',
                    LAYER, RANK, [0, 1])
    assert report['earliest_divergent_stage_by_row'] == {
        '0': 'l1_core', '1': 'ffn_mhc_input'}
    assert report['divergence_stages_mixed'] is True


def test_reference_layout_mapping_applied_and_unbound_rejected(tmp_path):
    _trees(tmp_path)
    report = bisect(tmp_path / 'candidate', tmp_path / 'reference',
                    LAYER, RANK, [0, 1])
    assert report['analysis_complete'] is True  # (1, 2, 32, 128) mapped
    root = tmp_path / 'reference'
    stem = 'ffn-L%d-rank%d-%06d-%s' % (LAYER, RANK, 0, 'kda-core-output')
    array = np.zeros((3, 2, 32, 128), dtype='<u2')
    array.tofile(root / (stem + '.bin'))
    (root / (stem + '.json')).write_text(json.dumps(
        {'stage': 'kda-core-output', 'layer': LAYER, 'rank': RANK,
         'invocation': 0, 'dtype': 'bf16-le', 'shape': [3, 2, 32, 128],
         'file': stem + '.bin'}))
    report = bisect(tmp_path / 'candidate', root, LAYER, RANK, [0, 1])
    assert report['analysis_complete'] is False
    assert report['incomplete_pairs'] == ['row0.l1_core']
    assert 'unbound KDA layout' in report['rows'][0]['pairs']['l1_core']['error']


def test_missing_required_pair_fails_but_diagnostic_never_does(tmp_path):
    _trees(tmp_path)
    _drop_reference(tmp_path / 'reference', 0, 'kda-o-proj-input0')
    report = bisect(tmp_path / 'candidate', tmp_path / 'reference',
                    LAYER, RANK, [0, 1])
    assert report['analysis_complete'] is True
    assert 'missing' in report['rows'][0]['pairs']['oproj_input_identity']

    _drop_reference(tmp_path / 'reference', 0, 'kda-gated-norm')
    report = bisect(tmp_path / 'candidate', tmp_path / 'reference', LAYER, RANK, [0, 1])
    assert report['analysis_complete'] is False
    assert report['incomplete_pairs'] == ['row0.gated_norm']
    assert report['rows'][0]['pairs']['gated_norm'] == {
        'missing': [None, 'kda-gated-norm']}


def test_dtype_mismatch_recorded_and_not_silently_compared(tmp_path):
    _trees(tmp_path)
    root = tmp_path / 'reference'
    stem = 'ffn-L%d-rank%d-%06d-%s' % (LAYER, RANK, 1, 'kda-core-output')
    array = np.zeros((1, 2, 32, 128), dtype='<f4')
    array.tofile(root / (stem + '.bin'))
    (root / (stem + '.json')).write_text(json.dumps(
        {'stage': 'kda-core-output', 'layer': LAYER, 'rank': RANK,
         'invocation': 1, 'dtype': 'f32-le', 'shape': [1, 2, 32, 128],
         'file': stem + '.bin'}))
    report = bisect(tmp_path / 'candidate', root, LAYER, RANK, [0, 1])
    assert report['analysis_complete'] is False
    assert report['incomplete_pairs'] == ['row1.l1_core']
    assert 'capture dtype differs' in report['rows'][1]['pairs']['l1_core']['error']


def test_rank_checked_rejects_sidecar_identity_mismatch(tmp_path):
    _trees(tmp_path)
    root = tmp_path / 'candidate'
    sidecar = root / ('ffn-L%d-rank%d-%06d-gated_norm.json' % (LAYER, RANK, 0))
    metadata = json.loads(sidecar.read_text())
    metadata['rank'] = 1
    sidecar.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='sidecar identity mismatch'):
        bisect(root, tmp_path / 'reference', LAYER, RANK, [0, 1])


def test_reference_policy_record_travels_with_the_row(tmp_path):
    _trees(tmp_path)
    root = tmp_path / 'reference'
    (root / ('ffn-L%d-rank%d-%06d-mhc-pre-policy.json'
             % (LAYER, RANK, 0))).write_text(json.dumps(
        {'policy': {'sinkhorn_repeat': 20, 'hc_pre_eps': 1e-06}}))
    report = bisect(tmp_path / 'candidate', root, LAYER, RANK, [0, 1])
    assert report['rows'][0]['reference_policy']['policy']['sinkhorn_repeat'] == 20
    assert report['rows'][1]['reference_policy'] is None


def test_main_writes_report_and_refuses_existing_out(tmp_path):
    _trees(tmp_path)
    out = tmp_path / 'report.json'
    l1_bisect.validate_cli(tmp_path / 'candidate', tmp_path / 'reference',
                           RANK, LAYER, out)
    out.write_text('{}')
    with pytest.raises(ValueError, match='must not exist'):
        l1_bisect.validate_cli(tmp_path / 'candidate', tmp_path / 'reference',
                               RANK, LAYER, out)
    for rank in (2, -1):
        with pytest.raises(ValueError):
            l1_bisect.validate_cli(tmp_path / 'candidate', tmp_path / 'reference',
                                   rank, LAYER, tmp_path / 'fresh')
    for layer in (2, -1):
        with pytest.raises(ValueError, match='L0->L1 boundary'):
            l1_bisect.validate_cli(tmp_path / 'candidate', tmp_path / 'reference',
                                   RANK, layer, tmp_path / 'fresh')
    with pytest.raises(ValueError):
        l1_bisect.validate_cli(tmp_path / 'missing', tmp_path / 'reference',
                               RANK, LAYER, tmp_path / 'fresh')
    out.unlink()
    import sys
    old = sys.argv
    sys.argv = ['l1_bisect.py', '--candidate', str(tmp_path / 'candidate'),
                '--reference', str(tmp_path / 'reference'), '--rank', '0',
                '--layer', '1', '--rows', '0,1', '--out', str(out)]
    try:
        l1_bisect.main()
    finally:
        sys.argv = old
    saved = json.loads(out.read_text())
    assert saved['analysis_complete'] is True
    assert saved['rows'][0]['earliest_divergent_stage'] is None


def test_main_writes_report_before_failing_on_incomplete(tmp_path):
    _trees(tmp_path)
    _drop_reference(tmp_path / 'reference', 0, 'kda-gated-norm')
    out = tmp_path / 'report.json'
    import sys
    old = sys.argv
    sys.argv = ['l1_bisect.py', '--candidate', str(tmp_path / 'candidate'),
                '--reference', str(tmp_path / 'reference'), '--rank', '0',
                '--layer', '1', '--rows', '0,1', '--out', str(out)]
    try:
        with pytest.raises(ValueError, match='missing or errored'):
            l1_bisect.main()
    finally:
        sys.argv = old
    saved = json.loads(out.read_text())
    assert saved['analysis_complete'] is False
    assert saved['incomplete_pairs'] == ['row0.gated_norm']
