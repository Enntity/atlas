# SPDX-License-Identifier: AGPL-3.0-only
import numpy as np
import pytest

import analyze_localization as audit


def test_integer_ids_are_not_float_subnormals(tmp_path):
    ids = np.array([[1, 17, 287]], dtype='<i4')
    (tmp_path / 'ids.bin').write_bytes(ids.tobytes())
    meta = {'dtype': 'i32-le', 'shape': [1, 3], 'file': 'ids.bin'}
    actual = audit.tensors((meta, tmp_path / 'sidecar.json'))[0][1]
    assert actual.dtype == np.dtype('<i4')
    np.testing.assert_array_equal(actual, ids)


def test_signed_bf16_order_crosses_zero():
    a = np.array([0x8001, 0x8000, 0], dtype='<u2')
    b = np.array([0x8000, 0, 1], dtype='<u2')
    assert audit.metrics(a, b)['max'] == 1
    assert audit.metrics(a[:1], b[-1:])['max'] == 3


def test_integer_membership_distinguishes_order_and_expert_changes():
    a = np.array([[1, 2], [3, 4]], dtype='<i4')
    b = np.array([[2, 1], [3, 5]], dtype='<i4')
    result = audit.metrics(a, b)
    assert result['kind'] == 'ids'
    assert result['diff_count'] == 3
    assert result['membership_different_rows'] == 1


def test_unknown_dtype_fails_closed(tmp_path):
    (tmp_path / 'raw.bin').write_bytes(b'\0' * 4)
    meta = {'dtype': 'unknown', 'file': 'raw.bin', 'shape': [1]}
    with pytest.raises(ValueError, match='dtype'):
        audit.tensors((meta, tmp_path / 'meta.json'))


def test_layout_requires_explicit_mapping():
    a = np.zeros((2, 4096), dtype='<u2')
    b = np.zeros((1, 2, 32, 128), dtype='<u2')
    with pytest.raises(ValueError, match='shape'):
        audit.metrics(a, b)
    assert audit.metrics(a, audit.kda_layout(b, a.shape))['exact']
    core = np.zeros((2, 32, 128), dtype='<u2')
    assert audit.metrics(core, audit.kda_layout(b, core.shape))['exact']
    with pytest.raises(ValueError, match='layout'):
        audit.kda_layout(np.zeros((2, 16, 256), dtype='<u2'), a.shape)


def test_mapping_uses_post_tp_and_direct_norm_input():
    pairs = {p[0]: p for p in audit.pair_list(0)}
    assert pairs['L3.moE_out'][2] == 'output_after_tp'
    assert 'L0.kda_attn_local' not in pairs
    assert pairs['L0.kda_oproj_input'][2:4] == ('gated_norm', 'gated')


def test_producer_gate_requires_both_tp_inputs():
    def get(rank):
        return np.array([rank], dtype='<u2'), np.array([rank + (rank == 1)], dtype='<u2')
    result = audit.producer_gate(get, (0, 1))
    assert not result['same_input']
    assert result['ranks']['0']['exact']
    assert not result['ranks']['1']['exact']
    with pytest.raises(ValueError, match='rank'):
        audit.producer_gate(get, ())


def test_nonfinite_and_mixed_types_rejected():
    with pytest.raises(ValueError, match='nonfinite'):
        audit.metrics(np.array([np.nan], dtype='<f4'), np.array([0], dtype='<f4'))
    with pytest.raises(ValueError, match='dtype'):
        audit.metrics(np.array([1], dtype='<i4'), np.array([1], dtype='<f4'))
