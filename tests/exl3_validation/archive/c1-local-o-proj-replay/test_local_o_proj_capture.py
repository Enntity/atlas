# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only tests for the C1 local o_proj capture-record format and assembler.

TDD: these tests were recorded failing (module missing) before
`local_o_proj_capture.py` was implemented. They round-trip the
`atlas-local-o-proj-binding-v1` capture record both engines emit, fail closed
on every missing field, and assemble a full five-file binding row that
`local_o_proj_replay.validate_binding` accepts (including the resident
sub-record extension).
"""
import json
from pathlib import Path

import numpy as np
import pytest

import local_o_proj_capture
from local_o_proj_capture import (assemble_binding_row, read_capture_record,
                                  validate_capture_record)
from local_o_proj_replay import BINDING_SCHEMA, PROJECTION_DTYPES, WEIGHT_LAYOUTS
from native_ptx import sha
from test_local_o_proj_replay import _write_norm_row

ENGINE = 'candidate'
RANK, ROW, TOKENS = 0, 0, 2
MODULE = 'model.layers.0.self_attn.o_proj'
WEIGHT_SHAPE = (4096, 4)  # EXL3 candidate geometry: [in, out], layout 'in-out'
RESIDENT_NAME = 'o-proj-resident-L0-rank0.bin'
WEIGHT_NAME = 'o-proj-weight-L0-rank0.npy'


def _projection(layout='in-out'):
    return {'module': MODULE, 'tp_pre_reduce': True, 'weight_layout': layout,
            'input_dtype': 'bf16', 'weight_dtype': 'bf16',
            'accumulator_dtype': 'f32', 'output_dtype': 'bf16'}


def _receipt():
    return {'engine': ENGINE, 'provider': 'exl3-native',
            'dispatch': {'dispatch_path': 'project_exl3 -> launch_local_bf16'},
            'weight_basis': 'exl3-mul1-reference-reconstruct-rne-bf16',
            'placement': {'tp_pre_reduce': True,
                          'capture_point': 'local o_proj GEMM output before the TP reduce'}}


def _binding(**overrides):
    record = {'schema': BINDING_SCHEMA, 'engine': ENGINE, 'rank': RANK,
              'layer': 0, 'invocation': ROW, 'tokens': TOKENS,
              'projection': _projection(),
              'files': {'weight': {'file': WEIGHT_NAME, 'shape': list(WEIGHT_SHAPE),
                                   'dtype': 'bf16-le', 'sha256': 'ab' * 32}},
              'resident': {'form': 'exl3-trellis-mul1', 'file': RESIDENT_NAME,
                           'dtype': 'exl3-shard-pack-v1', 'sha256': 'cd' * 32,
                           'recipe': {'name': 'exl3-mul1-reference-reconstruct-rne-bf16',
                                      'inputs': {'resident_sha256': 'cd' * 32}}}}
    record.update(overrides)
    return record


def _write_capture(tmp_path, *, binding=None, receipt=None, envelope=True,
                   weight_bits=None, resident=b'resident-bytes'):
    capture = tmp_path / 'capture'
    capture.mkdir(parents=True, exist_ok=True)
    weight_bits = weight_bits if weight_bits is not None else (
        np.arange(int(np.prod(WEIGHT_SHAPE)), dtype=np.uint16).reshape(WEIGHT_SHAPE))
    np.save(capture / WEIGHT_NAME, weight_bits, allow_pickle=False)
    binding = _binding() if binding is None else binding
    if isinstance(binding, dict) and 'files' in binding:
        binding['files']['weight']['sha256'] = sha(capture / WEIGHT_NAME)
    (capture / RESIDENT_NAME).write_bytes(resident)
    if isinstance(binding, dict) and 'resident' in binding:
        binding['resident']['sha256'] = sha(capture / RESIDENT_NAME)
        binding['resident']['recipe']['inputs']['resident_sha256'] = \
            binding['resident']['sha256']
    record = {'layer': 0, 'rank': RANK, 'invocation': ROW,
              'stage': 'o_proj_binding', 'binding': binding,
              'receipt': _receipt() if receipt is None else receipt}
    if envelope:
        record = {'schema': 'atlas-exl3-ffn-v1', 'tp_rank': RANK, 'tp_world': 2,
                  'tensors': [], **record}
    path = capture / f'ffn-L0-rank{RANK}-{ROW:06}-o_proj_binding.json'
    path.write_text(json.dumps(record))
    return capture, path, weight_bits


def test_read_and_validate_capture_record_round_trip(tmp_path):
    capture, path, _ = _write_capture(tmp_path)
    parsed = read_capture_record(path, engine=ENGINE, rank=RANK, row=ROW, tokens=TOKENS)
    assert parsed['binding']['schema'] == BINDING_SCHEMA
    assert parsed['binding']['projection'] == _projection()
    assert parsed['receipt']['placement']['tp_pre_reduce'] is True
    # The payload files the record references must hash exactly as recorded.
    entry = parsed['binding']['files']['weight']
    assert entry['shape'] == list(WEIGHT_SHAPE)
    assert sha(capture / entry['file']) == entry['sha256']


def test_capture_record_accepts_candidate_dispatch_nested_provider(tmp_path):
    """The candidate Rust receipt (o_proj_receipt.rs) nests the provider
    under `dispatch` instead of the reference's top-level field; both
    envelope styles must validate, and a receipt with neither must not."""
    candidate_receipt = {
        'entry': 'project_hot', 'decode': False, 'tokens': TOKENS,
        'dispatch': {'provider': 'exl3-native',
                     'dispatch_path': 'project_exl3 -> launch_local_bf16'},
        'placement': {'tp_pre_reduce': True,
                      'capture_point': 'local o_proj GEMM output before the TP reduce'}}
    _, path, _ = _write_capture(tmp_path / 'nested', receipt=candidate_receipt)
    parsed = read_capture_record(path, engine=ENGINE, rank=RANK, row=ROW,
                                 tokens=TOKENS)
    assert parsed['receipt']['dispatch']['provider'] == 'exl3-native'
    broken = dict(candidate_receipt)
    broken['dispatch'] = {'dispatch_path': 'no provider identity'}
    _, bad_path, _ = _write_capture(tmp_path / 'no-provider', receipt=broken)
    with pytest.raises(ValueError):
        read_capture_record(bad_path, engine=ENGINE, rank=RANK, row=ROW,
                            tokens=TOKENS)


@pytest.mark.parametrize('field', [
    'schema', 'engine', 'rank', 'layer', 'invocation', 'tokens', 'projection',
    'files', 'resident'])
def test_validate_capture_record_fails_closed_on_missing_fields(tmp_path, field):
    _, path, _ = _write_capture(tmp_path)
    broken = json.loads(path.read_text())
    del broken['binding'][field]
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError):
        read_capture_record(path, engine=ENGINE, rank=RANK, row=ROW, tokens=TOKENS)


@pytest.mark.parametrize('field', ['module', 'tp_pre_reduce', 'weight_layout',
                                   'input_dtype', 'weight_dtype',
                                   'accumulator_dtype', 'output_dtype'])
def test_validate_capture_record_fails_closed_on_projection_gaps(tmp_path, field):
    _, path, _ = _write_capture(tmp_path)
    broken = json.loads(path.read_text())
    del broken['binding']['projection'][field]
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError):
        read_capture_record(path, engine=ENGINE, rank=RANK, row=ROW, tokens=TOKENS)


@pytest.mark.parametrize('field', ['form', 'file', 'dtype', 'sha256', 'recipe'])
def test_validate_capture_record_fails_closed_on_resident_gaps(tmp_path, field):
    _, path, _ = _write_capture(tmp_path)
    broken = json.loads(path.read_text())
    del broken['binding']['resident'][field]
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError):
        read_capture_record(path, engine=ENGINE, rank=RANK, row=ROW, tokens=TOKENS)


@pytest.mark.parametrize('field', ['file', 'shape', 'dtype', 'sha256'])
def test_validate_capture_record_fails_closed_on_weight_entry_gaps(tmp_path, field):
    _, path, _ = _write_capture(tmp_path)
    broken = json.loads(path.read_text())
    del broken['binding']['files']['weight'][field]
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError):
        read_capture_record(path, engine=ENGINE, rank=RANK, row=ROW, tokens=TOKENS)


@pytest.mark.parametrize('field', ['provider', 'placement'])
def test_validate_capture_record_fails_closed_on_missing_receipt(tmp_path, field):
    _, path, _ = _write_capture(tmp_path)
    broken = json.loads(path.read_text())
    del broken['receipt'][field]
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError):
        read_capture_record(path, engine=ENGINE, rank=RANK, row=ROW, tokens=TOKENS)
    _, path2, _ = _write_capture(tmp_path / 'no-receipt')
    broken = json.loads(path2.read_text())
    del broken['receipt']
    path2.write_text(json.dumps(broken))
    with pytest.raises(ValueError):
        read_capture_record(path2, engine=ENGINE, rank=RANK, row=ROW, tokens=TOKENS)


def test_validate_capture_record_rejects_foreign_identity(tmp_path):
    _, path, _ = _write_capture(tmp_path)
    expected = {'engine': ENGINE, 'rank': RANK, 'row': ROW, 'tokens': TOKENS}
    for override in ({'engine': 'reference'}, {'rank': 1}, {'row': 1},
                     {'tokens': 3}):
        with pytest.raises(ValueError):
            read_capture_record(path, **{**expected, **override})


def test_assemble_binding_row_produces_a_validated_row(tmp_path):
    capture, path, weight_bits = _write_capture(tmp_path)
    fixtures = tmp_path / 'fixtures' / 'rank0-row0'
    arrays = _write_norm_row(fixtures, RANK, ROW, tokens=TOKENS)
    # Own-output: raw BF16 bits with rows == tokens, width == weight out side.
    own = np.full((TOKENS, WEIGHT_SHAPE[1]), 0x3f81, dtype='<u2')
    own_bin = capture / 'ffn-L0-rank0-000000-attention_local_output-output.bin'
    own_bin.write_bytes(own.tobytes())
    v24 = tmp_path / 'v24' / 'v24-minimal.npy'
    v24.parent.mkdir(exist_ok=True)
    v24_bits = arrays['reference-expected'].copy()
    np.save(v24, v24_bits, allow_pickle=False)
    out = tmp_path / 'rows' / 'rank0-row0'
    binding = assemble_binding_row(
        rank=RANK, row=ROW, engine=ENGINE, capture_path=path,
        fixture_row_dir=fixtures, own_output_bin=own_bin, v24_path=v24,
        out_dir=out)
    assert binding['schema'] == BINDING_SCHEMA
    assert set(binding['files']) == {'weight', 'own-output', 'candidate-original',
                                     'reference-exact', 'v24-corrected'}
    # The assembler self-checks through the harness validator; row files match.
    for name, entry in binding['files'].items():
        assert sha(out / entry['file']) == entry['sha256']
        value = np.load(out / entry['file'], allow_pickle=False)
        assert value.dtype.str == '<u2'
        assert list(value.shape) == entry['shape']
    # Resident file lands in the row with its recorded hash.
    assert sha(out / binding['resident']['file']) == binding['resident']['sha256']
    # The GEMM receipt accompanies the record in the row.
    receipt = json.loads((out / 'receipt.json').read_text())
    assert receipt['placement']['tp_pre_reduce'] is True
    # Own-output geometry follows the declared weight layout.
    assert binding['files']['own-output']['shape'] == [TOKENS, WEIGHT_SHAPE[1]]


def test_assemble_binding_row_fails_closed_on_tampered_weight(tmp_path):
    capture, path, _ = _write_capture(tmp_path)
    fixtures = tmp_path / 'fixtures' / 'rank0-row0'
    arrays = _write_norm_row(fixtures, RANK, ROW, tokens=TOKENS)
    own_bin = capture / 'x.bin'
    own_bin.write_bytes(np.zeros((TOKENS, WEIGHT_SHAPE[1]), dtype='<u2').tobytes())
    v24 = tmp_path / 'v24.npy'
    np.save(v24, np.zeros(arrays['expected'].shape, dtype='<u2'), allow_pickle=False)
    # Corrupt the weight bytes after the record was written.
    np.save(capture / WEIGHT_NAME,
            np.ones(WEIGHT_SHAPE, dtype=np.uint16), allow_pickle=False)
    with pytest.raises(ValueError):
        assemble_binding_row(rank=RANK, row=ROW, engine=ENGINE,
                             capture_path=path, fixture_row_dir=fixtures,
                             own_output_bin=own_bin, v24_path=v24,
                             out_dir=tmp_path / 'rows' / 'broken')


def test_assemble_binding_row_rewrites_resident_when_it_is_the_weight(tmp_path):
    # Reference-style capture: resident bytes ARE the weight npy.
    capture = tmp_path / 'capture'
    capture.mkdir(parents=True, exist_ok=True)
    weight_bits = np.arange(int(np.prod(WEIGHT_SHAPE)),
                            dtype=np.uint16).reshape(WEIGHT_SHAPE)
    np.save(capture / WEIGHT_NAME, weight_bits, allow_pickle=False)
    weight_sha = sha(capture / WEIGHT_NAME)
    binding = _binding()
    binding['files']['weight']['sha256'] = weight_sha
    binding['resident'] = {'form': 'bf16-module-parameter', 'file': WEIGHT_NAME,
                           'dtype': 'bf16-le', 'sha256': weight_sha,
                           'recipe': {'name': 'vllm-load-time-exl3-nonrouted-dequant',
                                      'inputs': {'resident_sha256': weight_sha}}}
    path = capture / 'record.json'
    path.write_text(json.dumps({'layer': 0, 'rank': RANK, 'invocation': ROW,
                                'stage': 'o-proj-binding', 'binding': binding,
                                'receipt': _receipt()}))
    fixtures = tmp_path / 'fixtures' / 'rank0-row0'
    arrays = _write_norm_row(fixtures, RANK, ROW, tokens=TOKENS)
    own_bin = capture / 'x.bin'
    own_bin.write_bytes(np.zeros((TOKENS, WEIGHT_SHAPE[1]), dtype='<u2').tobytes())
    v24 = tmp_path / 'v24.npy'
    np.save(v24, np.zeros(arrays['expected'].shape, dtype='<u2'), allow_pickle=False)
    out = tmp_path / 'rows' / 'rank0-row0'
    result = assemble_binding_row(rank=RANK, row=ROW, engine=ENGINE,
                                  capture_path=path, fixture_row_dir=fixtures,
                                  own_output_bin=own_bin, v24_path=v24,
                                  out_dir=out)
    # No duplicate 33 MB copy: the resident entry points at weight.npy.
    assert result['resident']['file'] == 'weight.npy'
    assert sha(out / 'weight.npy') == result['resident']['sha256']
    assert not (out / WEIGHT_NAME).exists()


def test_overlay_builder_matches_the_repo_contract():
    """The reference overlay's record builder must satisfy the same validator."""
    root = Path(local_o_proj_capture.DEFAULT_OVERLAY_DIR)
    overlay = root / 'o_proj_binding_router.py'
    if not overlay.is_file():
        pytest.skip(f'reference overlay unavailable at {overlay}')
    import importlib.util
    spec = importlib.util.spec_from_file_location('o_proj_binding_router', overlay)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    weight_entry = {'file': 'ffn-L0-rank0-000000-o-proj-weight.npy',
                    'shape': [4, 4096], 'dtype': 'bf16-le', 'sha256': 'ab' * 32}
    resident = {'form': 'bf16-module-parameter',
                'file': weight_entry['file'], 'dtype': 'bf16-le',
                'sha256': weight_entry['sha256'],
                'recipe': {'name': 'vllm-load-time-exl3-nonrouted-dequant',
                           'inputs': {'resident_sha256': weight_entry['sha256']}}}
    record = module.build_record(
        rank=0, layer=0, invocation=3, tokens=13,
        projection=module.build_projection(
            module='model.layers.0.self_attn.o_proj', input_dtype='bf16',
            weight_dtype='bf16', out_features=4096, in_features=4096,
            width=4096),
        weight_entry=weight_entry, resident=resident)
    receipt = module.build_receipt(provider='vllm-RowParallelLinear',
                                   capture_point='quant_method.apply return, '
                                   'before RowParallelLinear all-reduce')
    validate_capture_record(record, receipt, engine='reference', rank=0, row=3,
                            tokens=13)
