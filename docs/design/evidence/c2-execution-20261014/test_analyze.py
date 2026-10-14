# SPDX-License-Identifier: AGPL-3.0-only
"""Projection identity and corruption refusal without a GPU."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import analyze


def test_projection_preserves_sources_and_requires_exact_token_cases():
    selected = [{'id': 'arithmetic', 'prompt_ids': [1], 'continuation_ids': [2]}]
    source_cases = selected + [{'id': 'excluded', 'prompt_ids': [3], 'continuation_ids': []}]
    manifest = {'cases': source_cases, 'tokens_sha256': 'full-source-tokens',
                'logits_sha256': 'full-source-logits', 'generated_text': {'arithmetic': ['a', 'b'], 'excluded': ['c']},
                'captures': [{'case': 'arithmetic'}, {'case': 'arithmetic'}, {'case': 'excluded'}]}
    source = {'manifests': [manifest], 'rows': {(0, 'arithmetic'): np.ones((2, 4)),
                                             (0, 'excluded'): np.zeros((1, 4))}}
    result = analyze.project_run(source, selected, {'sha256': {'manifest.json': 'bound'}})
    assert set(result['rows']) == {(0, 'arithmetic')}
    projected = result['manifests'][0]
    assert projected['tokens_sha256'] == analyze.token_sha(selected)
    assert projected['diagnostic_projection']['original_tokens_sha256'] == 'full-source-tokens'
    assert projected['logits_sha256'] == manifest['logits_sha256']
    assert manifest['cases'] == source_cases and manifest['tokens_sha256'] == 'full-source-tokens'
    corrupt = [dict(selected[0], prompt_ids=[2])]
    with pytest.raises(ValueError, match='exactly match'):
        analyze.project_run(source, corrupt, {})


def native_fixture(root):
    archive = root / 'archive'
    archive.mkdir()
    array = np.arange(8, dtype='<u2').reshape(2, 4)
    np.save(archive / 'own-output.npy', array)
    entry = {'file': 'own-output.npy', 'shape': [2, 4], 'dtype': 'bf16-le',
             'sha256': analyze.sha(archive / 'own-output.npy')}
    (archive / 'binding.json').write_text(json.dumps({'schema': 'atlas-local-o-proj-binding-v1', 'engine': 'reference', 'layer': 0,
                                                                  'projection': {'tp_pre_reduce': True},
                                                                  'rank': 0, 'invocation': 0, 'files': {'own-output': entry}}))
    stem = 'ffn-L0-rank0-000000-attention_local_output'
    payload = root / (stem + '-output.bin')
    payload.write_bytes(array.tobytes())
    meta = {'schema': 'atlas-exl3-ffn-v1', 'layer': 0, 'tp_rank': 0, 'tp_world': 2,
            'invocation': 0, 'stage': 'attention_local_output',
            'tensors': [{'name': 'output', 'dtype': 'bf16-le', 'shape': [2, 4],
                         'bytes': 16, 'file': payload.name}]}
    sidecar = root / (stem + '.json')
    sidecar.write_text(json.dumps(meta))
    return archive, payload, sidecar


def test_native_first_measurement_detects_single_bf16_bit_change(tmp_path):
    archive, payload, sidecar = native_fixture(tmp_path)
    args = (sidecar, archive, 'own-output', 'binding.json', 0, 0)
    assert analyze.compare_native(*args)['byte_equal']
    corrupted = bytearray(payload.read_bytes())
    corrupted[0] ^= 1
    payload.write_bytes(corrupted)
    assert not analyze.compare_native(*args)['byte_equal']


def test_native_first_measurement_rejects_dtype_shape_and_archive_corruption(tmp_path):
    archive, payload, sidecar = native_fixture(tmp_path)
    args = (sidecar, archive, 'own-output', 'binding.json', 0, 0)
    original = json.loads(sidecar.read_text())
    wrong_shape = json.loads(sidecar.read_text())
    wrong_shape['tensors'][0]['shape'] = [4, 2]
    sidecar.write_text(json.dumps(wrong_shape))
    with pytest.raises(ValueError, match='shape mismatch'):
        analyze.compare_native(*args)
    wrong_dtype = json.loads(json.dumps(original))
    wrong_dtype['tensors'][0]['dtype'] = 'i32-le'
    sidecar.write_text(json.dumps(wrong_dtype))
    with pytest.raises(ValueError, match='byte count mismatch'):
        analyze.compare_native(*args)
    sidecar.write_text(json.dumps(original))
    with (archive / 'own-output.npy').open('ab') as output:
        output.write(b'corruption')
    with pytest.raises(ValueError, match='fingerprint mismatch'):
        analyze.compare_native(*args)


def test_norm_comparison_accepts_only_explicit_bound_head_flattening(tmp_path):
    archive = tmp_path / 'archive'
    archive.mkdir()
    array = np.arange(8, dtype='<u2').reshape(2, 2, 2)
    np.save(archive / 'expected.npy', array)
    entry = {'file': 'expected.npy', 'shape': [2, 2, 2], 'dtype': 'bf16-le',
             'sha256': analyze.sha(archive / 'expected.npy')}
    binding = {'schema': 'norm-replay-fixtures-v1', 'rank': 0, 'invocation': 0, 'candidate_binding': {'tokens': 2, 'heads': 2, 'dim': 2},
               'fixtures': {'expected': entry}}
    (archive / 'fixture.json').write_text(json.dumps(binding))
    stem = 'ffn-L0-rank0-000000-gated_norm'
    payload = tmp_path / (stem + '-gated.bin')
    payload.write_bytes(array.tobytes())
    meta = {'schema': 'atlas-exl3-ffn-v1', 'layer': 0, 'tp_rank': 0, 'tp_world': 2,
            'invocation': 0, 'stage': 'gated_norm', 'tensors': [
                {'name': 'gated', 'dtype': 'bf16-le', 'shape': [2, 4], 'bytes': 16, 'file': payload.name}]}
    sidecar = tmp_path / (stem + '.json')
    sidecar.write_text(json.dumps(meta))
    args = (sidecar, archive, 'expected', 'fixture.json', 0, 0)
    assert analyze.compare_native(*args)['byte_equal']
    binding['candidate_binding']['heads'] = 1
    binding['candidate_binding']['dim'] = 4
    (archive / 'fixture.json').write_text(json.dumps(binding))
    with pytest.raises(ValueError, match='norm archive/capture shape mismatch'):
        analyze.compare_native(*args)


def test_observer_neutrality_detects_changed_full_vocabulary_row():
    cases = [{'id': 'arithmetic', 'prompt_ids': [1], 'continuation_ids': [2, 3, 4]},
             {'id': 'longer', 'prompt_ids': [5], 'continuation_ids': [6, 7, 8]}]
    manifest = {'cases': cases, 'tokens_sha256': analyze.token_sha(cases), 'vocab': 16,
                'request_policy': {'temperature': 0, 'seed': 42, 'max_tokens': 1},
                'generated_text': {'arithmetic': [''] * 4, 'longer': [''] * 4}}
    arrays = {case['id']: np.arange(64, dtype=np.float32).reshape(4, 16) for case in cases}
    expected = analyze.merge_run([(manifest, arrays)])
    actual = analyze.merge_run([(manifest, arrays)])
    assert all(row['byte_equal'] for row in analyze.compare_final_logits(actual, expected))
    actual['rows'][(0, 'longer')][2, 15] += 0.25
    rows = analyze.compare_final_logits(actual, expected)
    assert [(r['case'], r['row']) for r in rows if not r['byte_equal']] == [('longer', 2)]


def test_registered_runtime_image_cannot_be_replaced_by_same_image_on_both_arms(tmp_path):
    registered = ['run-config.json', 'remote-config.json', 'provenance.json', 'preparation.json',
                  'candidate-rank0-create-command.json', 'candidate-rank1-create-command.json',
                  'thresholds.json', 'tooling/collect_logits.py', 'tooling/parity_protocol.py',
                  'tooling/attest_runtime_maps.py', 'tooling/cublas131.sha256']
    for name in registered:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}')
    config = {'schema': 'atlas-c2-ab-run-config-v1', 'endpoint': 'http://localhost', 'model': 'test',
              'cases_file': str(tmp_path / 'cases.json'), 'thresholds_file': str(tmp_path / 'thresholds.json'),
              'provenance_file': str(tmp_path / 'provenance.json'), 'capture_dir': str(tmp_path / 'capture'),
              'timeout_seconds': 60, 'rank_descriptors': {rank: str(tmp_path / f'candidate-rank{rank}-create-command.json')
                                                       for rank in ('0', '1')}}
    (tmp_path / 'run-config.json').write_text(json.dumps(config))
    (tmp_path / 'provenance.json').write_text(json.dumps({'engine_build': {'commit': 'bound-source',
                                                                        'image': 'unregistered-image'}}))
    binding = {'file_sha256': {name: analyze.sha(tmp_path / name) for name in registered},
               'runtime_source': 'bound-source', 'runtime_image_id': 'registered-image'}
    with pytest.raises(ValueError, match='runtime commit/image mismatch'):
        analyze.validate_live_registration(tmp_path, binding)
    (tmp_path / 'run-config.json').write_text('corrupted')
    with pytest.raises(ValueError, match='registered file fingerprint mismatch'):
        analyze.validate_live_registration(tmp_path, binding)
