# SPDX-License-Identifier: AGPL-3.0-only
"""Source-shaped L1 checks: actual fused phases, TP-rank and policy bindings."""
import json
import numpy as np
import pytest
import l1_bisect
from l1_test_fixture import runtime_tree, sha, sidecar, write_json

@pytest.mark.parametrize('layer,rank', [(0, 0), (1, 0), (1, 1)])
def test_actual_runtime_layout_is_complete(tmp_path, layer, rank):
    tree = runtime_tree(tmp_path, layer=layer, rank=rank)
    report = l1_bisect.bisect(**tree)
    assert report['analysis_complete']
    for row in report['rows']:
        assert row['earliest_divergent_stage'] is None
        assert row['request']['case'] == 'longer'
        assert row['reference_policy']['attention']['policy']['split_geometry']['n_splits'] == 'unknown'
        assert all((pair['metrics']['exact'] for pair in row['pairs'].values()))
    assert report['rows'][1]['pairs']['ffn_mhc_input']['reference_invocation'] == (3 if layer else 1)
    assert report['inputs']['binding_sha256'] == sha(tree['binding_path'])

def change_gated(tree):
    p = sidecar(tree, 'candidate', 0, 'gated_norm')
    meta = json.loads(p.read_text())
    binary = p.parent / meta['tensors'][0]['file']
    arr = np.fromfile(binary, dtype='<u2')
    arr[0] ^= 1
    arr.tofile(binary)

def test_gated_divergence_and_missing_earlier_block(tmp_path):
    tree = runtime_tree(tmp_path)
    change_gated(tree)
    report = l1_bisect.bisect(**tree)
    assert report['rows'][0]['earliest_divergent_stage'] == 'gated_norm'
    assert report['rows'][0]['pairs']['gated_norm']['metrics']['max'] == 1
    sidecar(tree, 'reference', 0, 'mhc-fused-residual-output').unlink()
    report = l1_bisect.bisect(**tree)
    assert not report['analysis_complete']
    assert report['rows'][0]['earliest_divergent_stage'] is None
    assert report['rows'][0]['attribution_blocked']
    assert not report['rows'][0]['pairs']['gated_norm']['metrics']['exact']

@pytest.mark.parametrize('field,value', [('request_id', 'wrong'), ('scheduled_tokens', 99), ('computed_before', 1), ('runner_source_sha256', '0' * 64)])
def test_reference_request_prefix_source_refused(tmp_path, field, value):
    tree = runtime_tree(tmp_path)
    p = sidecar(tree, 'reference', 0, 'kda-core-output')
    m = json.loads(p.read_text())
    m['chunk_position'][field] = value
    write_json(p, m)
    report = l1_bisect.bisect(**tree)
    assert not report['analysis_complete']
    assert 'error' in report['rows'][0]['pairs']['l1_core']
    assert report['rows'][0]['earliest_divergent_stage'] is None

@pytest.mark.parametrize('field,value', [('request_id', 'wrong'), ('rows', 99), ('sequence_parallel', True), ('hc_eps_bits', 0), ('sinkhorn_iters', 7), ('residual_precision', 'invented')])
def test_candidate_policy_refused(tmp_path, field, value):
    tree = runtime_tree(tmp_path)
    p = sidecar(tree, 'candidate', 0, 'attn_mhc_norm')
    m = json.loads(p.read_text())
    m['binding'][field] = value
    write_json(p, m)
    report = l1_bisect.bisect(**tree)
    assert not report['analysis_complete']
    assert report['rows'][0]['binding_errors']
    assert report['rows'][0]['earliest_divergent_stage'] is None

@pytest.mark.parametrize('mutation', ['missing', 'rank', 'layer', 'norm_eps', 'split_reason'])
def test_reference_policy_required_and_bound(tmp_path, mutation):
    tree = runtime_tree(tmp_path)
    p = sidecar(tree, 'reference', 1, 'mhc-fused-policy')
    if mutation == 'missing':
        p.unlink()
    else:
        m = json.loads(p.read_text())
        if mutation in ('rank', 'layer'):
            m[mutation] = 9
        elif mutation == 'norm_eps':
            m['policy']['norm_eps'] = 0.125
        else:
            del m['policy']['split_geometry']['n_splits_reason']
        write_json(p, m)
    report = l1_bisect.bisect(**tree)
    assert not report['analysis_complete']
    assert report['rows'][0]['binding_errors']

def test_prefix_hash_checked_even_with_updated_manifest_pin(tmp_path):
    tree = runtime_tree(tmp_path)
    p = tree['reference_artifact'] / 'manifest.json'
    m = json.loads(p.read_text())
    m['captures'][0]['prefix_sha256'] = '0' * 64
    write_json(p, m)
    registration = json.loads(tree['binding_path'].read_text())
    registration['reference_manifest_sha256'] = sha(p)
    write_json(tree['binding_path'], registration)
    with pytest.raises(ValueError, match='prefix'):
        l1_bisect.bisect(**tree)

def test_manifest_pin_checked(tmp_path):
    tree = runtime_tree(tmp_path)
    p = tree['candidate_artifact'] / 'manifest.json'
    p.write_text(p.read_text() + '\n')
    with pytest.raises(ValueError, match='manifest fingerprint'):
        l1_bisect.bisect(**tree)

def test_source_hook_receipt_checked(tmp_path):
    tree = runtime_tree(tmp_path)
    p = next(tree['reference_root'].glob('hook-source-*.json'))
    m = json.loads(p.read_text())
    m['sha256'] = '0' * 64
    write_json(p, m)
    with pytest.raises(ValueError, match='reference source'):
        l1_bisect.bisect(**tree)

def test_candidate_tp_rank_checked(tmp_path):
    tree = runtime_tree(tmp_path)
    p = sidecar(tree, 'candidate', 0, 'gated_norm')
    m = json.loads(p.read_text())
    m['tp_rank'] = 1
    write_json(p, m)
    with pytest.raises(ValueError, match='sidecar identity'):
        l1_bisect.bisect(**tree)

def test_optional_oproj_input_missing_is_diagnostic(tmp_path):
    tree = runtime_tree(tmp_path)
    sidecar(tree, 'reference', 0, 'kda-o-proj-input0').unlink()
    report = l1_bisect.bisect(**tree)
    assert report['analysis_complete']
    assert 'missing' in report['rows'][0]['pairs']['oproj_input_identity']

def test_cli_preserves_incomplete_report(tmp_path, monkeypatch):
    tree = runtime_tree(tmp_path)
    sidecar(tree, 'reference', 0, 'kda-core-output').unlink()
    out = tmp_path / 'report.json'
    monkeypatch.setattr('sys.argv', ['l1_bisect.py', '--candidate', str(tree['candidate_root']), '--reference', str(tree['reference_root']), '--rank', '0', '--layer', '1', '--rows', '0,1', '--out', str(out), '--binding', str(tree['binding_path']), '--candidate-artifact', str(tree['candidate_artifact']), '--reference-artifact', str(tree['reference_artifact'])])
    with pytest.raises(ValueError, match='missing or errored'):
        l1_bisect.main()
    report = json.loads(out.read_text())
    assert not report['analysis_complete']
    assert report['rows'][0]['earliest_divergent_stage'] is None

@pytest.mark.parametrize('field,value', [('model', 'wrong'), ('prompt', [99]), ('temperature', 0.25)])
def test_actual_http_request_envelope_bound(tmp_path, field, value):
    tree = runtime_tree(tmp_path)
    p = tree['reference_artifact'] / 'responses/longer-000000.json'
    m = json.loads(p.read_text())
    m['request'][field] = value
    write_json(p, m)
    registration = json.loads(tree['binding_path'].read_text())
    registration['reference_head_provenance'][0]['response_sha256'] = sha(p)
    write_json(tree['binding_path'], registration)
    with pytest.raises(ValueError, match='recorded HTTP request'):
        l1_bisect.bisect(**tree)

@pytest.mark.parametrize('failure', ['dtype', 'layout'])
def test_actual_reference_layout_and_dtype_fail_closed(tmp_path, failure):
    tree = runtime_tree(tmp_path)
    p = sidecar(tree, 'reference', 0, 'kda-core-output')
    m = json.loads(p.read_text())
    shape = (2, 3, 32, 128) if failure == 'layout' else (1, 3, 32, 128)
    array = np.zeros(shape, dtype='<u2' if failure == 'layout' else '<f4')
    array.tofile(p.parent / m['file'])
    m['shape'] = list(shape)
    m['dtype'] = 'bf16-le' if failure == 'layout' else 'f32-le'
    write_json(p, m)
    report = l1_bisect.bisect(**tree)
    assert not report['analysis_complete']
    assert 'error' in report['rows'][0]['pairs']['l1_core']
    assert report['rows'][0]['earliest_divergent_stage'] is None

def test_used_tensor_sidecars_and_payloads_are_fingerprinted(tmp_path):
    tree = runtime_tree(tmp_path)
    report = l1_bisect.bisect(**tree)
    p = sidecar(tree, 'candidate', 0, 'gated_norm')
    m = json.loads(p.read_text())
    payload = p.parent / m['tensors'][0]['file']
    hashes = report['inputs']['capture_file_sha256']
    assert hashes[str(p)] == sha(p)
    assert hashes[str(payload)] == sha(payload)
    assert not report['inputs']['unhashed_payloads']

def test_mixed_earliest_divergence_is_preserved(tmp_path):
    tree = runtime_tree(tmp_path)
    change_gated(tree)
    p = sidecar(tree, 'candidate', 1, 'ffn_mhc_norm')
    m = json.loads(p.read_text())
    payload = p.parent / m['tensors'][0]['file']
    arr = np.fromfile(payload, dtype='<u2')
    arr[0] ^= 1
    arr.tofile(payload)
    report = l1_bisect.bisect(**tree)
    assert report['earliest_divergent_stage_by_row'] == {'0': 'gated_norm', '1': 'ffn_mhc_input'}
    assert report['divergence_stages_mixed']

def test_unrepresentable_policy_keeps_incomplete_report(tmp_path):
    tree = runtime_tree(tmp_path)
    p = sidecar(tree, 'reference', 0, 'mhc-fused-policy')
    m = json.loads(p.read_text())
    m['policy']['rms_eps'] = 1e+300
    write_json(p, m)
    report = l1_bisect.bisect(**tree)
    assert not report['analysis_complete'] and report['rows'][0]['binding_errors']

@pytest.mark.parametrize('rows', [[], [-1], [0, 0], [2]])
def test_unregistered_or_invalid_rows_refused(tmp_path, rows):
    tree = runtime_tree(tmp_path)
    tree['rows'] = rows
    with pytest.raises(ValueError):
        l1_bisect.bisect(**tree)


def repin_head(tree, index=0):
    registration = json.loads(tree['binding_path'].read_text())
    record = registration['reference_head_provenance'][index]
    from pathlib import Path
    record['head_position_sha256'] = sha(Path(record['head_position_file']))
    write_json(tree['binding_path'], registration)


def test_observed_suffix_gpu_id_and_head_bytes_are_bound(tmp_path):
    tree = runtime_tree(tmp_path)
    report = l1_bisect.bisect(**tree)
    request = report['rows'][0]['request']['reference_request']
    assert request['response_id'] == 'cmpl-reference-req-0'
    assert request['gpu_request_id'] == 'cmpl-reference-req-0-0-00000000'
    registration = json.loads(tree['binding_path'].read_text())
    head = registration['reference_head_provenance'][0]
    for field in ('head_position_file', 'raw_sidecar_file'):
        from pathlib import Path
        path = Path(head[field])
        assert report['inputs']['file_sha256'][str(path)] == sha(path)


@pytest.mark.parametrize('field', ['head_position_sha256', 'raw_sidecar_sha256', 'response_sha256'])
def test_head_http_mapping_fingerprint_refused(tmp_path, field):
    tree = runtime_tree(tmp_path)
    registration = json.loads(tree['binding_path'].read_text())
    registration['reference_head_provenance'][0][field] = '0' * 64
    write_json(tree['binding_path'], registration)
    with pytest.raises(ValueError, match='fingerprint differs'):
        l1_bisect.bisect(**tree)


@pytest.mark.parametrize('gpu_id', ['cmpl-reference-req-0', 'cmpl-reference-req-0-1-00000000',
                                   'cmpl-reference-req-0-0-000000000', 'cmpl-reference-req-0-0-ZZ000000',
                                   'cmpl-reference-req-01-0-00000000'])
def test_wrong_response_prompt_index_or_random_suffix_refused(tmp_path, gpu_id):
    tree = runtime_tree(tmp_path)
    from pathlib import Path
    registration = json.loads(tree['binding_path'].read_text())
    p = Path(registration['reference_head_provenance'][0]['head_position_file'])
    m = json.loads(p.read_text())
    m['context']['request_id'] = gpu_id
    write_json(p, m)
    repin_head(tree)
    with pytest.raises(ValueError, match='GPU request ID'):
        l1_bisect.bisect(**tree)


def test_different_valid_random_id_is_not_accepted_by_prefix(tmp_path):
    tree = runtime_tree(tmp_path)
    from pathlib import Path
    registration = json.loads(tree['binding_path'].read_text())
    p = Path(registration['reference_head_provenance'][0]['head_position_file'])
    m = json.loads(p.read_text())
    m['context']['request_id'] = 'cmpl-reference-req-0-0-ffffffff'
    write_json(p, m)
    repin_head(tree)
    report = l1_bisect.bisect(**tree)
    assert not report['analysis_complete']
    assert 'error' in report['rows'][0]['pairs']['l1_core']
    assert report['rows'][0]['earliest_divergent_stage'] is None


@pytest.mark.parametrize('mutation', ['head_capture', 'raw_payload', 'row_sha', 'raw_metadata'])
def test_raw_head_provenance_consistency_refused(tmp_path, mutation):
    tree = runtime_tree(tmp_path)
    from pathlib import Path
    registration = json.loads(tree['binding_path'].read_text())
    head = registration['reference_head_provenance'][0]
    p = Path(head['head_position_file'])
    if mutation == 'head_capture':
        m = json.loads(p.read_text()); m['head_capture'] = 'logits-000001'; write_json(p, m)
        repin_head(tree)
    elif mutation == 'raw_payload':
        p = Path(head['raw_sidecar_file']).with_suffix('.f32')
        data = bytearray(p.read_bytes()); data[0] ^= 1; p.write_bytes(data)
    elif mutation == 'row_sha':
        p = tree['reference_artifact'] / 'manifest.json'
        m = json.loads(p.read_text()); m['captures'][0]['row_sha256'] = '0' * 64; write_json(p, m)
        registration['reference_manifest_sha256'] = sha(p); write_json(tree['binding_path'], registration)
    else:
        p = Path(head['raw_sidecar_file']); m = json.loads(p.read_text()); m['vocab'] = 64; write_json(p, m)
        head['raw_sidecar_sha256'] = sha(p); write_json(tree['binding_path'], registration)
    with pytest.raises(ValueError):
        l1_bisect.bisect(**tree)


@pytest.mark.parametrize('field', ['head_position_file', 'raw_sidecar_file'])
def test_head_provenance_paths_must_be_absolute(tmp_path, field):
    tree = runtime_tree(tmp_path)
    registration = json.loads(tree['binding_path'].read_text())
    from pathlib import Path
    registration['reference_head_provenance'][0][field] = Path(registration['reference_head_provenance'][0][field]).name
    write_json(tree['binding_path'], registration)
    with pytest.raises(ValueError, match='absolute head provenance path'):
        l1_bisect.bisect(**tree)
