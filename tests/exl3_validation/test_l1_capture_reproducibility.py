# SPDX-License-Identifier: AGPL-3.0-only
import hashlib
import json
import pytest
from l1_capture_reproducibility import compare

NAME = 'ffn-L1-rank0-000003-kda-o-proj'


def window(root, suffix, payload=b'\x01\x02', extra=None, pids=('1',)):
    for rank in ('0', '1'):
        directory = root/('rank'+rank)/'diagnostic'
        directory.mkdir(parents=True)
        (directory/'capture-enabled').write_text('')
        name = 'ffn-L1-rank'+rank+'-000003-kda-o-proj'
        (directory/(name+'.bin')).write_bytes(payload)
        # Rank 0 also arms the logits capture; rank 1 only the diagnostic one (as collected).
        markers = [str(root/'logits'/'capture-enabled')] if rank == '0' else []
        sidecar = {'dtype': 'bf16-le', 'shape': [1], 'file': name+'.bin', 'layer': 1, 'rank': int(rank),
                   'invocation': 3, 'stage': 'kda-o-proj',
                   'chunk_position': {'request_id': 'cmpl-c2-20261014-req-00-0-'+suffix, 'chunk_counter': 0,
                                      'active_markers': markers+[str(root/'diagnostic'/'capture-enabled')]}}
        sidecar.update(extra or {})
        (directory/(name+'.json')).write_text(json.dumps(sidecar))
        for pid in pids:
            (directory/('hook-source-vllm.models.glm5next.nvidia.model-'+pid+'.json')).write_text(
                json.dumps({'module': 'vllm.models.glm5next.nvidia.model', 'sha256': 'a'*64}))
    return root


def receipt(root):
    """Collection receipt listing every rank diagnostic file of the window, as l1_collect writes it."""
    files = {str(p.relative_to(root.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(root.rglob('*')) if p.is_file()}
    path = root.parent/'receipt.json'
    path.write_text(json.dumps({'file_sha256': files}))
    return path


def run(base, new):
    return compare(base, new, receipt(base), receipt(new))


def test_identical_captures_differing_only_in_per_run_fields_pass(tmp_path):
    report = run(window(tmp_path/'a'/'reference-observed', '9855b8fd'),
                 window(tmp_path/'b'/'reference-observed', '0123abcd', pids=('183',)))
    assert report['passed']
    assert report['ranks']['0']['payloads_compared'] == 1
    assert report['receipt_verified_files'] == {'baseline': 8, 'new': 8}


@pytest.mark.parametrize('case', ['payload', 'sidecar', 'type', 'marker-set', 'missing', 'extra',
                                  'hook-source', 'hook-count'])
def test_any_other_difference_fails_and_is_named(tmp_path, case):
    base = window(tmp_path/'a'/'reference-observed', '9855b8fd')
    new = window(tmp_path/'b'/'reference-observed', '0123abcd',
                 payload=b'\x01\x03' if case == 'payload' else b'\x01\x02',
                 extra={'shape': [2]} if case == 'sidecar' else {'layer': True} if case == 'type' else None,
                 pids=('1', '2') if case == 'hook-count' else ('1',))
    if case == 'marker-set':
        path = new/'rank0'/'diagnostic'/(NAME+'.json')
        sidecar = json.loads(path.read_text())
        sidecar['chunk_position']['active_markers'] = [str(new/'diagnostic'/'capture-enabled')]
        path.write_text(json.dumps(sidecar))
    if case == 'missing':
        (new/'rank0'/'diagnostic'/(NAME+'.bin')).unlink()
    if case == 'extra':
        (new/'rank0'/'diagnostic'/'ffn-L1-rank0-000004-kda-o-proj.bin').write_bytes(b'')
    if case == 'hook-source':
        for path in (new/'rank0'/'diagnostic').glob('hook-source-*.json'):
            path.write_text(json.dumps({'module': 'vllm.models.glm5next.nvidia.model', 'sha256': 'b'*64}))
    report = run(base, new)
    assert not report['passed']
    rank = report['ranks']['0']
    if case in ('hook-source', 'hook-count'):
        assert rank['hook_sources_equal'] is False
        return
    expected = {'payload': ('payload_mismatches', NAME+'.bin'), 'sidecar': ('sidecar_mismatches', NAME+'.json'),
                'type': ('sidecar_mismatches', NAME+'.json'), 'marker-set': ('sidecar_mismatches', NAME+'.json'),
                'missing': ('only_in_baseline', NAME+'.bin'),
                'extra': ('only_in_new', 'ffn-L1-rank0-000004-kda-o-proj.bin')}[case]
    assert rank[expected[0]] == [expected[1]]


@pytest.mark.parametrize('case', ['request-id', 'foreign-marker', 'records-marker', 'wrong-rank', 'receipt'])
def test_unregistered_forms_and_unbound_files_are_refused(tmp_path, case):
    base = window(tmp_path/'a'/'reference-observed', '9855b8fd')
    new = window(tmp_path/'b'/'reference-observed', '0123abcd')
    path = new/'rank0'/'diagnostic'/(NAME+'.json')
    sidecar = json.loads(path.read_text())
    if case == 'request-id':
        sidecar['chunk_position']['request_id'] = 'cmpl-c2-20261014-req-00-0-NOTHEX00'
    elif case == 'foreign-marker':
        sidecar['chunk_position']['active_markers'] = [str(base/'logits'/'capture-enabled'),
                                                       str(base/'diagnostic'/'capture-enabled')]
    elif case == 'records-marker':
        sidecar['chunk_position']['active_markers'] = [str(new/'records'/'capture-enabled')]
    path.write_text(json.dumps(sidecar))
    if case == 'wrong-rank':
        (new/'rank0'/'diagnostic'/'ffn-L1-rank1-000009-x.bin').write_bytes(b'')
        (base/'rank0'/'diagnostic'/'ffn-L1-rank1-000009-x.bin').write_bytes(b'')
    receipts = receipt(base), receipt(new)
    if case == 'receipt':
        (new/'rank1'/'diagnostic'/'ffn-L1-rank1-000003-kda-o-proj.bin').write_bytes(b'\x09')
    with pytest.raises(ValueError):
        compare(base, new, *receipts)
