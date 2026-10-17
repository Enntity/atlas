# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only tests for C2 fixture assembly (make_norm_fixtures)."""
import json
import struct
from pathlib import Path

import numpy as np
import pytest

import gated_norm_ablation
import make_norm_fixtures
from make_norm_fixtures import (MANIFEST_SCHEMA, assemble, payload_name,
                                validate_manifest)

WIDTH = 8
TOKENS = (2, 3)
REQUESTS = ('req-0', 'req-1')


def _bits(tokens, seed, width=WIDTH):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 0x7800, (tokens, width)).astype('<u2')


def _schedule(root, request_ids):
    path = root / 'schedule.json'
    path.write_text(json.dumps({'schema': 'atlas-norm-run-schedule-v1',
                                'requests': [{'order': i, 'request_id': request_id}
                                             for i, request_id in enumerate(request_ids)]}))
    return path


def _write_upstream(side_dir, rank, invocation, tokens, request_id, value, width=WIDTH):
    stem = f'ffn-L0-rank{rank}-{invocation:06d}-kda-gated-norm'
    filename = stem + '.bin'
    (side_dir / filename).write_bytes(value.tobytes())
    side = {'dtype': 'bf16-le', 'shape': [1, tokens, 2, width // 2], 'file': filename,
            'layer': 0, 'rank': rank, 'invocation': invocation,
            'stage': 'kda-gated-norm', 'source': 'installed-reference-forward-hook',
            'chunk_position': {'request_id': request_id, 'scheduled_tokens': tokens}}
    (side_dir / (stem + '.json')).write_text(json.dumps(side))
    return side


def _upstream_fixture(root, tokens=TOKENS, request_ids=REQUESTS):
    dirs, values = {}, {}
    for rank in (0, 1):
        side_dir = root / f'rank{rank}'
        side_dir.mkdir(parents=True)
        dirs[rank] = side_dir
        for invocation, (count, request_id) in enumerate(zip(tokens, request_ids)):
            value = _bits(count, 100 * rank + invocation)
            values[(rank, invocation)] = value
            _write_upstream(side_dir, rank, invocation, count, request_id, value)
    return dirs, values


def _write_replay_row(row_dir, rank, row, tokens, request_index, expected):
    row_dir.mkdir(parents=True, exist_ok=True)
    arrays = {'core': _bits(tokens, 7 + row), 'gate': _bits(tokens, 8 + row),
              'weight': _bits(1, 9 + row), 'expected': _bits(tokens, 10 + row),
              'reference-expected': expected}
    for name in ('core', 'gate', 'weight'):
        arrays['reference-' + name] = arrays[name].copy()
    manifest = {'schema': 'norm-replay-fixtures-v1', 'rank': rank, 'invocation': row,
                'tokens': tokens,
                'request': {'index': request_index, 'case': 'case', 'step': 0},
                'candidate_epsilon_f32': 1e-5,
                'candidate_epsilon_f32_bits': struct.unpack('<I', struct.pack('<f', 1e-5))[0],
                'reference_epsilon': 1e-5,
                'reference_epsilon_f32_bits': struct.unpack('<I', struct.pack('<f', 1e-5))[0],
                'epsilon_f32_equal': True,
                'equal_operands': {'core': True, 'gate': True, 'weight': True},
                'fixtures': {}}
    for name, value in arrays.items():
        path = row_dir / (name + '.npy')
        np.save(path, value, allow_pickle=False)
        manifest['fixtures'][name] = {'file': name + '.npy', 'shape': list(value.shape),
                                      'dtype': 'bf16-le',
                                      'sha256': make_norm_fixtures._sha(path)}
    (row_dir / 'fixture.json').write_text(json.dumps(manifest))


def _replay_fixture(root, tokens=TOKENS):
    fixtures = root / 'fixtures'
    values = {}
    for rank in (0, 1):
        for row, count in enumerate(tokens):
            value = _bits(count, 200 * rank + row)
            values[(rank, row)] = value
            _write_replay_row(fixtures / f'rank{rank}-row{row}', rank, row, count, row, value)
    return fixtures, values


def _v24_replay(root, tokens=TOKENS, values=None):
    values = values or {(rank, row): _bits(count, 200 * rank + row)
                        for rank in (0, 1) for row, count in enumerate(tokens)}
    for (rank, row), value in values.items():
        row_dir = root / f'v24-rank{rank}' / f'rank{rank}-row{row}'
        row_dir.mkdir(parents=True, exist_ok=True)
        np.save(row_dir / 'v24-minimal.npy', value, allow_pickle=False)
        (row_dir / 'receipt.json').write_text(json.dumps(
            {'schema': 'atlas-gated-norm-ablation-v1', 'rank': rank, 'row': row}))
    return root


def _assemble(tmp_path, tag, layout='upstream-dump', variant='reference-exact',
              replacements=None, request_ids=REQUESTS):
    work = tmp_path / tag
    work.mkdir()
    if layout == 'upstream-dump':
        dirs, values = _upstream_fixture(work / 'captures', TOKENS, request_ids)
        capture_dirs = dict(dirs)
    else:
        fixtures, values = _replay_fixture(work, TOKENS)
        capture_dirs = {0: fixtures, 1: fixtures}
    schedule = _schedule(work, list(request_ids))
    v24 = None
    if variant == 'v24-corrected':
        v24 = _v24_replay(work / 'ablation', TOKENS,
                          replacements if replacements is not None else values)
    out = work / ('fixture-' + variant)
    summary = assemble(layout, capture_dirs, schedule, variant, out, 'test-engine',
                       2, WIDTH, v24)
    return out, values, summary


def test_upstream_dump_round_trip_revalidates_manifest_and_payloads(tmp_path):
    out, values, summary = _assemble(tmp_path, 'a')
    manifest = validate_manifest(out)
    assert manifest['schema'] == MANIFEST_SCHEMA
    assert manifest['variant'] == 'reference-exact'
    assert [entry['invocation'] for entry in manifest['entries']] == [0, 1, 0, 1]
    assert [entry['tp_rank'] for entry in manifest['entries']] == [0, 0, 1, 1]
    for entry in manifest['entries']:
        assert entry['stage'] == 'gated_norm' and entry['tensor'] == 'gated'
        assert entry['dtype'] == 'bf16-le'
        assert entry['shape'] == [TOKENS[entry['invocation']], WIDTH]
        assert entry['file'] == payload_name(entry['invocation'], entry['tp_rank'])
        assert entry['request_id'] == REQUESTS[entry['invocation']]
        raw = (out / entry['file']).read_bytes()
        assert raw == values[(entry['tp_rank'], entry['invocation'])].tobytes()
    assert summary['entries'] == 4
    assert (out / 'provenance.json').is_file()


def test_replay_fixtures_layout_round_trip_reuses_validate_fixture_rows(tmp_path):
    out, values, summary = _assemble(tmp_path, 'a', layout='replay-fixtures')
    manifest = validate_manifest(out)
    assert manifest['entries'][0]['request_id'] == 'req-0'
    for entry in manifest['entries']:
        raw = (out / entry['file']).read_bytes()
        assert raw == values[(entry['tp_rank'], entry['invocation'])].tobytes()
    provenance = json.loads((out / 'provenance.json').read_text())
    assert provenance['capture_layout'] == 'replay-fixtures'
    assert any('fixture.json' in name for name in provenance['source_files'])


def test_reference_and_v24_variants_stay_separate_with_byte_identity_facts(tmp_path):
    ref_out, values, _ = _assemble(tmp_path, 'a', variant='reference-exact')
    v24_out, _, _ = _assemble(tmp_path, 'b', variant='v24-corrected')
    assert ref_out != v24_out
    ref_manifest = validate_manifest(ref_out)
    v24_manifest = validate_manifest(v24_out)
    assert ref_manifest['variant'] == 'reference-exact'
    assert v24_manifest['variant'] == 'v24-corrected'
    provenance = json.loads((v24_out / 'provenance.json').read_text())
    rows = provenance['byte_identity_with_reference_exact']
    assert rows and all(row['byte_identical_to_reference_exact'] for row in rows)
    # A6: identity is reported as a fact; the sets are never collapsed.
    assert (ref_out / payload_name(0, 0)).read_bytes() == (v24_out / payload_name(0, 0)).read_bytes()
    assert ref_manifest['entries'][0]['sha256'] == v24_manifest['entries'][0]['sha256']


def test_v24_nonidentical_values_reported_not_collapsed(tmp_path):
    work = tmp_path / 'b'
    work.mkdir()
    dirs, values = _upstream_fixture(work / 'captures')
    replacements = dict(values)
    replacements[(0, 1)] = _bits(3, 999)
    v24 = _v24_replay(work / 'ablation', TOKENS, replacements)
    out = work / 'fixture-v24-corrected'
    assemble('upstream-dump', dirs, _schedule(work, list(REQUESTS)), 'v24-corrected', out,
             'test-engine', 2, WIDTH, v24)
    provenance = json.loads((out / 'provenance.json').read_text())
    rows = {(row['tp_rank'], row['invocation']): row
            for row in provenance['byte_identity_with_reference_exact']}
    assert rows[(0, 1)]['byte_identical_to_reference_exact'] is False
    assert rows[(0, 0)]['byte_identical_to_reference_exact'] is True
    assert validate_manifest(out)['variant'] == 'v24-corrected'


def test_validate_manifest_rejects_bad_payload_hash(tmp_path):
    out, _, _ = _assemble(tmp_path, 'a')
    payload = bytearray((out / payload_name(0, 0)).read_bytes())
    payload[-1] ^= 0x01
    (out / payload_name(0, 0)).write_bytes(bytes(payload))
    with pytest.raises(ValueError, match='hash mismatch'):
        validate_manifest(out)


def test_validate_manifest_rejects_shape_mismatch(tmp_path):
    out, _, _ = _assemble(tmp_path, 'a')
    manifest = json.loads((out / 'manifest.json').read_text())
    manifest['entries'][0]['shape'] = [999, WIDTH]
    (out / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='byte count does not match shape'):
        validate_manifest(out)


def test_validate_manifest_rejects_duplicate_entries(tmp_path):
    out, _, _ = _assemble(tmp_path, 'a')
    manifest = json.loads((out / 'manifest.json').read_text())
    manifest['entries'].append(dict(manifest['entries'][0]))
    (out / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=r'duplicate \(layer, tp_rank, invocation, tensor\)'):
        validate_manifest(out)


def test_upstream_captures_reject_duplicate_declared_identity(tmp_path):
    dirs, _ = _upstream_fixture(tmp_path / 'captures')
    value = _bits(2, 55)
    stem = 'ffn-L0-rank0-000001-kda-gated-norm'
    (dirs[0] / (stem + '.bin')).write_bytes(value.tobytes())
    (dirs[0] / (stem + '.json')).write_text(json.dumps(
        {'dtype': 'bf16-le', 'shape': [1, 2, 2, WIDTH // 2], 'file': stem + '.bin',
         'layer': 0, 'rank': 0, 'invocation': 0, 'stage': 'kda-gated-norm',
         'source': 'installed-reference-forward-hook',
         'chunk_position': {'request_id': 'req-0', 'scheduled_tokens': 2}}))
    schedule = _schedule(tmp_path, list(REQUESTS))
    with pytest.raises(ValueError, match='duplicate'):
        assemble('upstream-dump', dirs, schedule, 'reference-exact', tmp_path / 'out',
                 'test-engine', 2, WIDTH)


def test_schedule_mismatch_refuses_to_emit(tmp_path):
    work = tmp_path / 'a'
    work.mkdir()
    dirs, _ = _upstream_fixture(work / 'captures', request_ids=('req-1', 'req-0'))
    schedule = _schedule(work, list(REQUESTS))
    out = work / 'out'
    with pytest.raises(ValueError, match='mismatch'):
        assemble('upstream-dump', dirs, schedule, 'reference-exact', out, 'test-engine', 2, WIDTH)
    assert not out.exists()

    work2 = tmp_path / 'b'
    work2.mkdir()
    fixtures, _ = _replay_fixture(work2)
    _write_replay_row(work2 / 'fixtures' / 'rank0-row0', 0, 0, 2, 1, _bits(2, 3))
    with pytest.raises(ValueError, match='request order/invocation mismatch'):
        assemble('replay-fixtures', {0: fixtures, 1: fixtures}, schedule, 'reference-exact',
                 out, 'test-engine', 2, WIDTH)
    assert not out.exists()


def test_missing_request_id_refuses_to_emit(tmp_path):
    work = tmp_path / 'a'
    work.mkdir()
    dirs, _ = _upstream_fixture(work / 'captures')
    sidecar = dirs[0] / 'ffn-L0-rank0-000000-kda-gated-norm.json'
    side = json.loads(sidecar.read_text())
    del side['chunk_position']['request_id']
    sidecar.write_text(json.dumps(side))
    out = work / 'out'
    with pytest.raises(ValueError, match='missing request id'):
        assemble('upstream-dump', dirs, _schedule(work, list(REQUESTS)), 'reference-exact',
                 out, 'test-engine', 2, WIDTH)
    assert not out.exists()

    _upstream_fixture(work / 'captures2')
    long_schedule = _schedule(work, ['req-0', 'req-1', 'req-2'])
    with pytest.raises(ValueError, match='do not cover the schedule'):
        assemble('upstream-dump', {0: work / 'captures2' / 'rank0',
                                   1: work / 'captures2' / 'rank1'},
                 long_schedule, 'reference-exact', out, 'test-engine', 2, WIDTH)
    assert not out.exists()


def test_unknown_capture_request_id_refused(tmp_path):
    work = tmp_path / 'a'
    work.mkdir()
    dirs, _ = _upstream_fixture(work / 'captures')
    with pytest.raises(ValueError, match='not present in run schedule'):
        assemble('upstream-dump', dirs, _schedule(work, ['req-0']), 'reference-exact',
                 work / 'out', 'test-engine', 2, WIDTH)


def test_existing_output_directory_refused(tmp_path):
    dirs, _ = _upstream_fixture(tmp_path / 'captures')
    schedule = _schedule(tmp_path, list(REQUESTS))
    out = tmp_path / 'out'
    out.mkdir()
    with pytest.raises(ValueError, match='must not exist'):
        assemble('upstream-dump', dirs, schedule, 'reference-exact', out, 'test-engine', 2, WIDTH)


def test_manifest_key_drift_rejected(tmp_path):
    out, _, _ = _assemble(tmp_path, 'a')
    manifest = json.loads((out / 'manifest.json').read_text())
    manifest['extra'] = True
    (out / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='schema mismatch'):
        validate_manifest(out)


def test_v24_requires_replay_root_exactly_for_variant(tmp_path):
    dirs, _ = _upstream_fixture(tmp_path / 'captures')
    schedule = _schedule(tmp_path, list(REQUESTS))
    with pytest.raises(ValueError, match='v24-replay-root'):
        assemble('upstream-dump', dirs, schedule, 'v24-corrected', tmp_path / 'out',
                 'test-engine', 2, WIDTH)
    _v24_replay(tmp_path / 'ablation')
    with pytest.raises(ValueError, match='v24-replay-root'):
        assemble('upstream-dump', dirs, schedule, 'reference-exact', tmp_path / 'out',
                 'test-engine', 2, WIDTH, tmp_path / 'ablation')


def test_validate_fixture_seam_rejects_tampered_replay_input(tmp_path):
    work = tmp_path / 'a'
    work.mkdir()
    fixtures, _ = _replay_fixture(work)
    row_dir = fixtures / 'rank0-row0'
    manifest = json.loads((row_dir / 'fixture.json').read_text())
    manifest['fixtures']['reference-expected']['sha256'] = '0' * 64
    (row_dir / 'fixture.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='hash mismatch'):
        assemble('replay-fixtures', {0: fixtures, 1: fixtures},
                 _schedule(work, list(REQUESTS)), 'reference-exact',
                 work / 'out', 'test-engine', 2, WIDTH)


def test_validate_fixture_seam_is_the_reused_validator():
    assert make_norm_fixtures.gated_norm_ablation is gated_norm_ablation


def test_main_cli_accepts_rank_path_capture_dirs(tmp_path, monkeypatch):
    dirs, _ = _upstream_fixture(tmp_path / 'captures')
    schedule = _schedule(tmp_path, list(REQUESTS))
    out = tmp_path / 'out'
    monkeypatch.setattr('sys.argv', [
        'make_norm_fixtures.py', '--capture-layout', 'upstream-dump',
        '--capture-dir', f'0={dirs[0]}', '--capture-dir', f'1={dirs[1]}',
        '--schedule', str(schedule), '--variant', 'reference-exact', '--out', str(out),
        '--engine', 'cli-test', '--tp-world', '2', '--gated-width', str(WIDTH)])
    make_norm_fixtures.main()
    assert validate_manifest(out)['variant'] == 'reference-exact'
