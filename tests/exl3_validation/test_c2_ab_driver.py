# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only tests for the C2 A/B run driver (c2_ab_driver)."""
import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

import collect_logits
import make_norm_fixtures
from c2_ab_driver import (KNOB_ENV, RECORD_ENV, UPSTREAM_ENV, arm_configurations,
                          load_run_config, locate_substitution_records,
                          parse_serve_descriptor, run_ab, validate_plan)
from test_make_norm_fixtures import WIDTH, _schedule, _upstream_fixture
from test_parity import provenance

COLLECT_KEYS = {'endpoint', 'model', 'cases', 'capture_dir', 'destination',
                'provenance', 'timeout', 'thresholds'}


def _descriptor_args(rank):
    return ['docker', 'create', '--name', f'atlas-fixture-{rank}',
            '--env', 'ATLAS_NO_DECODE_GRAPHS=1', '--env', 'ATLAS_NO_DECODE_GRAPHS_MULTISEQ=1',
            '--env', 'ATLAS_DEBUG_NO_GRAPH=1', '--env', 'NCCL_DEBUG=WARN',
            'atlas-exl3-validation:src-8b6b47f14d5b-rust1.97.1', 'serve',
            '/root/model-snapshot', '--tp-size', '2', '--rank', str(rank)]


def _write_descriptors(root):
    paths = {}
    for rank in (0, 1):
        path = root / f'descriptor-rank{rank}.json'
        path.write_text(json.dumps(_descriptor_args(rank)))
        paths[str(rank)] = str(path)
    return paths


def _fixture_dir(root, variant='reference-exact'):
    dirs, _ = _upstream_fixture(root / 'captures')
    out = root / 'fixture'
    make_norm_fixtures.assemble('upstream-dump', dirs, _schedule(root, ['req-0', 'req-1']),
                                variant, out, 'test-engine', 2, WIDTH)
    return out


def _run_config(root):
    contents = {'cases.json': '[{"id": "c"}]', 'thresholds.json': '{"top_k": 10}',
                'prov.json': '{"environment": {}}'}
    files = {}
    for name, content in contents.items():
        path = root / name
        path.write_text(content)
        files[name] = str(path)
    return {'schema': 'atlas-c2-ab-run-config-v1', 'endpoint': 'http://127.0.0.1:8011',
            'model': 'test-model', 'cases_file': files['cases.json'],
            'thresholds_file': files['thresholds.json'],
            'provenance_file': files['prov.json'], 'capture_dir': '/tmp/capture',
            'timeout_seconds': 60, 'rank_descriptors': _write_descriptors(root)}


class FakeCollector:
    """Echo collector plus an engine double: writes substitution records where
    the exported ATLAS_EXL3_NORM_SUBSTITUTE_RECORD_DIR points and gated_norm
    captures where ATLAS_EXL3_UPSTREAM_DUMP_DIR points (the engine writes the
    latter for both arms; run A's copies are the A/B alignment source)."""

    def __init__(self, entries, record_mode='all', vary_calls=(), fail_call=None,
                 gated_mode='match'):
        self.entries, self.record_mode = entries, record_mode
        self.vary_calls, self.fail_call = set(vary_calls), fail_call
        self.gated_mode = gated_mode
        self.calls, self.seen_keys = [], []

    def __call__(self, endpoint, model, cases, capture_dir, destination, provenance,
                 timeout, thresholds):
        call = len(self.calls)
        self.calls.append({'endpoint': endpoint, 'model': model, 'cases': cases,
                           'capture_dir': capture_dir, 'destination': destination,
                           'provenance': provenance, 'timeout': timeout,
                           'thresholds': thresholds})
        self.seen_keys.append(set(self.calls[-1]))
        if self.fail_call == call:
            raise RuntimeError('collector failed')
        destination.mkdir(parents=True)
        suffix = f'-varied-{call}' if call in self.vary_calls else ''
        rows = [{'row_sha256': hashlib.sha256(f'row-{row}{suffix}'.encode()).hexdigest()}
                for row in range(2)]
        manifest = {'logits_sha256': hashlib.sha256(f'logits{suffix}'.encode()).hexdigest(),
                    'captures': rows}
        (destination / 'manifest.json').write_text(json.dumps(manifest))
        record_dir = provenance.get('environment', {}).get(RECORD_ENV)
        if record_dir:
            self._write_records(Path(record_dir))
        upstream = provenance.get('environment', {}).get(UPSTREAM_ENV)
        if upstream:
            self._write_gated_captures(Path(upstream))
        return manifest

    def _write_gated_captures(self, upstream):
        for entry in self.entries:
            if self.gated_mode == 'absent':
                continue
            payload = (f"other-{entry['file']}" if self.gated_mode == 'mismatch'
                       else f"pre-{entry['file']}")
            name = (f"ffn-L0-rank{entry['tp_rank']}-{entry['invocation']:06d}"
                    '-gated_norm-gated.bin')
            (upstream / name).write_bytes(payload.encode())

    def _write_records(self, record_dir):
        for index, entry in enumerate(self.entries):
            if self.record_mode == 'none':
                continue
            if self.record_mode == 'missing' and index == 0:
                continue
            payload_stage = ('pre-substitution' if self.record_mode == 'bad-payload-stage'
                             else 'post-substitution')
            stem = (f"ffn-L0-rank{entry['tp_rank']}-{entry['invocation']:06d}"
                    '-norm_substitution')
            payload_file = stem + '-gated.bin'
            size = math.prod(entry['shape']) * 2
            (record_dir / payload_file).write_bytes(b'\x00' * size)
            record = {
                'schema': 'atlas-exl3-ffn-v1', 'layer': 0, 'tp_rank': entry['tp_rank'],
                'tp_world': 2, 'invocation': entry['invocation'],
                'stage': 'norm_substitution', 'stream': 0,
                'tensors': [{'name': 'gated', 'dtype': 'bf16-le', 'shape': entry['shape'],
                             'bytes': size, 'file': payload_file}],
                'binding': {
                    'schema': 'atlas-norm-substitution-v1', 'payload_stage': payload_stage,
                    'variant': 'reference-exact', 'request_id': entry['request_id'],
                    'invocation': entry['invocation'], 'shape': entry['shape'],
                    'fixture_file': entry['file'], 'fixture_sha256': entry['sha256'],
                    'pre_substitution_sha256': hashlib.sha256(
                        f"pre-{entry['file']}".encode()).hexdigest()}}
            (record_dir / (stem + '.json')).write_text(json.dumps(record))


def _plan(root, attempt='c2-attempt-01', repeats=2, variant='reference-exact'):
    evidence = root / 'evidence'
    evidence.mkdir(exist_ok=True)
    fixture_dir = _fixture_dir(root / 'fixtures', variant)
    run_config = _run_config(root)
    return {'evidence_root': evidence, 'attempt': attempt, 'variant': variant,
            'fixture_dir': fixture_dir, 'repeats': repeats,
            'arm_configs': arm_configurations(run_config, fixture_dir, variant),
            'provenance': provenance('atlas'), 'cases': [{'id': 'c'}],
            'thresholds': {'top_k': 10}}


def _entries(plan):
    return make_norm_fixtures.validate_manifest(plan['fixture_dir'])['entries']


def test_run_ab_preserves_every_attempt_with_full_fingerprints(tmp_path):
    plan = _plan(tmp_path)
    receipt = run_ab(plan, FakeCollector(_entries(plan)))
    attempt = plan['evidence_root'] / plan['attempt']
    assert (attempt / 'receipt.json').is_file() and (attempt / 'plan.json').is_file()
    for arm in ('a', 'b'):
        for repeat in range(2):
            run_name = f'arm-{arm}-repeat-{repeat:02d}'
            assert (attempt / 'runs' / run_name / 'collection' / 'manifest.json').is_file()
            run_receipt = receipt['arms'][arm][repeat]
            for rank in ('0', '1'):
                descriptor = run_receipt['fingerprints']['descriptors'][rank]
                assert descriptor['image'] == 'atlas-exl3-validation:src-8b6b47f14d5b-rust1.97.1'
                assert descriptor['source_snapshot'] == '8b6b47f14d5b'
                assert descriptor['command'][0] == 'serve'
            assert run_receipt['collection']['logits_sha256']
    assert receipt['causality']['status'] == 'byte-determinism-confirmed'
    assert all(receipt['byte_determinism'][arm]['byte_deterministic'] for arm in ('a', 'b'))


def test_collector_is_wrapped_with_the_collect_logits_contract(tmp_path):
    plan = _plan(tmp_path)
    collector = FakeCollector(_entries(plan))
    run_ab(plan, collector)
    assert len(collector.calls) == 4
    assert collector.seen_keys == [COLLECT_KEYS] * 4
    assert collector.calls[0]['model'] == 'test-model'
    assert collector.calls[0]['thresholds'] == {'top_k': 10}


def test_existing_attempt_directory_refused_and_never_reused(tmp_path):
    plan = _plan(tmp_path)
    run_ab(plan, FakeCollector(_entries(plan)))
    attempt = plan['evidence_root'] / plan['attempt']
    before = (attempt / 'receipt.json').read_bytes()
    with pytest.raises(ValueError, match='attempt directory must not exist'):
        run_ab(plan, FakeCollector(_entries(plan)))
    assert (attempt / 'receipt.json').read_bytes() == before  # never reused or deleted


def test_insufficient_repeats_refused(tmp_path):
    plan = _plan(tmp_path, repeats=1)
    with pytest.raises(ValueError, match='at least 2 repeats'):
        run_ab(plan, FakeCollector(_entries(plan)))
    assert not (plan['evidence_root'] / plan['attempt']).exists()


def test_mixed_configurations_between_arms_refused(tmp_path):
    for problem in ('endpoint', 'command', 'descriptor-env', 'knob-baked-in'):
        root = tmp_path / problem
        root.mkdir()
        plan = _plan(root)
        arm_a, arm_b = (copy.deepcopy(config) for config in plan['arm_configs'])
        if problem == 'endpoint':
            arm_b['endpoint'] = 'http://127.0.0.1:9999'
        elif problem == 'command':
            arm_b['descriptors']['0']['command'] = ['serve', 'other']
        elif problem == 'descriptor-env':
            arm_b['descriptors']['1']['env']['NCCL_DEBUG'] = 'INFO'
        else:
            arm_a['descriptors']['0']['env'][KNOB_ENV] = '/baked'
        plan['arm_configs'] = (arm_a, arm_b)
        with pytest.raises(ValueError, match='mixed configurations between arms'):
            run_ab(plan, FakeCollector(_entries(plan)))
        assert not (plan['evidence_root'] / plan['attempt']).exists()


def test_arm_a_env_unset_and_arm_b_exports_fresh_record_dir_per_run(tmp_path):
    plan = _plan(tmp_path)
    receipt = run_ab(plan, FakeCollector(_entries(plan)))
    attempt = plan['evidence_root'] / plan['attempt']
    record_dirs = []
    for arm in ('a', 'b'):
        for repeat in range(2):
            run_root = attempt / 'runs' / f'arm-{arm}-repeat-{repeat:02d}'
            for rank in ('0', '1'):
                args = json.loads((run_root / f'create-command-rank{rank}.json').read_text())
                env = {pair.split('=', 1)[0]: pair.split('=', 1)[1]
                       for flag, pair in zip(args, args[1:]) if flag == '--env'}
                if arm == 'a':
                    assert KNOB_ENV not in env and RECORD_ENV not in env  # unset entirely
                else:
                    assert env[KNOB_ENV] == str(plan['fixture_dir'])
                    record_dirs.append(env[RECORD_ENV])
                    assert env[RECORD_ENV].startswith(str(attempt / 'records'))
    assert len(set(record_dirs)) == 2  # repeats never share a records dir
    knob = receipt['arms']['b'][0]['knob']
    assert knob['engaged'] is True and knob['variant'] == 'reference-exact'
    assert receipt['arms']['a'][0]['knob']['engaged'] is False


def test_substitution_receipts_located_and_unconsumed_entries_reported(tmp_path):
    plan = _plan(tmp_path)
    entries = _entries(plan)
    receipt = run_ab(plan, FakeCollector(entries, record_mode='missing'))
    for run in receipt['arms']['b']:
        located = run['substitution_records']
        assert located['records_found'] == len(entries) - 1
        assert located['post_hoc_gap'] is True
        assert [gap['file'] for gap in located['unconsumed_entries']] == [entries[0]['file']]
        assert all(record['payload_stage'] == 'post-substitution'
                   for record in located['records'])
    assert not receipt['arms']['a'][0].get('substitution_records')


def test_substitution_records_complete_means_no_gap(tmp_path):
    plan = _plan(tmp_path)
    receipt = run_ab(plan, FakeCollector(_entries(plan)))
    for run in receipt['arms']['b']:
        assert run['substitution_records']['post_hoc_gap'] is False
        assert run['substitution_records']['unconsumed_entries'] == []
    assert receipt['pre_hash_alignment']['aligned'] is True
    assert receipt['b_repeat_pre_hash_determinism']['runs_compared'] == 2


def test_pre_hash_alignment_matches_run_a_gated_norm_captures(tmp_path):
    plan = _plan(tmp_path)
    receipt = run_ab(plan, FakeCollector(_entries(plan), gated_mode='match'))
    alignment = receipt['pre_hash_alignment']
    assert alignment['source'] == 'run-a-gated-norm-captures'
    assert alignment['aligned'] is True and alignment['a_side_conflicts'] == []
    expected = {(entry['tp_rank'], entry['invocation']) for entry in _entries(plan)}
    assert {(e['tp_rank'], e['invocation']) for e in alignment['entries']} == expected
    for entry in alignment['entries']:
        assert entry['status'] == 'match'
        assert entry['a_side_sha256']
        assert all(digest == entry['a_side_sha256']
                   for digest in entry['b_pre_substitution_sha256'].values())
    assert receipt['b_repeat_pre_hash_determinism']['byte_deterministic'] is True


def test_pre_hash_alignment_mismatch_reported_per_rank_invocation(tmp_path):
    plan = _plan(tmp_path)
    receipt = run_ab(plan, FakeCollector(_entries(plan), gated_mode='mismatch'))
    alignment = receipt['pre_hash_alignment']
    assert alignment['aligned'] is False
    assert alignment['entries'] and all(e['status'] == 'mismatch'
                                        for e in alignment['entries'])
    for entry in alignment['entries']:
        assert entry['a_side_sha256']
        assert all(digest != entry['a_side_sha256']
                   for digest in entry['b_pre_substitution_sha256'].values())


def test_pre_hash_alignment_reports_a_side_capture_absent_without_fallback(tmp_path):
    plan = _plan(tmp_path)
    receipt = run_ab(plan, FakeCollector(_entries(plan), gated_mode='absent'))
    alignment = receipt['pre_hash_alignment']
    assert alignment['aligned'] is False
    assert alignment['entries']
    for entry in alignment['entries']:
        assert entry['status'] == 'a-side-capture-absent'
        assert entry['a_side_sha256'] is None
    # B-repeat determinism is reported separately and must NOT silently
    # stand in for the missing A-side evidence.
    assert receipt['b_repeat_pre_hash_determinism']['byte_deterministic'] is True


def test_wrong_payload_stage_record_refused_and_failure_preserved(tmp_path):
    plan = _plan(tmp_path)
    with pytest.raises(ValueError, match='payload_stage'):
        run_ab(plan, FakeCollector(_entries(plan), record_mode='bad-payload-stage'))
    attempt = plan['evidence_root'] / plan['attempt']
    failure = json.loads((attempt / 'failure.json').read_text())
    assert failure['exception_type'] == 'ValueError'
    assert (attempt / 'runs' / 'arm-a-repeat-00' / 'collection' / 'manifest.json').is_file()


def test_byte_determinism_variation_marks_causality_unresolved(tmp_path):
    plan = _plan(tmp_path)
    receipt = run_ab(plan, FakeCollector(_entries(plan), vary_calls=(3,)))
    assert receipt['byte_determinism']['a']['byte_deterministic'] is True
    assert receipt['byte_determinism']['b']['byte_deterministic'] is False
    varying = receipt['byte_determinism']['b']['varying_runs']
    assert [item['run'] for item in varying] == ['arm-b-repeat-01']
    assert varying[0]['rows_differing'] == [0, 1]
    assert receipt['causality']['status'] == 'unresolved'


def test_collector_failure_preserves_earlier_runs(tmp_path):
    plan = _plan(tmp_path)
    with pytest.raises(RuntimeError, match='collector failed'):
        run_ab(plan, FakeCollector(_entries(plan), fail_call=2))
    attempt = plan['evidence_root'] / plan['attempt']
    assert (attempt / 'failure.json').is_file()
    assert (attempt / 'runs' / 'arm-a-repeat-01' / 'collection' / 'manifest.json').is_file()
    assert (attempt / 'records' / 'arm-a-repeat-00').is_dir()


def test_parse_serve_descriptor_fingerprints_and_fail_closed(tmp_path):
    path = tmp_path / 'descriptor.json'
    path.write_text(json.dumps(_descriptor_args(0)))
    descriptor = parse_serve_descriptor(path)
    assert descriptor['image'] == 'atlas-exl3-validation:src-8b6b47f14d5b-rust1.97.1'
    assert descriptor['source_snapshot'] == '8b6b47f14d5b'
    assert descriptor['command'] == ['serve', '/root/model-snapshot', '--tp-size', '2', '--rank', '0']
    assert descriptor['env']['ATLAS_NO_DECODE_GRAPHS'] == '1'

    args = _descriptor_args(0)
    args[args.index('serve') - 1] = 'atlas-exl3-validation:latest'  # no src-* snapshot
    path.write_text(json.dumps(args))
    with pytest.raises(ValueError, match='source snapshot'):
        parse_serve_descriptor(path)

    args = _descriptor_args(0)
    index = args.index('ATLAS_DEBUG_NO_GRAPH=1')
    del args[index - 1:index + 1]  # drop the --env pair
    path.write_text(json.dumps(args))
    with pytest.raises(ValueError, match='graph interlock'):
        parse_serve_descriptor(path)


def test_load_run_config_contract(tmp_path):
    config = _run_config(tmp_path)
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    loaded = load_run_config(path)
    assert loaded['rank_descriptors'] == config['rank_descriptors']
    bad = dict(config, extra=True)
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='atlas-c2-ab-run-config-v1'):
        load_run_config(path)


def test_locate_substitution_records_rejects_record_without_entry(tmp_path):
    plan = _plan(tmp_path)
    entries = _entries(plan)
    records = tmp_path / 'records'
    records.mkdir()
    stem = 'ffn-L0-rank5-000000-norm_substitution'
    (records / (stem + '-gated.bin')).write_bytes(b'\x00' * (2 * WIDTH * 2))
    record = {
        'schema': 'atlas-exl3-ffn-v1', 'layer': 0, 'tp_rank': 5, 'tp_world': 2,
        'invocation': 0, 'stage': 'norm_substitution', 'stream': 0,
        'tensors': [{'name': 'gated', 'dtype': 'bf16-le', 'shape': [2, WIDTH],
                     'bytes': 2 * WIDTH * 2, 'file': stem + '-gated.bin'}],
        'binding': {'schema': 'atlas-norm-substitution-v1', 'payload_stage': 'post-substitution',
                    'variant': 'reference-exact', 'request_id': 'req-x', 'invocation': 0,
                    'shape': [2, WIDTH], 'fixture_file': 'x.bin',
                    'fixture_sha256': '0' * 64, 'pre_substitution_sha256': '1' * 64}}
    (records / (stem + '.json')).write_text(json.dumps(record))
    with pytest.raises(ValueError, match='without manifest entry'):
        locate_substitution_records(records, entries, 'reference-exact')


def test_driver_default_collector_is_the_existing_collector():
    import inspect
    import c2_ab_driver
    assert c2_ab_driver.run_ab.__defaults__ == (collect_logits.collect,)
    signature = inspect.signature(collect_logits.collect)
    assert set(signature.parameters) == COLLECT_KEYS


def test_main_cli_wiring_runs_the_attempt(tmp_path, monkeypatch):
    import c2_ab_driver
    root = tmp_path / 'cli'
    root.mkdir()
    fixture_dir = _fixture_dir(root / 'fixtures')
    config_path = root / 'config.json'
    config_path.write_text(json.dumps(_run_config(root)))
    evidence = root / 'evidence'
    evidence.mkdir()
    monkeypatch.setattr('sys.argv', [
        'c2_ab_driver.py', '--evidence-root', str(evidence), '--attempt', 'cli-attempt',
        '--variant', 'reference-exact', '--fixture-dir', str(fixture_dir),
        '--config', str(config_path), '--repeats', '2'])
    collector = FakeCollector(make_norm_fixtures.validate_manifest(fixture_dir)['entries'])
    c2_ab_driver.main(collector)
    assert (evidence / 'cli-attempt' / 'receipt.json').is_file()
