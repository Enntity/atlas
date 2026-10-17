# SPDX-License-Identifier: AGPL-3.0-only
"""CPU lifecycle checks against recorded transport commands, no GPU/network."""
import io
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import pytest
from c2_ab_driver import run_ab
from c2_remote_adapter import RemoteAdapter
from test_c2_ab_driver import FakeCollector, _entries, _plan

IMAGE = 'sha256:' + 'a' * 64


def config():
    return {'hosts': {'0': ['spark0'], '1': ['spark1']},
            'image_ids': {'0': IMAGE, '1': IMAGE}, 'remote_python': ['python3'],
            'remote_harness_dir': '/remote/tests', 'request_ids': ['req-0', 'req-1'],
            'readiness_command': ['curl', '-fsS', 'http://localhost/health'],
            'readiness_timeout_seconds': 2, 'command_timeout_seconds': 10,
            'provider_checks': {rank: {'cublas': ['attest', '/remote/tests/attest_runtime_maps.py', '{container}'],
                                      'cublaslt': ['attest', '/remote/tests/attest_runtime_maps.py', '{container}']}
                                for rank in ('0', '1')},
            'remote_harness_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                      for name in ('collect_logits.py', 'parity_protocol.py', 'attest_runtime_maps.py')}}


class Transport:
    def __init__(self, fault=None):
        self.commands = []
        self.fault = fault

    def __call__(self, argv, **kwargs):
        args = shlex.split(argv[-1])
        self.commands.append((argv[0], args))
        output, code, stderr = b'', 0, b''
        if args[0] == 'sha256sum':
            name = Path(args[1]).name
            output = (config()['remote_harness_sha256'][name] + '  ' + args[1]).encode()
        elif args[:2] == ['docker', 'inspect']:
            code, stderr = 1, b'Error: No such object' 
        if args[:3] == ['docker', 'image', 'inspect']:
            output = (IMAGE + '\n').encode()
        elif args[:2] == ['docker', 'create']:
            output = (('0' if argv[0] == 'spark0' else '1') * 64 + '\n').encode()
        elif args[:2] == ['tar', '-cf']:
            data = io.BytesIO()
            with tarfile.open(fileobj=data, mode='w'):
                pass
            output = data.getvalue()
        if self.fault and args[:len(self.fault)] == self.fault:
            code = 1
        return subprocess.CompletedProcess(argv, code, output, stderr)


def test_remote_lifecycle_pins_image_waits_then_attests_and_stops_both_ranks(tmp_path):
    plan = _plan(tmp_path)
    transport = Transport()
    adapter = RemoteAdapter(config(), transport)
    run_ab(plan, FakeCollector(_entries(plan)), lifecycle=adapter.lifecycle)
    commands = transport.commands
    creates = [args for _, args in commands if args[:2] == ['docker', 'create']]
    assert len(creates) == 8
    names = [args[args.index('--name') + 1] for args in creates]
    assert len(set(names)) == 8
    assert all(args[args.index('serve') - 1] == IMAGE for args in creates)
    assert all('--rm' not in args for args in creates)
    for rank in ('spark0', 'spark1'):
        assert len([a for h, a in commands if h == rank and a[:2] == ['docker', 'stop']]) == 4
        assert len([a for h, a in commands if h == rank and a[:2] == ['docker', 'wait']]) == 4
    first_health = next(i for i, (_, a) in enumerate(commands) if a[0] == 'curl')
    first_attest = next(i for i, (_, a) in enumerate(commands) if a[0] == 'attest')
    assert first_health < first_attest
    assert all('{container}' not in a[2] for _, a in commands if a[0] == 'attest')
    assert not any(a[:2] == ['docker', 'rm'] for _, a in commands)


def test_collector_failure_stops_and_collects_both_ranks(tmp_path):
    plan = _plan(tmp_path)
    transport = Transport()
    adapter = RemoteAdapter(config(), transport)
    with pytest.raises(RuntimeError, match='collector failed'):
        run_ab(plan, FakeCollector(_entries(plan), fail_call=0), lifecycle=adapter.lifecycle)
    assert [host for host, a in transport.commands if a[:2] == ['docker', 'stop']] == ['spark0', 'spark1']
    assert [host for host, a in transport.commands if a[:2] == ['docker', 'wait']] == ['spark0', 'spark1']
    assert len([a for _, a in transport.commands if a[:2] == ['tar', '-cf']]) == 8


def test_second_rank_start_failure_stops_every_created_container(tmp_path):
    plan = _plan(tmp_path)
    transport = Transport(['docker', 'start'])
    adapter = RemoteAdapter(config(), transport)
    with pytest.raises(RuntimeError, match='command failed'):
        run_ab(plan, FakeCollector(_entries(plan)), lifecycle=adapter.lifecycle)
    assert [host for host, a in transport.commands if a[:2] == ['docker', 'stop']] == ['spark0', 'spark1']
    assert not any(a[0] == 'curl' for _, a in transport.commands)


def test_partial_explicit_config_refused_before_remote_commands():
    cfg = config()
    del cfg['provider_checks']['1']['cublaslt']
    transport = Transport()
    with pytest.raises(ValueError, match='both provider'):
        RemoteAdapter(cfg, transport)
    assert not transport.commands


def test_remote_collector_uses_explicit_schedule_and_recovers_manifest(tmp_path):
    root = tmp_path / 'attempt' / 'runs' / 'arm-a-repeat-00'
    root.mkdir(parents=True)
    manifest = {'logits_sha256': 'a' * 64, 'captures': []}
    class CollectorTransport(Transport):
        def __call__(self, argv, **kwargs):
            result = super().__call__(argv, **kwargs)
            args = shlex.split(argv[-1])
            if args[:2] == ['tar', '-cf'] and args[4].endswith('collection'):
                payload = json.dumps(manifest).encode()
                archive = io.BytesIO()
                with tarfile.open(fileobj=archive, mode='w') as tar:
                    entry = tarfile.TarInfo('manifest.json')
                    entry.size = len(payload)
                    tar.addfile(entry, io.BytesIO(payload))
                result.stdout = archive.getvalue()
            return result
    transport = CollectorTransport()
    adapter = RemoteAdapter(config(), transport)
    adapter.active = {'run_root': root}
    result = adapter.collect(endpoint='http://localhost:8011', model='test-model',
                             cases=[{'id': 'test'}], capture_dir=root / 'logits',
                             destination=root / 'collection', provenance={},
                             timeout=60, thresholds={})
    assert result == manifest
    command = next(a for _, a in transport.commands if a[0] == 'python3')
    assert command[1] == '/remote/tests/collect_logits.py'
    assert command[command.index('--request-ids') + 1] == str(root / 'collector-inputs' / 'request_ids.json')
    assert json.loads((root / 'collector-inputs' / 'request_ids.json').read_text()) == ['req-0', 'req-1']
    assert command[command.index('--capture-dir') + 1] == str(root / 'logits')


def test_cli_dry_run_validates_plan_without_remote_mutation(tmp_path, monkeypatch, capsys):
    from c2_remote_adapter import main
    from test_c2_ab_driver import _run_config
    plan = _plan(tmp_path)
    cfg = _run_config(tmp_path)
    Path(cfg['cases_file']).write_text(json.dumps([{'id': 'c', 'prompt_ids': [1], 'continuation_ids': [2]}]))
    local = tmp_path / 'run-config.json'
    remote = tmp_path / 'remote-config.json'
    local.write_text(json.dumps(cfg))
    remote.write_text(json.dumps(config()))
    def refuse(*args, **kwargs):
        raise AssertionError('dry-run must not execute transport')
    monkeypatch.setattr('subprocess.run', refuse)
    monkeypatch.setattr('sys.argv', ['c2_remote_adapter.py', '--evidence-root', str(plan['evidence_root']),
                                  '--attempt', 'dry-run', '--variant', 'reference-exact',
                                  '--fixture-dir', str(plan['fixture_dir']), '--run-config', str(local),
                                  '--remote-config', str(remote), '--repeats', '2', '--dry-run'])
    main()
    assert json.loads(capsys.readouterr().out)['mode'] == 'dry-run'
    assert not (plan['evidence_root'] / 'dry-run').exists()


def test_ambiguous_create_timeout_inspects_and_stops_only_proven_owned_name(tmp_path):
    plan = _plan(tmp_path)
    class AmbiguousTransport(Transport):
        def __init__(self):
            super().__init__()
            self.created_name = None
        def __call__(self, argv, **kwargs):
            args = shlex.split(argv[-1])
            result = super().__call__(argv, **kwargs)
            if args[:2] == ['docker', 'create']:
                self.created_name = args[args.index('--name') + 1]
                raise subprocess.TimeoutExpired(argv, 10)
            if args[:2] == ['docker', 'inspect'] and args[-1] == self.created_name:
                result.returncode, result.stdout, result.stderr = 0, b'0' * 64, b''
            return result
    transport = AmbiguousTransport()
    adapter = RemoteAdapter(config(), transport)
    with pytest.raises(subprocess.TimeoutExpired):
        run_ab(plan, FakeCollector(_entries(plan)), lifecycle=adapter.lifecycle)
    stops = [(h, a) for h, a in transport.commands if a[:2] == ['docker', 'stop']]
    assert stops == [('spark0', ['docker', 'stop', '0' * 64])]
    root = plan['evidence_root'] / plan['attempt'] / 'runs' / 'arm-a-repeat-00'
    assert any(json.loads(p.read_text()).get('status') == 'transport-failed'
               for p in root.glob('remote-*.json'))


def test_remote_harness_mismatch_fails_before_create(tmp_path):
    plan = _plan(tmp_path)
    class WrongHash(Transport):
        def __call__(self, argv, **kwargs):
            result = super().__call__(argv, **kwargs)
            if shlex.split(argv[-1])[0] == 'sha256sum':
                result.stdout = b'f' * 64
            return result
    transport = WrongHash()
    adapter = RemoteAdapter(config(), transport)
    with pytest.raises(ValueError, match='remote harness hash mismatch'):
        run_ab(plan, FakeCollector(_entries(plan)), lifecycle=adapter.lifecycle)
    assert not any(a[:2] == ['docker', 'create'] for _, a in transport.commands)


def test_invalid_request_ids_fail_before_any_remote_command():
    for ids in (['white space'], ['line\nbreak'], ['nonascii-\u00e9']):
        cfg = dict(config(), request_ids=ids)
        transport = Transport()
        with pytest.raises(ValueError, match='request IDs'):
            RemoteAdapter(cfg, transport)
        assert not transport.commands


def test_reversed_fixture_schedule_rejected_before_execution(tmp_path):
    from c2_remote_adapter import validate_schedule
    import make_norm_fixtures
    plan = _plan(tmp_path)
    plan['cases'] = [{'continuation_ids': [2]}]
    fixture = make_norm_fixtures.validate_manifest(plan['fixture_dir'])
    validate_schedule(plan, fixture, ['req-0', 'req-1'])
    with pytest.raises(ValueError, match='ordered fixture invocations'):
        validate_schedule(plan, fixture, ['req-1', 'req-0'])


def test_preexisting_generated_name_never_becomes_owned_or_stopped(tmp_path):
    plan = _plan(tmp_path)
    class ExistingName(Transport):
        def __call__(self, argv, **kwargs):
            result = super().__call__(argv, **kwargs)
            if shlex.split(argv[-1])[:2] == ['docker', 'inspect']:
                result.returncode, result.stdout, result.stderr = 0, b'e' * 64, b''
            return result
    transport = ExistingName()
    adapter = RemoteAdapter(config(), transport)
    with pytest.raises(ValueError, match='absence could not be proven'):
        run_ab(plan, FakeCollector(_entries(plan)), lifecycle=adapter.lifecycle)
    assert not any(a[:2] in (['docker', 'create'], ['docker', 'stop']) for _, a in transport.commands)


def test_lowercase_docker_absence_response_allows_fresh_owned_container(tmp_path):
    class LowercaseDocker(Transport):
        def __call__(self, argv, **kwargs):
            result = super().__call__(argv, **kwargs)
            args = shlex.split(argv[-1])
            if args[:2] == ['docker', 'inspect'] and result.returncode == 1:
                result.stderr = result.stderr.lower()
            return result
    plan = _plan(tmp_path)
    transport = LowercaseDocker()
    adapter = RemoteAdapter(config(), transport)
    run_ab(plan, FakeCollector(_entries(plan)), lifecycle=adapter.lifecycle)
    assert len([a for _, a in transport.commands if a[:2] == ['docker', 'create']]) == 8
