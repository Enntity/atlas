# SPDX-License-Identifier: AGPL-3.0-only
"""Real HTTP collectors and phase counters behind a CPU-only remote I/O seam."""
import importlib.util
import io
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest
import prepare
import run_neutrality
from collect_logits import collect
from test_parity import provenance, thresholds
from test_c2_ab_driver import _descriptor_args
from test_c2_remote_adapter import config as remote_config

PHASE = Path(__file__).parent / 'fixtures/collect_phase_logits.py'


def phase_module(path):
    spec = importlib.util.spec_from_file_location('frozen_phase_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def prepared(tmp_path):
    state = {'capture': tmp_path / 'frozen-capture'}
    state['capture'].mkdir()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            capture = state['capture']
            assert (capture / 'capture-enabled').exists()
            index = len(list(capture.glob('logits-*.json')))
            stem = f'logits-{index:06d}'
            data = np.array([[100, 100, 100, 100], [0, 1, 2, 3]], dtype='<f4')
            (capture / (stem + '.f32')).write_bytes(data.tobytes())
            (capture / (stem + '.json')).write_text(json.dumps({'dtype': 'f32-le', 'rows': 2, 'vocab': 4, 'file': stem + '.f32'}))
            response = json.dumps({'choices': [{'text': 'x'}]}).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(response)
    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f'http://127.0.0.1:{server.server_port}'
    cases = [{'id': f'case{i}', 'prompt_ids': [1, 2], 'continuation_ids': [3, 3, 3]} for i in range(4)]
    prov = provenance('atlas')
    prov['native_numerical_policy'] = {'source_identity': prov['engine_build']['source_sha256'], 'kv_cache_effective': 'fp8_g128'}
    profile = tmp_path / 'profile'
    profile.mkdir()
    for rank in (0, 1):
        args = _descriptor_args(rank)
        index = args.index('serve') - 1
        args[index:index] = ['--env', 'ATLAS_EXL3_LOGITS_DIR=/tmp/capture', '--volume', '/old:/tmp/capture']
        (profile / f'cand-rank{rank}-create-command.json').write_text(json.dumps(args))
        if rank == 0:
            prov['serve_flags'] = args[args.index('serve'):]
    collect(endpoint, 'model', cases, state['capture'], tmp_path / 'short16', prov, 5, thresholds())
    snapshot = {'head_start_sequence': 16, 'prior_head_files': {p.name: prepare.sha(p) for p in state['capture'].glob('logits-*')}}
    frozen_tooling = tmp_path / 'frozen-tooling'
    frozen_tooling.mkdir()
    shutil.copyfile(PHASE, frozen_tooling / 'collect_phase_logits.py')
    shutil.copyfile(prepare.REPO / 'tests/exl3_validation/parity_protocol.py', frozen_tooling / 'parity_protocol.py')
    phase_module(frozen_tooling / 'collect_phase_logits.py').collect(endpoint, 'model', cases, state['capture'], tmp_path / 'long', prov, 5, thresholds(), snapshot)
    build = {'schema': 'atlas-exl3-build-manifest-v1', 'git_commit': 'b' * 40,
             'source_snapshot_sha256': 'c' * 64, 'binaries': {'spark': {'sha256': 'd' * 64}}}
    for name, value in [('build.json', build), ('prov.json', prov), ('remote.json', remote_config())]:
        (tmp_path / name).write_text(json.dumps(value))
    shutil.copyfile(prepare.REPO / 'docker/gb10/glm-5.3-flash-exl3/exl3/cublas131.sha256', tmp_path / 'pins')
    prior = tmp_path / 'prior-failed'
    prior.mkdir()
    (prior / 'failure.json').write_text(json.dumps({'schema': 'atlas-c2-ab-failure-v1', 'message': 'before create'}))
    (prior / 'lifecycle.json').write_text(json.dumps({'containers': {}, 'owned_names': {}, 'started': {}, 'teardown_errors': []}))
    cfg = {'schema': 'atlas-c2-neutrality-preparation-config-v1', 'prep_root': str(tmp_path / 'prep'),
           'evidence_root': str(tmp_path), 'image_tag': 'atlas:src-' + 'c' * 12, 'image_id': 'sha256:' + 'a' * 64,
           'runtime_source': 'b' * 40, 'source_snapshot_sha256': 'c' * 64, 'build_manifest': str(tmp_path / 'build.json'),
           'profile_dir': str(profile), 'frozen_short': str(tmp_path / 'short16'), 'frozen_long': str(tmp_path / 'long'),
           'baseline_provenance': str(tmp_path / 'prov.json'), 'remote_template': str(tmp_path / 'remote.json'),
           'phase_collector': str(PHASE), 'pins_file': str(tmp_path / 'pins'), 'attempt': 'neutral01',
           'runtime_binary_path': '/usr/local/bin/spark', 'preserved_failures': [{
               'attempt': str(prior), 'phase': 'failed-before-create', 'failure_sha256': prepare.sha(prior / 'failure.json'),
               'lifecycles': [{'path': str(prior / 'lifecycle.json'), 'sha256': prepare.sha(prior / 'lifecycle.json')}]}]}
    state['config'] = cfg
    plan = prepare.prepare_neutrality(cfg)
    plan_path = Path(cfg['prep_root']) / 'neutrality-plan.json'
    state['capture'] = Path(cfg['prep_root']) / 'neutral01/runs/arm-a-repeat-00/logits'
    try:
        yield plan_path, plan, state
    finally:
        server.shutdown()
        server.server_close()


class CPUTransport:
    def __init__(self, plan, long_failure=False):
        self.plan, self.long_failure = plan, long_failure
        self.commands = []
    def __call__(self, argv, **kwargs):
        args = shlex.split(argv[-1])
        self.commands.append((argv[0], args))
        output, stderr, code = b'', b'', 0
        if args[:3] == ['docker', 'image', 'inspect']:
            output = self.plan['runtime']['image'].encode()
        elif args[:2] == ['docker', 'inspect']:
            code, stderr = 1, b'error: no such object'
        elif args[:2] == ['docker', 'create']:
            mounts = [args[i + 1].split(':')[1] for i, arg in enumerate(args[:-1]) if arg == '--volume']
            assert len(mounts) == len(set(mounts)), 'Docker duplicate mount point'
            output = (('a' if argv[0] == 'spark0' else 'b') * 64).encode()
        elif args[:2] == ['docker', 'wait']:
            output = b'0\n'
        elif args[0] == 'sudo':
            pins = run_neutrality.read_pins(args[-1])
            report = {'target': args[4], 'container': args[4], 'pid': 123, 'maps_source': '/proc/123/maps',
                      'fs_root': '/proc/123/root', 'pins': args[-1], 'ok': True, 'issues': [],
                      'libraries': [{'name': name, 'path': item['path'], 'sha256': item['sha256'],
                                     'pinned_sha256': item['sha256'], 'ok': True} for name, item in pins.items() if name.startswith('libcublas')]}
            output = json.dumps(report).encode()
        elif args[:2] == ['docker', 'exec']:
            output = (self.plan['runtime']['binary_sha256'] + '  /usr/local/bin/spark').encode()
        elif args[0] == 'sha256sum':
            output = (prepare.sha(args[1]) + '  ' + args[1]).encode()
        elif args[0] == 'python3':
            if self.long_failure and args[1].endswith('collect_phase_logits.py'):
                code, stderr = 1, b'injected long failure'
            else:
                return subprocess.run(args, capture_output=True, timeout=kwargs['timeout'])
        elif args[:2] == ['tar', '-cf']:
            source = Path(args[4])
            archive = io.BytesIO()
            with tarfile.open(fileobj=archive, mode='w') as tar:
                for child in sorted(source.iterdir()):
                    tar.add(child, arcname=child.name)
            output = archive.getvalue()
        return subprocess.CompletedProcess(argv, code, output, stderr)


def test_single_lifecycle_real_collectors_preserve_32_rows_and_16_to_31_counter(prepared):
    path, plan, state = prepared
    transport = CPUTransport(plan)
    proof = run_neutrality.run(path, transport)
    assert proof['passed'] and proof['rows_total'] == 32
    commands = [args for _, args in transport.commands]
    short = next(i for i, a in enumerate(commands) if a[0] == 'python3' and len(a) > 1 and a[1].endswith('collect_logits.py'))
    snapshot = next(i for i, a in enumerate(commands) if a[:2] == ['python3', '-c'])
    long = next(i for i, a in enumerate(commands) if a[0] == 'python3' and len(a) > 1 and a[1].endswith('collect_phase_logits.py'))
    assert short < snapshot < long
    assert len(list(state['capture'].glob('logits-*.json'))) == 32
    assert len([a for a in commands if a[:2] == ['docker', 'create']]) == 2
    assert len([a for a in commands if a[:2] == ['docker', 'stop']]) == 2
    run_neutrality.verify_proof(proof, plan, path)
    changed = state['capture'] / 'logits-000031.f32'
    corrupted = bytearray(changed.read_bytes())
    corrupted[-1] ^= 1
    changed.write_bytes(corrupted)
    with pytest.raises(ValueError, match='raw capture differs'):
        run_neutrality.verify_proof(proof, plan, path)


def test_long_collection_failure_stops_both_ranks_and_retains_short_results(prepared):
    path, plan, state = prepared
    transport = CPUTransport(plan, long_failure=True)
    with pytest.raises(RuntimeError, match='command failed'):
        run_neutrality.run(path, transport)
    attempt = Path(plan['preparation_root']) / plan['attempt']
    assert (attempt / 'failure.json').is_file()
    assert (attempt / 'runs/arm-a-repeat-00/short16/manifest.json').is_file()
    assert not (attempt / 'neutrality-proof.json').exists()
    assert [host for host, a in transport.commands if a[:2] == ['docker', 'stop']] == ['spark0', 'spark1']
    assert [host for host, a in transport.commands if a[:2] == ['docker', 'wait']] == ['spark0', 'spark1']
    assert prepare.read(attempt / 'runs/arm-a-repeat-00/lifecycle.json')['teardown_errors'] == []


def test_completed_proof_rejects_forged_lifecycle_provider_binary_and_runner(prepared):
    import copy
    path, plan, state = prepared
    proof = run_neutrality.run(path, CPUTransport(plan))
    root = state['capture'].parent
    lifecycle_path = root / 'lifecycle.json'
    original = lifecycle_path.read_bytes()
    lifecycle = prepare.read(lifecycle_path)
    del lifecycle['started']['1']
    lifecycle_path.write_text(json.dumps(lifecycle))
    forged = copy.deepcopy(proof)
    forged['lifecycle_sha256'] = prepare.sha(lifecycle_path)
    with pytest.raises(ValueError, match='both-rank lifecycle ownership'):
        run_neutrality.verify_proof(forged, plan, path)
    lifecycle_path.write_bytes(original)
    for kind, field, value, expected in [('provider', 'target', 'foreign-container', 'provider target'),
                                          ('binary', None, 'f' * 64 + '  /usr/local/bin/spark', 'binary receipt')]:
        receipt = next(item for item in proof['execution_receipts'] if item['kind'] == kind)
        stdout = Path(receipt['path']).with_suffix('.stdout')
        original = stdout.read_bytes()
        if kind == 'provider':
            report = json.loads(original)
            report[field] = value
            stdout.write_text(json.dumps(report))
        else:
            stdout.write_text(value)
        forged = copy.deepcopy(proof)
        next(item for item in forged['execution_receipts'] if item['path'] == receipt['path'])['stdout_sha256'] = prepare.sha(stdout)
        with pytest.raises(ValueError, match=expected):
            run_neutrality.verify_proof(forged, plan, path)
        stdout.write_bytes(original)
    changed_plan = copy.deepcopy(plan)
    changed_plan['runner_bindings']['prepare.py'] = '0' * 64
    with pytest.raises(ValueError, match='runner source changed'):
        run_neutrality.verify_proof(proof, changed_plan, path)


def test_changed_frozen_phase_source_is_refused_before_new_prep_creation(prepared, tmp_path):
    path, plan, state = prepared
    cfg = dict(state['config'])
    cfg['prep_root'] = str(tmp_path / 'second-prep')
    changed = tmp_path / 'changed-phase.py'
    changed.write_bytes(PHASE.read_bytes() + b'\n# altered source\n')
    cfg['phase_collector'] = str(changed)
    with pytest.raises(ValueError, match='frozen long artifact source'):
        prepare.prepare_neutrality(cfg)
    assert not Path(cfg['prep_root']).exists()
