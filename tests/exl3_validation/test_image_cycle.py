# SPDX-License-Identifier: AGPL-3.0-only
"""CPU tests for the one-command image lifecycle driver (no ssh, no GPU)."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import image_cycle


IMAGE_ID = 'sha256:' + 'ab' * 32
MANIFEST = {
    'schema': 'atlas-exl3-build-manifest-v1',
    'git_commit': 'f' * 40,
    'source_snapshot_sha256': 'cd' * 32,
    'binaries': {'spark': {'sha256': '11' * 32}},
    'ptx_modules': ['a.ptx', 'b.ptx'],
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FakeRunner:
    """Canned stage outputs; records every argv it was asked to run."""

    def __init__(self, plan, *, busy_rank1=False, build_rc=0, wrong_manifest=False,
                 transfer_mismatch=False):
        self.plan = plan
        self.busy_rank1 = busy_rank1
        self.build_rc = build_rc
        self.wrong_manifest = wrong_manifest
        self.transfer_mismatch = transfer_mismatch
        self.calls = []

    def run(self, argv, *, env=None, timeout):
        self.calls.append(list(argv))
        joined = ' '.join(argv)
        if 'inspect.sh' in joined:
            out = Path(argv[argv.index('--out') + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / 'spark0.txt').write_text(
                '-- gpu\n580.159.03, 0 %\n-- gpu processes '
                '(ownership: anything running here owns the GPU)\n\n'
                '-- docker ps -a\n')
            busy = '\npid, name\n' if self.busy_rank1 else ''
            (out / 'spark1.txt').write_text(
                '-- gpu\n580.159.03, 0 %\n-- gpu processes '
                '(ownership: anything running here owns the GPU)\n' + busy +
                '-- docker ps -a\n')
            return {'rc': 0, 'stdout': b'', 'stderr': b''}
        if 'build_image.sh' in joined:
            line = (f'{IMAGE_ID} {json.dumps({"ATLAS_GIT_SHA": "x"})}\n'
                    if self.build_rc == 0 else '')
            return {'rc': self.build_rc,
                    'stdout': ('log line\n' + line).encode(),
                    'stderr': b''}
        if 'docker image inspect' in joined:
            host = argv[-2]
            image_id = IMAGE_ID
            if self.transfer_mismatch and host.endswith('.11'):
                image_id = 'sha256:' + 'cd' * 32
            return {'rc': 0, 'stdout': (image_id + '\n').encode(), 'stderr': b''}
        if '--entrypoint /bin/cat' in joined:
            value = dict(MANIFEST)
            if self.wrong_manifest:
                value = dict(value, source_snapshot_sha256='00' * 32)
            return {'rc': 0, 'stdout': json.dumps(value).encode(), 'stderr': b''}
        if '--mode' in argv:
            prep_root = Path(self.plan['prep_root'])
            prep_root.mkdir(parents=True, exist_ok=True)
            (prep_root / 'neutrality-plan.json').write_text('{}')
            return {'rc': 0, 'stdout': b'', 'stderr': b''}
        if '--execute' in argv:
            attempt = self.plan['config']['neutrality']['preparation_inputs']['attempt']
            root = Path(self.plan['prep_root']) / attempt
            root.mkdir(parents=True, exist_ok=True)
            (root / 'neutrality-proof.json').write_text(
                json.dumps({'passed': True, 'rows_total': 32}))
            return {'rc': 0, 'stdout': b'{"passed": true}', 'stderr': b''}
        raise AssertionError('unexpected argv: ' + joined)

    def stream(self, producer_argv, consumer_argv, *, timeout):
        self.calls.append(list(producer_argv))
        self.calls.append(list(consumer_argv))
        return {'stream_bytes': 4096, 'stream_sha256': 'be' * 32,
                'producer_rc': 0, 'consumer_rc': 0}


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='atlas-image-cycle-')
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.evidence = root / 'evidence'
        self.evidence.mkdir()
        files = {}
        for name in ('inspect.sh', 'prepare.py', 'runner.py',
                     'baseline-provenance.json', 'remote-template.json',
                     'phase-collector.py', 'pins.sha256'):
            path = root / name
            path.write_text('# fixture ' + name + '\n')
            files[name] = path
        profile = root / 'profile'
        profile.mkdir()
        for rank in ('0', '1'):
            (profile / f'cand-rank{rank}-create-command.json').write_text('[]')
        self.frozen = {}
        for suite in ('short', 'long'):
            directory = root / f'frozen-{suite}'
            directory.mkdir()
            (directory / 'manifest.json').write_text('{"cases": []}')
            self.frozen[suite] = directory
        self.config = {
            'schema': 'atlas-exl3-image-cycle-config-v1',
            'cycle_id': 'imgcycle-fixture01',
            'evidence_root': str(self.evidence),
            'jump': 'colum@192.168.1.50',
            'build_host': 'colum@10.50.0.10',
            'ranks': {'0': 'colum@10.50.0.10', '1': 'colum@10.50.0.11'},
            'build': {'checkout_dir': '/srv/checkout',
                      'blas_pair_dir': '/srv/blas-pair',
                      'source_snapshot_sha256': 'cd' * 32,
                      'rust_version': '1.97.1', 'cpus': '0-9'},
            'image_tag': 'atlas-exl3-validation:src-cdcdcdcdcdcd-rust1.97.1',
            'neutrality': {
                'prepare_script': str(files['prepare.py']),
                'prepare_script_sha256': sha(files['prepare.py']),
                'runner_script': str(files['runner.py']),
                'runner_script_sha256': sha(files['runner.py']),
                'preparation_inputs': {
                    'profile_dir': str(profile),
                    'frozen_short': str(self.frozen['short']),
                    'frozen_long': str(self.frozen['long']),
                    'baseline_provenance': str(files['baseline-provenance.json']),
                    'remote_template': str(files['remote-template.json']),
                    'phase_collector': str(files['phase-collector.py']),
                    'pins_file': str(files['pins.sha256']),
                    'attempt': 'cycle-neutral01',
                    'preserved_failures': [],
                    'runtime_binary_path': '/usr/local/bin/spark'}},
            'inspect_script': str(files['inspect.sh']),
            'inspect_script_sha256': sha(files['inspect.sh']),
        }
        self.config_path = root / 'cycle.json'
        self.write_config()

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config))

    def plan(self):
        return image_cycle.validate(self.config_path)


class ValidateTests(Fixture):
    def test_valid_config_produces_plan_with_bindings(self):
        plan = self.plan()
        self.assertEqual(plan['config']['cycle_id'], 'imgcycle-fixture01')
        self.assertTrue(plan['prep_root'].endswith('04-neutrality/preparation'))
        self.assertIn('inspect_script', plan['bindings'])

    def test_image_tag_mismatch_rejected(self):
        self.config['image_tag'] = 'atlas-exl3-validation:src-other-rust1.97.1'
        self.write_config()
        with self.assertRaisesRegex(ValueError, 'derivation'):
            self.plan()

    def test_bound_hash_mismatch_rejected(self):
        Path(self.config['inspect_script']).write_text('tampered\n')
        with self.assertRaisesRegex(ValueError, 'hash differs: inspect_script'):
            self.plan()

    def test_unsafe_cycle_id_rejected(self):
        self.config['cycle_id'] = 'Cycle_1'
        self.write_config()
        with self.assertRaisesRegex(ValueError, 'cycle id'):
            self.plan()

    def test_missing_profile_descriptor_rejected(self):
        Path(self.config['neutrality']['preparation_inputs']['profile_dir'],
             'cand-rank1-create-command.json').unlink()
        with self.assertRaisesRegex(ValueError, 'profile descriptor'):
            self.plan()

    def test_dry_run_reports_plan_without_creating_cycle_root(self):
        plan = self.plan()
        result = image_cycle.dry_run(plan)
        self.assertEqual(result['mode'], 'dry-run')
        self.assertFalse(Path(plan['cycle_root']).exists())
        names = [stage['name'] for stage in result['stages']]
        self.assertEqual(names, ['preflight', 'build', 'transfer',
                                 'fingerprint', 'neutrality'])


class ParsingTests(unittest.TestCase):
    def test_idle_and_busy_gpu_sections(self):
        idle = ('-- gpu processes (ownership)\n\n-- docker ps -a\n')
        busy = ('-- gpu processes (ownership)\n1234 python 100MiB\n'
                '-- docker ps -a\n')
        self.assertTrue(image_cycle.gpu_processes_idle(idle))
        self.assertFalse(image_cycle.gpu_processes_idle(busy))
        with self.assertRaisesRegex(ValueError, 'section'):
            image_cycle.gpu_processes_idle('no section here')

    def test_build_inspect_line_parsed_exactly_once(self):
        stdout = ('build output\n'
                  f'{IMAGE_ID} {{"ATLAS_GIT_SHA": "abc"}}\n'
                  'build manifest: /usr/local/share/...\n')
        image_id, labels = image_cycle.parse_build_image_id(stdout)
        self.assertEqual(image_id, IMAGE_ID)
        self.assertEqual(labels, {'ATLAS_GIT_SHA': 'abc'})
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            image_cycle.parse_build_image_id(stdout + IMAGE_ID + ' {}\n')
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            image_cycle.parse_build_image_id('no image line\n')


class ExecuteTests(Fixture):
    def execute(self, runner_kwargs=None):
        plan = self.plan()
        runner = FakeRunner(plan, **(runner_kwargs or {}))
        original = image_cycle.SubprocessRunner
        image_cycle.SubprocessRunner = lambda: runner
        try:
            return plan, runner, image_cycle.execute(plan)
        finally:
            image_cycle.SubprocessRunner = original

    def test_full_chain_binds_every_stage_receipt(self):
        plan, runner, receipt = self.execute()
        self.assertTrue(receipt['completed'])
        self.assertEqual(receipt['image_id'], IMAGE_ID)
        self.assertEqual(receipt['git_commit'], 'f' * 40)
        self.assertIn('01-build/build-stdout.log', receipt['stage_files'])
        self.assertIn('02-transfer/transfer.json', receipt['stage_files'])
        for relative, digest in receipt['stage_files'].items():
            self.assertEqual(sha(Path(plan['cycle_root']) / relative), digest)
        proof = Path(receipt['neutrality']['proof_path'])
        self.assertEqual(sha(proof), receipt['neutrality']['proof_sha256'])
        self.assertEqual(len(runner.calls), 10)

    def test_neutrality_config_written_from_verified_fingerprint(self):
        plan, _, _ = self.execute()
        path = Path(plan['cycle_root']) / '04-neutrality' \
            / 'neutrality-preparation-config.json'
        written = json.loads(path.read_text())
        self.assertEqual(written['image_id'], IMAGE_ID)
        self.assertEqual(written['runtime_source'], 'f' * 40)
        self.assertEqual(written['source_snapshot_sha256'], 'cd' * 32)
        self.assertEqual(written['build_manifest'],
                         str((Path(plan['cycle_root']) / '03-fingerprint'
                              / 'build-manifest.json').resolve()))
        self.assertEqual(written['attempt'], 'cycle-neutral01')
        self.assertEqual(written['schema'],
                         'atlas-c2-neutrality-preparation-config-v1')

    def patch(self, runner):
        original = image_cycle.SubprocessRunner
        image_cycle.SubprocessRunner = lambda: runner
        self.addCleanup(setattr, image_cycle, 'SubprocessRunner', original)

    def test_busy_rank_aborts_at_preflight(self):
        plan = self.plan()
        runner = FakeRunner(plan, busy_rank1=True)
        self.patch(runner)
        with self.assertRaisesRegex(ValueError, 'spark1 still owns'):
            image_cycle.execute(plan)
        failure = json.loads(
            (Path(plan['cycle_root']) / 'failure.json').read_text())
        self.assertEqual(failure['stage'], 'preflight')

    def test_build_failure_records_stage_and_logs(self):
        plan = self.plan()
        runner = FakeRunner(plan, build_rc=1)
        self.patch(runner)
        with self.assertRaisesRegex(ValueError, 'no-cache build failed'):
            image_cycle.execute(plan)
        failure = json.loads(
            (Path(plan['cycle_root']) / 'failure.json').read_text())
        self.assertEqual(failure['stage'], 'build')
        self.assertTrue((Path(plan['cycle_root']) / '01-build'
                         / 'build-stdout.log').is_file())

    def test_manifest_rank_divergence_rejected(self):
        plan = self.plan()
        runner = FakeRunner(plan, wrong_manifest=True)
        self.patch(runner)
        with self.assertRaisesRegex(ValueError, 'source snapshot differs'):
            image_cycle.execute(plan)

    def test_transfer_rank_mismatch_rejected(self):
        plan = self.plan()
        runner = FakeRunner(plan, transfer_mismatch=True)
        self.patch(runner)
        with self.assertRaisesRegex(ValueError, 'spark1 image id differs'):
            image_cycle.execute(plan)
        failure = json.loads(
            (Path(plan['cycle_root']) / 'failure.json').read_text())
        self.assertEqual(failure['stage'], 'transfer')


class StreamTests(unittest.TestCase):
    """The real streamer must enforce both transport exit codes."""

    def stream(self, producer, consumer):
        return image_cycle.SubprocessRunner().stream(
            ['bash', '-c', producer], ['bash', '-c', consumer], timeout=60)

    def test_happy_stream_hashes_and_passes(self):
        result = self.stream('printf 12345678', 'cat > /dev/null')
        self.assertEqual(result['stream_bytes'], 8)
        self.assertEqual(result['stream_sha256'],
                         hashlib.sha256(b'12345678').hexdigest())
        self.assertEqual((result['producer_rc'], result['consumer_rc']), (0, 0))

    def test_producer_failure_rejected(self):
        with self.assertRaisesRegex(ValueError, r'save rc=3'):
            self.stream('printf data; exit 3', 'cat > /dev/null')

    def test_consumer_failure_rejected(self):
        with self.assertRaisesRegex(ValueError, r'load rc=7'):
            self.stream('printf data', 'cat > /dev/null; exit 7')


if __name__ == '__main__':
    unittest.main()
