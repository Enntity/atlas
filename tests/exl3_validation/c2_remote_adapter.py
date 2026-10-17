# SPDX-License-Identifier: AGPL-3.0-only
"""Explicit remote lifecycle for C2; never removes containers or evidence.

Each host command prefix receives one shell-quoted remote command (SSH or a
spark wrapper). Inputs and run paths are mirrored at their absolute paths.
Harness files must already be staged and bound by the operator. Provider
checks must use the registered checker, with separate cuBLAS/cuBLASLt legs.
"""
from contextlib import contextmanager
import io
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time


def require(ok, message):
    if not ok:
        raise ValueError(message)


class RemoteAdapter:
    def __init__(self, config, runner=subprocess.run):
        self.config, self.runner = config, runner
        keys = {'hosts', 'image_ids', 'remote_python', 'remote_harness_dir',
                'request_ids', 'readiness_command', 'readiness_timeout_seconds',
                'command_timeout_seconds', 'provider_checks', 'remote_harness_sha256'}
        require(set(config) == keys, 'remote config requires exactly ' + str(sorted(keys)))
        for field in ('hosts', 'image_ids', 'provider_checks'):
            require(set(config[field]) == {'0', '1'}, field + ' must cover both ranks')
        for rank in ('0', '1'):
            self._argv(config['hosts'][rank])
            require(config['image_ids'][rank].startswith('sha256:')
                    and len(config['image_ids'][rank]) == 71, 'full image ID required')
            checks = config['provider_checks'][rank]
            require(set(checks) == {'cublas', 'cublaslt'}, 'both provider checks required')
            for command in checks.values():
                self._argv(command)
                require(str(Path(config['remote_harness_dir']) / 'attest_runtime_maps.py') in command,
                        'provider command must invoke attested runtime maps checker')
        for field in ('remote_python', 'readiness_command'):
            self._argv(config[field])
        require(Path(config['remote_harness_dir']).is_absolute(), 'absolute harness directory required')
        require(isinstance(config['request_ids'], list) and config['request_ids']
                and all(isinstance(x, str) and x and all(33 <= ord(c) <= 126 for c in x)
                        for x in config['request_ids'])
                and len(set(config['request_ids'])) == len(config['request_ids']),
                'explicit unique scheduled request IDs required')
        for field in ('readiness_timeout_seconds', 'command_timeout_seconds'):
            require(type(config[field]) in (float, int) and config[field] > 0,
                    field + ' must be positive')
        hashes = config['remote_harness_sha256']
        required = {'collect_logits.py', 'parity_protocol.py', 'attest_runtime_maps.py'}
        require(isinstance(hashes, dict) and set(hashes) == required,
                'all collector/protocol/provider harness hashes required')
        for name, digest in hashes.items():
            local = Path(__file__).parent / name
            require(isinstance(digest, str) and digest == hashlib.sha256(local.read_bytes()).hexdigest(),
                    'harness hash must match local source: ' + name)
        self.active = None
        self.sequence = 0
        self.staged = set()

    @staticmethod
    def _argv(args):
        require(isinstance(args, list) and args and all(isinstance(x, str) and x for x in args),
                'commands must be explicit nonempty argv lists')

    def _run(self, rank, args, *, data=None, check=True):
        self._argv(args)
        prefix = None
        if self.active:
            prefix = self.active['run_root'] / f'remote-{self.sequence:04d}-rank{rank}'
            self.sequence += 1
            prefix.with_suffix('.json').write_text(json.dumps({'argv': args, 'status': 'issued'}) + '\n')
        try:
            result = self.runner(self.config['hosts'][rank] + [shlex.join(args)], input=data,
                                 capture_output=True, timeout=self.config['command_timeout_seconds'])
        except Exception as exc:
            if prefix:
                prefix.with_suffix('.json').write_text(json.dumps(
                    {'argv': args, 'status': 'transport-failed', 'exception': str(exc)}) + '\n')
                for suffix, attr in (('.stdout', 'stdout'), ('.stderr', 'stderr')):
                    payload = getattr(exc, attr, None)
                    if isinstance(payload, bytes):
                        prefix.with_suffix(suffix).write_bytes(payload)
            raise
        if prefix:
            prefix.with_suffix('.json').write_text(json.dumps(
                {'argv': args, 'returncode': result.returncode}, indent=2) + '\n')
            prefix.with_suffix('.stdout').write_bytes(result.stdout)
            prefix.with_suffix('.stderr').write_bytes(result.stderr)
        if check and result.returncode:
            raise RuntimeError(f'rank {rank} command failed ({result.returncode}): {args[0]}')
        return result

    def _stage(self, rank, path):
        path = Path(path).resolve()
        require(path.is_dir(), 'staging requires existing local directory')
        # mkdir without -p for the leaf refuses existing remote evidence.
        self._run(rank, ['mkdir', '-p', str(path.parent)])
        self._run(rank, ['mkdir', str(path)])
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode='w') as tar:
            for child in sorted(path.iterdir()):
                tar.add(child, arcname=child.name)
        self._run(rank, ['tar', '-xf', '-', '-C', str(path)], data=archive.getvalue())

    def _retrieve(self, rank, remote, destination):
        result = self._run(rank, ['tar', '-cf', '-', '-C', str(remote), '.'])
        with tarfile.open(fileobj=io.BytesIO(result.stdout)) as tar:
            members = tar.getmembers()
            for member in members:
                require(member.isdir() or member.isfile(), 'remote evidence must be regular files')
                target = (Path(destination) / member.name).resolve()
                require(target.is_relative_to(Path(destination).resolve()), 'unsafe remote archive path')
                if member.isfile() and target.exists():
                    payload = tar.extractfile(member).read()
                    require(target.read_bytes() == payload, 'rank evidence collision: ' + member.name)
            tar.extractall(destination, members=members, filter='data')

    def _create_args(self, rank, context):
        args = json.loads((context['run_root'] / context['create_commands'][rank]['file']).read_text())
        require('--name' in args, 'descriptor must explicitly name container')
        name_index = args.index('--name') + 1
        args[name_index] = 'atlas-c2-' + context['run_root'].parent.parent.name + '-' + context['run_root'].name + '-r' + rank
        require('--rm' not in args, 'automatic container removal forbidden')
        index = args.index('serve') - 1
        require(args[index] == context['create_commands'][rank]['image'], 'descriptor image mismatch')
        # Pin content ID after validating the registered tag on the host.
        args[index] = self.config['image_ids'][rank]
        mounts = []
        for path in (context['run_root'], context['records_dir'], context['gated_norm_dir'],
                     Path(context['arm']['capture_dir'])):
            mounts += ['--volume', f'{path}:{path}']
        if context['arm']['knob']['engaged']:
            path = context['arm']['knob']['substitute_dir']
            mounts += ['--volume', f'{path}:{path}:ro']
        return args[:index] + mounts + args[index:]

    @contextmanager
    def lifecycle(self, **context):
        self.active = context
        created, owned_names, started, errors = {}, {}, {}, []
        try:
            commands = {rank: self._create_args(rank, context) for rank in ('0', '1')}
            # Check both hosts before any mutation; never stop another owner's work.
            for rank in ('0', '1'):
                for name, digest in self.config['remote_harness_sha256'].items():
                    path = str(Path(self.config['remote_harness_dir']) / name)
                    actual_hash = self._run(rank, ['sha256sum', path]).stdout.decode().split()
                    require(actual_hash and actual_hash[0] == digest, 'remote harness hash mismatch: ' + path)
                image = context['create_commands'][rank]['image']
                actual = self._run(rank, ['docker', 'image', 'inspect', '--format', '{{.Id}}', image]).stdout.decode().strip()
                require(actual == self.config['image_ids'][rank], 'image identity mismatch on rank ' + rank)
                require(not self._run(rank, ['docker', 'ps', '-q']).stdout.strip(), 'host has running containers')
                require(not self._run(rank, ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']).stdout.strip(),
                        'host has GPU process owner')
            for rank in ('0', '1'):
                for path in (context['run_root'], context['records_dir'], context['gated_norm_dir']):
                    self._stage(rank, path)
                if context['arm']['knob']['engaged']:
                    fixture = Path(context['arm']['knob']['substitute_dir'])
                    if (rank, fixture) not in self.staged:
                        self._stage(rank, fixture)
                        self.staged.add((rank, fixture))
                name = commands[rank][commands[rank].index('--name') + 1]
                existing = self._run(rank, ['docker', 'inspect', '--format', '{{.Id}}', name], check=False)
                require(existing.returncode == 1 and (b'no such object' in existing.stderr.lower()
                        or b'no such container' in existing.stderr.lower()),
                        'generated container name absence could not be proven')
                owned_names[rank] = name
                result = self._run(rank, commands[rank])
                container = result.stdout.decode().strip()
                require(len(container) == 64 and all(x in '0123456789abcdef' for x in container),
                        'docker create must return full container ID')
                created[rank] = container
                (context['run_root'] / f'live-create-rank{rank}.json').write_text(json.dumps(commands[rank]) + '\n')
            for rank in ('1', '0'):
                self._run(rank, ['docker', 'start', created[rank]])
                started[rank] = created[rank]
            deadline = time.monotonic() + self.config['readiness_timeout_seconds']
            while True:
                result = self._run('0', self.config['readiness_command'], check=False)
                if result.returncode == 0:
                    break
                require(time.monotonic() < deadline, 'server health readiness timed out')
                time.sleep(min(1, max(0, deadline - time.monotonic())))
            for rank in ('0', '1'):
                for provider in ('cublas', 'cublaslt'):
                    command = [arg.replace('{container}', created[rank])
                               for arg in self.config['provider_checks'][rank][provider]]
                    self._run(rank, command)
            yield
        finally:
            # Try every teardown/collection even when one rank fails.
            for rank, name in owned_names.items():
                container = created.get(rank)
                if container is None:
                    try:
                        inspected = self._run(rank, ['docker', 'inspect', '--format', '{{.Id}}', name], check=False)
                        if inspected.returncode != 0:
                            continue
                        container = inspected.stdout.decode().strip()
                        require(len(container) == 64 and all(c in '0123456789abcdef' for c in container),
                                'ambiguous owned container inspection')
                    except Exception as exc:
                        errors.append(str(exc))
                        continue
                operations = [['docker', 'logs', container], ['docker', 'stop', container],
                              ['docker', 'wait', container]]
                for args in operations:
                    try:
                        self._run(rank, args)
                    except Exception as exc:
                        errors.append(str(exc))
                for path in (context['records_dir'], context['gated_norm_dir'], context['run_root'] / 'ffn',
                             context['run_root'] / 'logits'):
                    try:
                        self._retrieve(rank, path, path)
                    except Exception as exc:
                        errors.append(str(exc))
            (context['run_root'] / 'lifecycle.json').write_text(json.dumps(
                {'containers': created, 'owned_names': owned_names, 'started': started, 'teardown_errors': errors}, indent=2) + '\n')
            self.active = None
            if errors:
                raise RuntimeError('remote teardown/evidence retrieval failed: ' + '; '.join(errors))

    def collect(self, **kwargs):
        require(self.active is not None, 'remote collector requires active lifecycle')
        root = self.active['run_root']
        inputs = root / 'collector-inputs'
        inputs.mkdir()
        for name, value in (('cases', kwargs['cases']), ('thresholds', kwargs['thresholds']),
                            ('provenance', kwargs['provenance']), ('request_ids', self.config['request_ids'])):
            (inputs / (name + '.json')).write_text(json.dumps(value, indent=2) + '\n')
        self._stage('0', inputs)
        capture = Path(kwargs['capture_dir'])
        args = self.config['remote_python'] + [str(Path(self.config['remote_harness_dir']) / 'collect_logits.py')]
        for key, value in (('endpoint', kwargs['endpoint']), ('model', kwargs['model']),
                           ('capture-dir', capture), ('out', kwargs['destination']), ('timeout', kwargs['timeout'])):
            args += ['--' + key, str(value)]
        for name in ('cases', 'thresholds', 'provenance', 'request_ids'):
            args += ['--' + name.replace('_', '-'), str(inputs / (name + '.json'))]
        try:
            self._run('0', args)
        finally:
            Path(kwargs['destination']).mkdir(parents=True, exist_ok=True)
            self._retrieve('0', kwargs['destination'], kwargs['destination'])
        return json.loads((kwargs['destination'] / 'manifest.json').read_text())


def validate_schedule(plan, fixture, scheduled):
    """Bind teacher-forced request order to both rank fixture invocations."""
    require(len(scheduled) == sum(len(case['continuation_ids']) + 1 for case in plan['cases']),
            'scheduled request IDs must cover all collector prefixes')
    for rank in (0, 1):
        ordered = [entry['request_id'] for entry in sorted(
            (entry for entry in fixture['entries'] if entry['tp_rank'] == rank),
            key=lambda entry: entry['invocation'])]
        collapsed = [value for index, value in enumerate(ordered)
                     if index == 0 or value != ordered[index - 1]]
        require(scheduled == collapsed, 'request schedule must bind ordered fixture invocations')

def main():
    """Prepare/review a complete explicit plan, or execute its remote lifecycle."""
    import argparse
    import c2_ab_driver as driver
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('evidence-root', 'attempt', 'variant', 'fixture-dir', 'run-config', 'remote-config'):
        parser.add_argument('--' + flag, required=True)
    parser.add_argument('--repeats', required=True, type=int)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    run_config = driver.load_run_config(args.run_config)
    adapter = RemoteAdapter(json.loads(Path(args.remote_config).read_text()))
    plan = {'evidence_root': Path(args.evidence_root), 'attempt': args.attempt,
            'variant': args.variant, 'fixture_dir': Path(args.fixture_dir),
            'repeats': args.repeats,
            'arm_configs': driver.arm_configurations(run_config, args.fixture_dir, args.variant),
            'provenance': json.loads(Path(run_config['provenance_file']).read_text()),
            'cases': json.loads(Path(run_config['cases_file']).read_text()),
            'thresholds': json.loads(Path(run_config['thresholds_file']).read_text())}
    fixture = driver.validate_plan(plan)
    scheduled = adapter.config['request_ids']
    validate_schedule(plan, fixture, scheduled)
    if args.dry_run:
        print(json.dumps({'mode': 'dry-run', 'attempt': args.attempt,
                          'runs': args.repeats * 2, 'request_ids': scheduled,
                          'image_ids': adapter.config['image_ids'],
                          'remote_harness_dir': adapter.config['remote_harness_dir']}, indent=2))
    else:
        receipt = driver.run_ab(plan, collector=adapter.collect, lifecycle=adapter.lifecycle)
        print(json.dumps({'attempt': args.attempt, 'causality': receipt['causality']['status']}))


if __name__ == '__main__':
    main()
