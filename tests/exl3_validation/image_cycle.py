# SPDX-License-Identifier: AGPL-3.0-only
"""One-command validation-image lifecycle between GPU experiments.

Drives preflight -> build -> transfer -> fingerprint -> neutrality -> receipt
for the GLM-5.3 EXL3 validation image, replacing the per-campaign re-derivation
of that chain (each stage was previously assembled by hand; see the receipt
layouts this mirrors in the evidence root: preflight-inspection-driver.log,
canonical-build-command.json, image-transfer.json, image-fingerprint/,
neutrality-*).

Every stage appends receipts under ``<evidence_root>/<cycle_id>/``; nothing is
deleted or retried in place. The neutrality preparation config is *written by
this driver* from the verified fingerprint stage (its image identity cannot be
known before the build) and the operator's bound campaign inputs. The final
receipt binds every stage receipt by SHA256. Configuration is explicit: no
defaults, fail fast on any mismatch (PCND).

Usage:
    image_cycle.py --config cycle.json --dry-run   # validate + print plan
    image_cycle.py --config cycle.json --execute   # run the chain
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import select
import subprocess
import sys
import time

CONFIG_SCHEMA = 'atlas-exl3-image-cycle-config-v1'
RECEIPT_SCHEMA = 'atlas-exl3-image-cycle-receipt-v1'
MANIFEST_SCHEMA = 'atlas-exl3-build-manifest-v1'
NEUTRALITY_SCHEMA = 'atlas-c2-neutrality-preparation-config-v1'
CONFIG_KEYS = {'schema', 'cycle_id', 'evidence_root', 'jump', 'build_host',
               'ranks', 'build', 'image_tag', 'neutrality', 'inspect_script',
               'inspect_script_sha256'}
BUILD_KEYS = {'checkout_dir', 'blas_pair_dir', 'source_snapshot_sha256',
              'rust_version', 'cpus'}
NEUTRALITY_KEYS = {'prepare_script', 'prepare_script_sha256',
                   'runner_script', 'runner_script_sha256', 'preparation_inputs'}
# The exact key set prepare.py --mode neutrality accepts (its NEUTRAL_KEYS);
# the driver derives the five image-identity fields from the fingerprint stage.
PREPARATION_INPUT_KEYS = {'profile_dir', 'frozen_short', 'frozen_long',
                          'baseline_provenance', 'remote_template',
                          'phase_collector', 'pins_file', 'attempt',
                          'preserved_failures', 'runtime_binary_path'}
SSH_OPTIONS = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15']
MANIFEST_PATH = '/usr/local/share/atlas/fingerprint/build-manifest.json'
TIMEOUTS = {'preflight': 600, 'build': 7200, 'transfer': 3600,
            'inspect': 300, 'prepare': 1800, 'neutrality': 14400}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def text(value):
    return json.dumps(value, indent=2, allow_nan=False) + '\n'


def write_json(path, value):
    with Path(path).open('x') as target:
        target.write(text(value))


class SubprocessRunner:
    """Local subprocess boundary with an explicit timeout on every command."""

    def run(self, argv, *, env=None, timeout):
        try:
            done = subprocess.run(argv, capture_output=True, env=env,
                                  timeout=timeout)
        except subprocess.TimeoutExpired as error:
            raise ValueError(f'command timed out after {timeout}s: {argv[:3]}') from error
        return {'rc': done.returncode, 'stdout': done.stdout, 'stderr': done.stderr}

    def stream(self, producer_argv, consumer_argv, *, timeout):
        """Pipe producer stdout to consumer stdin while hashing the stream."""
        deadline = time.monotonic() + timeout
        digest = hashlib.sha256()
        size = 0
        producer = subprocess.Popen(producer_argv, stdout=subprocess.PIPE)
        consumer = subprocess.Popen(consumer_argv, stdin=subprocess.PIPE)
        broken = None
        timed_out = False
        try:
            assert producer.stdout is not None and consumer.stdin is not None
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([producer.stdout], [], [], max(0, remaining))[0]:
                    timed_out = remaining <= 0
                    break
                chunk = producer.stdout.read(65536)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
                try:
                    consumer.stdin.write(chunk)
                except BrokenPipeError as error:
                    broken = error
            if timed_out:
                producer.kill()
                consumer.kill()
            try:
                consumer.stdin.close()
            except (BrokenPipeError, ValueError):
                pass
        finally:
            producer.wait(timeout=60)
            consumer.wait(timeout=60)
        require(not timed_out, f'image transfer exceeded {timeout}s')
        require(broken is None, 'docker load closed the transfer stream early')
        require(producer.returncode == 0 and consumer.returncode == 0,
                'transfer command failed (save rc=%s, load rc=%s)'
                % (producer.returncode, consumer.returncode))
        return {'stream_bytes': size, 'stream_sha256': digest.hexdigest(),
                'producer_rc': producer.returncode,
                'consumer_rc': consumer.returncode}


def ssh_argv(jump, host, command):
    return ['ssh', *SSH_OPTIONS, '-J', jump, host, command]


def derived_tag(config):
    snapshot = config['build']['source_snapshot_sha256']
    rust = config['build']['rust_version']
    return f'atlas-exl3-validation:src-{snapshot[:12]}-rust{rust}'


def gpu_processes_idle(text_body):
    """True iff the inspect receipt's GPU-process section lists no processes."""
    start = text_body.find('-- gpu processes')
    require(start >= 0, 'inspect receipt lacks the GPU-process section')
    end = text_body.find('\n-- ', start + 1)
    section = text_body[start:end if end >= 0 else None]
    lines = [line for line in section.splitlines()[1:] if line.strip()]
    return not lines


def parse_build_image_id(stdout_body):
    """Extract (image_id, labels) from build_image.sh's inspect line."""
    matches = re.findall(r'^(sha256:[0-9a-f]{64}) (\{.*\})$', stdout_body, re.M)
    require(len(matches) == 1,
            'expected exactly one image-inspect line in build output')
    image_id, labels = matches[0]
    return image_id, json.loads(labels)


def _bound(path_value, expected, label, bindings):
    path = Path(path_value)
    require(path.is_absolute() and path.is_file(), 'missing bound input: ' + label)
    actual = sha(path)
    require(actual == expected, 'bound input hash differs: ' + label)
    bindings[label] = {'path': str(path), 'sha256': actual}


def validate(config_path):
    config_path = Path(config_path).resolve()
    config = read(config_path)
    require(set(config) == CONFIG_KEYS and config['schema'] == CONFIG_SCHEMA,
            'explicit image-cycle configuration keys/schema required')
    require(re.fullmatch(r'[a-z0-9][a-z0-9-]{0,55}', config['cycle_id']) is not None,
            'filename-safe cycle id required')
    require(Path(config['evidence_root']).is_absolute(), 'evidence root must be absolute')
    for key in ('jump', 'build_host', 'image_tag'):
        require(isinstance(config[key], str) and config[key], 'explicit ' + key + ' required')
    require(set(config['ranks']) == {'0', '1'}
            and all(isinstance(v, str) and v for v in config['ranks'].values()),
            'both explicit rank hosts required')
    build = config['build']
    require(set(build) == BUILD_KEYS, 'explicit build keys required')
    require(re.fullmatch(r'[0-9a-f]{64}', build['source_snapshot_sha256']) is not None,
            'full source snapshot required')
    require(re.fullmatch(r'[0-9]+\.[0-9]+\.[0-9]+', build['rust_version']) is not None,
            'exact Rust release required')
    for key in ('checkout_dir', 'blas_pair_dir', 'cpus'):
        require(isinstance(build[key], str) and build[key],
                'explicit build.' + key + ' required')
    require(config['image_tag'] == derived_tag(config),
            'image tag differs from snapshot/rust derivation')
    neutrality = config['neutrality']
    require(set(neutrality) == NEUTRALITY_KEYS, 'explicit neutrality keys required')
    bindings = {}
    _bound(config['inspect_script'], config['inspect_script_sha256'],
           'inspect_script', bindings)
    for key in ('prepare_script', 'runner_script'):
        _bound(neutrality[key], neutrality[key + '_sha256'], key, bindings)
    inputs = neutrality['preparation_inputs']
    require(set(inputs) == PREPARATION_INPUT_KEYS,
            'explicit neutrality preparation inputs required')
    require(re.fullmatch(r'[a-z0-9][a-z0-9-]{0,55}', inputs['attempt']) is not None,
            'filename-safe neutrality attempt required')
    require(isinstance(inputs['preserved_failures'], list),
            'preserved failures must be an explicit list')
    for key in ('profile_dir', 'frozen_short', 'frozen_long', 'baseline_provenance',
                'remote_template', 'phase_collector', 'pins_file',
                'runtime_binary_path'):
        require(isinstance(inputs[key], str) and inputs[key],
                'preparation input missing: ' + key)
    for key in ('profile_dir', 'frozen_short', 'frozen_long', 'baseline_provenance',
                'remote_template', 'phase_collector', 'pins_file'):
        require(Path(inputs[key]).is_absolute(),
                'preparation input must be absolute: ' + key)
    require(Path(inputs['runtime_binary_path']).is_absolute(),
            'runtime binary path must be absolute (it names a container path)')
    for key in ('frozen_short', 'frozen_long', 'baseline_provenance',
                'remote_template', 'phase_collector', 'pins_file'):
        require(Path(inputs[key]).exists(),
                'preparation input missing on disk: ' + key)
    for rank in ('0', '1'):
        original = Path(inputs['profile_dir']) / f'cand-rank{rank}-create-command.json'
        require(original.is_file(), 'profile descriptor missing: ' + str(original))
        bindings[f'profile_rank{rank}'] = {'path': str(original),
                                           'sha256': sha(original)}
    for key in ('frozen_short', 'frozen_long', 'baseline_provenance',
                'remote_template', 'phase_collector', 'pins_file'):
        bound = Path(inputs[key])
        if key in ('frozen_short', 'frozen_long'):
            bound = bound / 'manifest.json'
            require(bound.is_file(), 'frozen suite manifest missing: ' + key)
        bindings[key] = {'path': str(bound), 'sha256': sha(bound)}
    cycle_root = Path(config['evidence_root']) / config['cycle_id']
    return {'config': config, 'config_path': str(config_path),
            'config_sha256': sha(config_path), 'bindings': bindings,
            'cycle_root': str(cycle_root),
            'prep_root': str(cycle_root / '04-neutrality' / 'preparation')}


def preflight_commands(plan):
    config = plan['config']
    env = dict(os.environ, JUMP=config['jump'],
               SPARK0=config['ranks']['0'], SPARK1=config['ranks']['1'])
    return env, ['bash', config['inspect_script'], '--out',
                 str(Path(plan['cycle_root']) / '00-preflight')]


def build_command(plan):
    build = plan['config']['build']
    remote = (f"cd {build['checkout_dir']} && "
              f"env BLAS131_PAIR_DIR={build['blas_pair_dir']} "
              f"bash tests/exl3_validation/build_image.sh "
              f"{build['source_snapshot_sha256']} {build['rust_version']} {build['cpus']}")
    return ssh_argv(plan['config']['jump'], plan['config']['build_host'], remote)


def transfer_commands(plan):
    tag = plan['config']['image_tag']
    return (ssh_argv(plan['config']['jump'], plan['config']['ranks']['0'],
                     f'docker save {tag}'),
            ssh_argv(plan['config']['jump'], plan['config']['ranks']['1'], 'docker load'))


def inspect_command(plan, rank):
    tag = plan['config']['image_tag']
    return ssh_argv(plan['config']['jump'], plan['config']['ranks'][rank],
                    f'docker image inspect --format {{{{.Id}}}} {tag}')


def manifest_command(plan, rank):
    tag = plan['config']['image_tag']
    return ssh_argv(plan['config']['jump'], plan['config']['ranks'][rank],
                    f'docker run --rm --entrypoint /bin/cat {tag} {MANIFEST_PATH}')


def neutrality_commands(plan, prepare_config_path):
    neutrality = plan['config']['neutrality']
    return ([sys.executable, neutrality['prepare_script'], '--mode', 'neutrality',
             '--config', str(prepare_config_path)],
            [sys.executable, neutrality['runner_script'], '--plan',
             str(Path(plan['prep_root']) / 'neutrality-plan.json'), '--execute'])


def preparation_config(plan, fingerprint):
    """Fill the image-identity fields prepare.py binds from verified outputs."""
    inputs = plan['config']['neutrality']['preparation_inputs']
    return {'schema': NEUTRALITY_SCHEMA, 'prep_root': plan['prep_root'],
            'evidence_root': plan['config']['evidence_root'],
            'image_tag': plan['config']['image_tag'],
            'image_id': fingerprint['image_id'],
            'runtime_source': fingerprint['git_commit'],
            'source_snapshot_sha256': fingerprint['source_snapshot_sha256'],
            'build_manifest': fingerprint['manifest_path'],
            **{key: inputs[key] for key in PREPARATION_INPUT_KEYS}}


def stage_preflight(plan, runner):
    root = Path(plan['cycle_root']) / '00-preflight'
    root.mkdir(parents=True, exist_ok=False)
    env, argv = preflight_commands(plan)
    result = runner.run(argv, env=env, timeout=TIMEOUTS['preflight'])
    require(result['rc'] == 0, 'preflight inspection failed: '
            + result['stderr'].decode(errors='replace')[-400:])
    files = {}
    for rank in ('0', '1'):
        path = root / f'spark{rank}.txt'
        require(path.is_file(), 'preflight receipt missing for spark' + rank)
        require(gpu_processes_idle(path.read_text()),
                'spark' + rank + ' still owns GPU processes')
        files[f'spark{rank}.txt'] = sha(path)
    return {'files': files, 'both_ranks_idle': True}


def stage_build(plan, runner):
    root = Path(plan['cycle_root']) / '01-build'
    root.mkdir(parents=True, exist_ok=False)
    argv = build_command(plan)
    write_json(root / 'build-command.json', {'argv': argv})
    result = runner.run(argv, env=dict(os.environ), timeout=TIMEOUTS['build'])
    (root / 'build-stdout.log').write_bytes(result['stdout'])
    (root / 'build-stderr.log').write_bytes(result['stderr'])
    write_json(root / 'build-exit.json', {'returncode': result['rc']})
    require(result['rc'] == 0,
            'no-cache build failed (rc=' + str(result['rc']) + ')')
    image_id, labels = parse_build_image_id(result['stdout'].decode(errors='replace'))
    return {'image_id': image_id, 'labels': labels,
            'stdout_sha256': sha(root / 'build-stdout.log'),
            'command_sha256': sha(root / 'build-command.json')}


def stage_transfer(plan, runner, expected_image_id):
    root = Path(plan['cycle_root']) / '02-transfer'
    root.mkdir(parents=True, exist_ok=False)
    save_argv, load_argv = transfer_commands(plan)
    stream = runner.stream(save_argv, load_argv, timeout=TIMEOUTS['transfer'])
    ids = {}
    for rank in ('0', '1'):
        result = runner.run(inspect_command(plan, rank), env=dict(os.environ),
                            timeout=TIMEOUTS['inspect'])
        require(result['rc'] == 0, 'image inspect failed on spark' + rank)
        ids[rank] = result['stdout'].decode().strip()
        require(ids[rank] == expected_image_id,
                'spark' + rank + ' image id differs from build')
    receipt = {'image_id': expected_image_id,
               'image_tag': plan['config']['image_tag'],
               'transport': 'docker save streamed via workstation to docker load',
               'same_image_both_ranks': ids['0'] == ids['1'],
               **stream}
    write_json(root / 'transfer.json', receipt)
    return {'receipt_path': str(root / 'transfer.json')}


def stage_fingerprint(plan, runner, expected_image_id):
    root = Path(plan['cycle_root']) / '03-fingerprint'
    root.mkdir(parents=True, exist_ok=False)
    manifests = {}
    for rank in ('0', '1'):
        result = runner.run(manifest_command(plan, rank), env=dict(os.environ),
                            timeout=TIMEOUTS['inspect'])
        require(result['rc'] == 0, 'manifest extraction failed on spark' + rank)
        manifest = json.loads(result['stdout'])
        require(manifest.get('schema') == MANIFEST_SCHEMA,
                'unexpected build manifest schema on spark' + rank)
        require(manifest.get('source_snapshot_sha256')
                == plan['config']['build']['source_snapshot_sha256'],
                'manifest source snapshot differs on spark' + rank)
        require(re.fullmatch(r'[0-9a-f]{40}', manifest.get('git_commit', '')) is not None,
                'manifest git commit missing on spark' + rank)
        manifests[rank] = manifest
    require(manifests['0'] == manifests['1'], 'build manifests differ between ranks')
    manifest = manifests['0']
    manifest_path = root / 'build-manifest.json'
    write_json(manifest_path, manifest)
    binaries = manifest.get('binaries', {})
    require('spark' in binaries
            and re.fullmatch(r'[0-9a-f]{64}', binaries['spark'].get('sha256', '')) is not None,
            'manifest lacks the spark binary hash')
    receipt = {'image_id': expected_image_id,
               'manifest_path': str(manifest_path.resolve()),
               'manifest_sha256': sha(manifest_path),
               'git_commit': manifest['git_commit'],
               'source_snapshot_sha256': manifest['source_snapshot_sha256'],
               'spark_binary_sha256': binaries['spark']['sha256'],
               'ptx_modules': len(manifest.get('ptx_modules', [])),
               'ranks_equal': True}
    write_json(root / 'fingerprint.json', receipt)
    return receipt


def stage_neutrality(plan, runner, fingerprint):
    root = Path(plan['cycle_root']) / '04-neutrality'
    root.mkdir(parents=True, exist_ok=False)
    prep_config_path = root / 'neutrality-preparation-config.json'
    write_json(prep_config_path, preparation_config(plan, fingerprint))
    prep_argv, run_argv = neutrality_commands(plan, prep_config_path)
    write_json(root / 'neutrality-commands.json',
               {'prepare_argv': prep_argv, 'runner_argv': run_argv,
                'preparation_config_sha256': sha(prep_config_path)})
    result = runner.run(prep_argv, env=dict(os.environ), timeout=TIMEOUTS['prepare'])
    (root / 'prepare-stdout.log').write_bytes(result['stdout'])
    (root / 'prepare-stderr.log').write_bytes(result['stderr'])
    require(result['rc'] == 0, 'neutrality preparation failed')
    result = runner.run(run_argv, env=dict(os.environ), timeout=TIMEOUTS['neutrality'])
    (root / 'runner-stdout.log').write_bytes(result['stdout'])
    (root / 'runner-stderr.log').write_bytes(result['stderr'])
    require(result['rc'] == 0, 'neutrality execution failed')
    proof_path = Path(plan['prep_root']) / plan['config']['neutrality'] \
        ['preparation_inputs']['attempt'] / 'neutrality-proof.json'
    require(proof_path.is_file(), 'neutrality proof missing after execution')
    proof = read(proof_path)
    require(proof.get('passed') is True and proof.get('rows_total') == 32,
            'full32 neutrality proof did not pass')
    return {'proof_path': str(proof_path), 'proof_sha256': sha(proof_path)}


def execute(plan):
    root = Path(plan['cycle_root'])
    root.mkdir(parents=True, exist_ok=False)
    runner = SubprocessRunner()
    stages = {}
    stage_names = ('preflight', 'build', 'transfer', 'fingerprint', 'neutrality')
    current = stage_names[0]
    try:
        for current in stage_names:
            if current == 'preflight':
                stages[current] = stage_preflight(plan, runner)
            elif current == 'build':
                stages[current] = stage_build(plan, runner)
            elif current == 'transfer':
                stages[current] = stage_transfer(
                    plan, runner, stages['build']['image_id'])
            elif current == 'fingerprint':
                stages[current] = stage_fingerprint(
                    plan, runner, stages['build']['image_id'])
            else:
                stages[current] = stage_neutrality(plan, runner,
                                                   stages['fingerprint'])
    except Exception as error:
        if not (root / 'failure.json').exists():
            write_json(root / 'failure.json',
                       {'schema': 'atlas-exl3-image-cycle-failure-v1',
                        'stage': current, 'recorded_stages': stages,
                        'error': str(error)})
        raise
    stage_files = {}
    for path in sorted(root.rglob('*')):
        if path.is_file():
            stage_files[str(path.relative_to(root))] = sha(path)
    receipt = {'schema': RECEIPT_SCHEMA, 'cycle_id': plan['config']['cycle_id'],
               'config_path': plan['config_path'],
               'config_sha256': plan['config_sha256'],
               'image_tag': plan['config']['image_tag'],
               'image_id': stages['build']['image_id'],
               'git_commit': stages['fingerprint']['git_commit'],
               'source_snapshot_sha256':
                   stages['fingerprint']['source_snapshot_sha256'],
               'stage_files': stage_files,
               'neutrality': stages['neutrality'],
               'completed': True}
    write_json(root / 'image-cycle-receipt.json', receipt)
    return receipt


def dry_run(plan):
    prep_config_path = Path(plan['prep_root']).parent / 'neutrality-preparation-config.json'
    prep_argv, run_argv = neutrality_commands(plan, prep_config_path)
    save_argv, load_argv = transfer_commands(plan)
    return {'mode': 'dry-run', 'cycle_root': plan['cycle_root'],
            'image_tag': plan['config']['image_tag'],
            'config_sha256': plan['config_sha256'],
            'bound_inputs': plan['bindings'],
            'stages': [
                {'name': 'preflight', 'argv': preflight_commands(plan)[1]},
                {'name': 'build', 'argv': build_command(plan)},
                {'name': 'transfer', 'save_argv': save_argv, 'load_argv': load_argv},
                {'name': 'fingerprint',
                 'argv': [manifest_command(plan, rank) for rank in ('0', '1')]},
                {'name': 'neutrality', 'prepare_argv': prep_argv,
                 'runner_argv': run_argv}]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    plan = validate(args.config)
    if args.dry_run:
        print(text(dry_run(plan)), end='')
    else:
        print(text(execute(plan)), end='')


if __name__ == '__main__':
    main()
