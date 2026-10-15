# SPDX-License-Identifier: AGPL-3.0-only
"""One candidate lifecycle, full short16 then full long; append-only and explicit."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / 'tests/exl3_validation'))
import c2_ab_driver as driver
from c2_remote_adapter import RemoteAdapter
from parity_protocol import load_artifact, read_raw
from attest_runtime_maps import read_pins
from prepare import read, require, sha, write

SNAPSHOT_SOURCE = '''import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);expected=int(sys.argv[3])
if (root/'capture-enabled').exists(): raise ValueError('prior capture marker must be absent')
files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('logits-*')}
sidecars=sorted(name for name in files if name.endswith('.json'))
if sidecars != [f'logits-{i:06d}.json' for i in range(expected)]: raise ValueError('full short suite counter required')
with out.open('x') as target: json.dump({'head_start_sequence':expected,'prior_head_files':files},target,indent=2)
'''


def compare_suite(frozen, actual):
    old, a = frozen
    new, b = actual
    for key in ('cases', 'tokens_sha256', 'vocab', 'request_policy', 'thresholds'):
        require(old[key] == new[key], 'full neutrality artifact identity differs: ' + key)
    for key in ('serve_flags', 'execution', 'checkpoint_revision',
                'checkpoint_sha256', 'tokenizer_sha256', 'config_sha256'):
        require(old['provenance'][key] == new['provenance'][key], 'full neutrality numerical profile differs: ' + key)
    policies = []
    for manifest in (old, new):
        policy = copy.deepcopy(manifest['provenance']['native_numerical_policy'])
        require(policy.pop('source_identity') == manifest['provenance']['engine_build']['source_sha256'],
                'numerical policy source identity differs from actual source')
        policies.append(policy)
    require(policies[0] == policies[1], 'full neutrality numerical policy differs beyond source identity')
    rows, equal, text_equal = 0, 0, True
    row_results = []
    for case in old['cases']:
        name = case['id']
        require(a[name].dtype == b[name].dtype == np.dtype('float32') and a[name].shape == b[name].shape,
                'full neutrality row/vocabulary dtype or shape differs')
        text_equal = text_equal and old['generated_text'][name] == new['generated_text'][name]
        for row in range(a[name].shape[0]):
            first, second = a[name][row].astype('<f4').tobytes(), b[name][row].astype('<f4').tobytes()
            same = first == second
            rows += 1
            equal += same
            row_results.append({'case': name, 'row': row, 'byte_equal': same,
                                'frozen_sha256': hashlib.sha256(first).hexdigest(),
                                'actual_sha256': hashlib.sha256(second).hexdigest()})
    require(rows == 16, 'full suite must contain all16 rows')
    return {'rows': rows, 'rows_byte_equal': equal, 'generated_text_equal': text_equal,
            'passed': equal == rows and text_equal, 'row_results': row_results}


def verify_runner_sources(plan):
    module_paths = {'prepare.py': Path(__file__).with_name('prepare.py'),
                    'run_neutrality.py': Path(__file__),
                    'c2_remote_adapter.py': REPO / 'tests/exl3_validation/c2_remote_adapter.py'}
    require(set(plan['runner_bindings']) == set(module_paths), 'explicit local runner source bindings required')
    for name, path in module_paths.items():
        require(plan['runner_bindings'][name] == sha(path), 'local neutrality runner source changed: ' + name)


def execution_receipts(plan, root):
    """Bind both container identities, exact create argv, providers and binaries."""
    root = Path(root)
    lifecycle = read(root / 'lifecycle.json')
    for field in ('containers', 'owned_names', 'started'):
        require(set(lifecycle[field]) == {'0', '1'}, 'both-rank lifecycle ownership required')
    require(lifecycle['containers'] == lifecycle['started'], 'started container identities differ')
    require(len(set(lifecycle['containers'].values())) == 2 and all(
        len(value) == 64 and all(c in '0123456789abcdef' for c in value)
        for value in lifecycle['containers'].values()), 'full distinct container identities required')
    prep = Path(plan['preparation_root'])
    adapter = RemoteAdapter(read(prep / 'remote-config.json'))
    descriptors = {rank: driver.parse_serve_descriptor(path) for rank, path in plan['rank_descriptors'].items()}
    context = {'arm': {'capture_dir': str(root / 'logits'), 'knob': {'engaged': False}},
               'run_root': root, 'records_dir': root.parent.parent / 'records' / root.name,
               'gated_norm_dir': root.parent.parent / 'gated_norm' / root.name,
               'create_commands': {rank: {'file': f'create-command-rank{rank}.json', 'image': plan['image_tag']}
                                   for rank in ('0', '1')}}
    receipts = []
    pins = read_pins(prep / 'tooling/cublas131.sha256')
    for rank in ('0', '1'):
        cid, name = lifecycle['containers'][rank], lifecycle['owned_names'][rank]
        expected = adapter._create_args(rank, context)
        require(name == expected[expected.index('--name') + 1], 'owned container name differs from source-bound role')
        require(read(root / f'create-command-rank{rank}.json') == read(plan['rank_descriptors'][rank]),
                'rendered neutrality descriptor differs from registered source')
        require(read(root / f'live-create-rank{rank}.json') == expected, 'actual create descriptor differs from registered source')
        commands = [(path, read(path)) for path in sorted(root.glob(f'remote-*-rank{rank}.json'))]
        definitions = [('image', ['docker', 'image', 'inspect', '--format', '{{.Id}}', plan['image_tag']]),
                       ('create', expected), ('start', ['docker', 'start', cid]),
                       ('binary', ['docker', 'exec', name, 'sha256sum', plan['runtime_binary_path']]),
                       ('stop', ['docker', 'stop', cid]), ('wait', ['docker', 'wait', cid])]
        provider_args = adapter.config['provider_checks'][rank]['cublas']
        require(provider_args == adapter.config['provider_checks'][rank]['cublaslt'], 'atomic provider checker argv must match')
        definitions += [('provider', [arg.replace('{container}', cid) for arg in provider_args])]
        for kind, argv in definitions:
            matching = [(path, meta) for path, meta in commands if meta['argv'] == argv]
            require(len(matching) == (2 if kind == 'provider' else 1), 'required command receipt missing or duplicated: ' + kind)
            for path, meta in matching:
                require(meta['returncode'] == 0, 'required command did not succeed: ' + kind)
                stdout = path.with_suffix('.stdout')
                text = stdout.read_text().strip()
                if kind == 'image':
                    require(text == plan['runtime']['image'], 'actual image receipt differs')
                elif kind == 'create':
                    require(text == cid, 'actual created container receipt differs')
                elif kind == 'binary':
                    require(text.split() == [plan['runtime']['binary_sha256'], plan['runtime_binary_path']], 'actual binary receipt differs')
                elif kind == 'wait':
                    require(text == '0', 'neutrality container did not exit cleanly')
                elif kind == 'provider':
                    report = json.loads(text)
                    require(report['target'] == report['container'] == cid and report['ok'] is True
                            and report['issues'] == [] and type(report['pid']) is int and report['pid'] > 0,
                            'actual provider target/verdict differs')
                    require(report['pins'] == str(prep / 'tooling/cublas131.sha256')
                            and report['maps_source'] == f"/proc/{report['pid']}/maps"
                            and report['fs_root'] == f"/proc/{report['pid']}/root", 'provider mapping source differs')
                    libraries = {item['name']: item for item in report['libraries']}
                    require(set(libraries) == {'libcublas.so.13', 'libcublasLt.so.13'}, 'both pinned providers required')
                    for library, item in libraries.items():
                        require(item['path'] == pins[library]['path'] and item['ok'] is True
                                and item['sha256'] == item['pinned_sha256'] == pins[library]['sha256'], 'provider library differs from pins')
                receipts.append({'rank': rank, 'kind': kind, 'path': str(path), 'sha256': sha(path),
                                 'stdout_sha256': sha(stdout), 'stderr_sha256': sha(path.with_suffix('.stderr'))})
    return receipts


def verify_proof(proof, plan, plan_path):
    """Revalidate raw captures and both complete suites; PASS booleans alone are insufficient."""
    verify_runner_sources(plan)
    require(proof['schema'] == 'atlas-c2-full-neutrality-proof-v1' and proof['passed'] is True
            and proof['runtime'] == plan['runtime'] and proof['neutrality_plan_sha256'] == sha(plan_path)
            and proof['suite_order'] == ['short16', 'long'] and proof['rows_total'] == 32,
            'full source-bound neutrality proof required')
    prep = Path(plan['preparation_root'])
    root = prep / plan['attempt'] / 'runs/arm-a-repeat-00'
    for name, digest in plan['file_sha256'].items():
        require(sha(prep / name) == digest, 'neutrality prepared source changed')
    lifecycle = root / 'lifecycle.json'
    require(proof['lifecycle_sha256'] == sha(lifecycle) and not read(lifecycle)['teardown_errors']
            and proof['lifecycle_teardown_errors'] == [], 'neutrality lifecycle proof incomplete')
    require(proof['execution_receipts'] == execution_receipts(plan, root),
            'actual execution receipt fingerprints changed')
    for suite, start, module in (('short16', 0, 'collect_logits.py'), ('long', 16, 'collect_phase_logits.py')):
        source = plan['frozen'][suite]
        for name, digest in source['sha256'].items():
            require(sha(Path(source['path']) / name) == digest, 'neutrality frozen source changed')
        actual = proof['actual_artifacts'][suite]
        require(actual['path'] == str(root / suite), 'neutrality actual artifact path mismatch')
        for name, digest in actual['sha256'].items():
            require(sha(root / suite / name) == digest, 'neutrality actual artifact fingerprint mismatch')
        loaded = load_artifact(root / suite)
        manifest = loaded[0]
        require(manifest['provenance']['engine_build'] == plan['runtime'], 'neutrality actual runtime mismatch')
        require(manifest['harness_sha256'] == {name: plan['file_sha256']['tooling/' + name]
                for name in (module, 'parity_protocol.py')}, 'neutrality executed collector module mismatch')
        result = compare_suite(load_artifact(source['path']), loaded)
        require(result['passed'] and result == proof['suite_results'][suite], 'neutrality suite result does not recompute')
        for index, capture in enumerate(manifest['captures']):
            sidecar = root / 'logits' / f'logits-{start + index:06d}.json'
            require(capture['capture'] == str(sidecar), 'neutrality phase counter/path mismatch')
            row, meta = read_raw(sidecar)
            require(meta == capture['capture_metadata'] and sha(sidecar.parent / meta['file']) == capture['raw_sha256']
                    and hashlib.sha256(row.astype('<f4').tobytes()).hexdigest() == capture['row_sha256'],
                    'neutrality raw capture differs from collected row')
    snapshot = read(root / 'long' / 'pre_registration.json')['phase_snapshot']
    require(snapshot['head_start_sequence'] == 16 and set(snapshot['prior_head_files']) == {
        f'logits-{index:06d}.{suffix}' for index in range(16) for suffix in ('json', 'f32')},
        'full short-to-long phase snapshot required')
    require(all(sha(root / 'logits' / name) == digest for name, digest in snapshot['prior_head_files'].items()),
            'short-to-long prior captures changed')
    return proof


def validate_plan(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = read(plan_path)
    prep = Path(plan['preparation_root'])
    require(plan['schema'] == 'atlas-c2-full-neutrality-plan-v1' and plan_path == prep / 'neutrality-plan.json',
            'source-bound full neutrality plan required')
    require(plan['collection_order'] == ['short16', 'long'] and plan['request_count'] == 32,
            'both full suites in frozen order required')
    for name, digest in plan['file_sha256'].items():
        require(sha(prep / name) == digest, 'prepared source fingerprint mismatch: ' + name)
    verify_runner_sources(plan)
    for source in plan['frozen'].values():
        for name, digest in source['sha256'].items():
            require(sha(Path(source['path']) / name) == digest, 'frozen artifact fingerprint mismatch')
    require(not (prep / plan['attempt']).exists(), 'fresh neutrality attempt required')
    RemoteAdapter(read(prep / 'remote-config.json'))
    for rank, descriptor in plan['rank_descriptors'].items():
        parsed = driver.parse_serve_descriptor(descriptor)
        require(parsed['source_snapshot'] == plan['runtime']['source_sha256'][:12], 'image source snapshot differs')
        require(not any(key.startswith('ATLAS_EXL3_UPSTREAM_') or key.startswith('ATLAS_EXL3_FFN_')
                        or key in driver.KNOB_ENV_KEYS for key in parsed['env']), 'neutrality must capture logits only')
    return plan


def run(plan_path, runner=None):
    plan = validate_plan(plan_path)
    prep = Path(plan['preparation_root'])
    config = read(prep / 'remote-config.json')
    adapter = RemoteAdapter(config) if runner is None else RemoteAdapter(config, runner)
    attempt = prep / plan['attempt']
    attempt.mkdir()
    root = attempt / 'runs/arm-a-repeat-00'
    records = attempt / 'records/arm-a-repeat-00'
    upstream = attempt / 'gated_norm/arm-a-repeat-00'
    for path in (root, records, upstream, root / 'logits', root / 'ffn'):
        path.mkdir(parents=True, exist_ok=False)
    descriptors = {rank: driver.parse_serve_descriptor(path) for rank, path in plan['rank_descriptors'].items()}
    arm = {'arm': 'a', 'descriptors': descriptors, 'capture_dir': str(root / 'logits'),
           'knob': {'engaged': False, 'substitute_dir': None, 'variant': None}}
    commands = driver._render_create_commands(arm, root, {})
    inputs = root / 'collector-inputs'
    inputs.mkdir()
    provenance = copy.deepcopy(read(prep / 'provenance.json'))
    provenance['environment'] = dict(descriptors['0']['env'])
    provenance['rank_environments'] = {rank: descriptors[rank]['env'] for rank in ('0', '1')}
    write(inputs / 'provenance.json', provenance)
    for name in ('cases-short16.json', 'cases-long.json', 'thresholds.json'):
        write(inputs / name, read(prep / name))
    write(inputs / 'short-request-ids.json', config['request_ids'][:16])
    context = {'arm': arm, 'run_root': root, 'records_dir': records,
               'gated_norm_dir': upstream, 'create_commands': commands}
    adapter.active = context
    try:
        # Fresh immutable harness staging; host-file mutations precede GPU ownership.
        for rank in ('0', '1'):
            adapter._stage(rank, prep / 'tooling')
            phase_hash = adapter._run(rank, ['sha256sum', str(prep / 'tooling/collect_phase_logits.py')]).stdout.decode().split()
            require(phase_hash and phase_hash[0] == plan['file_sha256']['tooling/collect_phase_logits.py'],
                    'remote frozen phase collector fingerprint mismatch')
        with adapter.lifecycle(**context):
            # Bind actual binaries to the exact named containers created by this lifecycle.
            for rank in ('0', '1'):
                create = read(root / f'live-create-rank{rank}.json')
                name = create[create.index('--name') + 1]
                binary = adapter._run(rank, ['docker', 'exec', name, 'sha256sum', plan['runtime_binary_path']]).stdout.decode().split()
                require(binary and binary[0] == plan['runtime']['binary_sha256'], 'actual running binary hash differs from build')
            for suite, collector in (('short16', 'collect_logits.py'), ('long', 'collect_phase_logits.py')):
                destination = root / suite
                args = config['remote_python'] + [str(prep / 'tooling' / collector)]
                values = [('endpoint', plan['collector_inputs']['endpoint']), ('model', plan['collector_inputs']['model']),
                          ('cases', inputs / f'cases-{suite}.json'), ('capture-dir', root / 'logits'),
                          ('out', destination), ('provenance', inputs / 'provenance.json'),
                          ('thresholds', inputs / 'thresholds.json'), ('timeout', plan['collector_inputs']['timeouts'][suite])]
                if suite == 'short16':
                    values += [('request-ids', inputs / 'short-request-ids.json')]
                else:
                    adapter._run('0', config['remote_python'] + ['-c', SNAPSHOT_SOURCE, str(root / 'logits'),
                                                              str(root / 'long-snapshot.json'), '16'])
                    values += [('phase-snapshot', root / 'long-snapshot.json')]
                for key, value in values:
                    args += ['--' + key, str(value)]
                try:
                    adapter._run('0', args)
                finally:
                    destination.mkdir(exist_ok=True)
                    adapter._retrieve('0', destination, destination)
            adapter._retrieve('0', root, root / 'remote-runroot')
        results = {suite: compare_suite(load_artifact(plan['frozen'][suite]['path']), load_artifact(root / suite))
                   for suite in ('short16', 'long')}
        lifecycle = read(root / 'lifecycle.json')
        require(not lifecycle['teardown_errors'], 'neutrality lifecycle did not release both ranks')
        proof = {'schema': 'atlas-c2-full-neutrality-proof-v1', 'runtime': plan['runtime'],
                 'neutrality_plan_sha256': sha(plan_path), 'suite_order': ['short16', 'long'],
                 'rows_total': 32, 'suite_results': results, 'passed': all(item['passed'] for item in results.values()),
                 'lifecycle_teardown_errors': lifecycle['teardown_errors'], 'lifecycle_sha256': sha(root / 'lifecycle.json'),
                 'execution_receipts': execution_receipts(plan, root),
                 'actual_artifacts': {suite: {'path': str(root / suite), 'sha256': {
                     name: sha(root / suite / name) for name in ('manifest.json', 'logits.npz', 'pre_registration.json')}}
                                      for suite in ('short16', 'long')},
                 'scope': 'full32 frozen rows and generated text; no new statistical qualification verdict'}
        if proof['passed']:
            verify_proof(proof, plan, plan_path)
        write(attempt / 'neutrality-proof.json', proof)
        return proof
    except Exception as exc:
        write(attempt / 'failure.json', {'schema': 'atlas-c2-neutrality-failure-v1', 'exception': str(exc)})
        raise
    finally:
        adapter.active = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.dry_run:
        plan = validate_plan(args.plan)
        print(json.dumps({'mode': 'dry-run', 'image': plan['runtime']['image'], 'rows': 32,
                          'collection_order': plan['collection_order'], 'attempt': plan['attempt']}))
    else:
        proof = run(args.plan)
        print(json.dumps({'passed': proof['passed'], 'rows_total': proof['rows_total']}))


if __name__ == '__main__':
    main()
