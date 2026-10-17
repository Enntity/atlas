# SPDX-License-Identifier: AGPL-3.0-only
"""Offline, append-only preparation for full neutrality and a fresh C2 campaign."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / 'tests/exl3_validation'))
import c2_ab_driver as driver
from c2_remote_adapter import RemoteAdapter
from parity_protocol import load_artifact, token_sha
import make_norm_fixtures

NEUTRAL_KEYS = {'schema', 'prep_root', 'evidence_root', 'image_tag', 'image_id', 'runtime_source',
               'source_snapshot_sha256', 'build_manifest', 'profile_dir', 'frozen_short', 'frozen_long',
               'baseline_provenance', 'remote_template', 'phase_collector', 'pins_file', 'attempt',
               'preserved_failures', 'runtime_binary_path'}
C2_KEYS = {'schema', 'prep_root', 'evidence_root', 'neutrality_prep', 'neutrality_proof', 'prior_c2_prep',
           'attempt', 'preserved_failures'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as target:
        target.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def bind_runtime(config, build):
    require(re.fullmatch(r'sha256:[0-9a-f]{64}', config['image_id']) is not None, 'full explicit image ID required')
    require(re.fullmatch(r'[0-9a-f]{40}', config['runtime_source']) is not None, 'full runtime commit required')
    require(re.fullmatch(r'[0-9a-f]{64}', config['source_snapshot_sha256']) is not None, 'full source snapshot required')
    require(build['schema'] == 'atlas-exl3-build-manifest-v1'
            and build['git_commit'] == config['runtime_source']
            and build['source_snapshot_sha256'] == config['source_snapshot_sha256'],
            'build manifest source identity differs from explicit binding')
    require('src-' + config['source_snapshot_sha256'][:12] in config['image_tag'], 'image tag snapshot mismatch')
    binary = build['binaries']['spark']['sha256']
    require(re.fullmatch(r'[0-9a-f]{64}', binary) is not None, 'explicit built binary hash required')
    return {'commit': config['runtime_source'], 'binary_sha256': binary,
            'source_sha256': config['source_snapshot_sha256'], 'image': config['image_id']}


def rewrite_descriptor(args, image_tag, capture):
    """Numerical command/environment are unchanged; image and observer path are explicit."""
    args = copy.deepcopy(args)
    require(args[:2] == ['docker', 'create'] and args.count('serve') == 1, 'registered create/serve descriptor required')
    args[args.index('serve') - 1] = image_tag
    filtered, cursor = [], 0
    while cursor < len(args):
        if args[cursor] == '--volume' and args[cursor + 1].split(':')[1] == '/tmp/capture':
            cursor += 2
        else:
            filtered.append(args[cursor])
            cursor += 1
    args = filtered
    found = False
    for index, arg in enumerate(args[:-1]):
        if arg == '--env' and args[index + 1].startswith('ATLAS_EXL3_LOGITS_DIR='):
            args[index + 1] = 'ATLAS_EXL3_LOGITS_DIR=' + str(capture)
            found = True
        if arg == '--volume':
            pieces = args[index + 1].split(':')
            if len(pieces) in (2, 3) and pieces[1] == '/tmp/capture':
                args[index + 1] = str(capture) + ':' + str(capture)
    require(found, 'registered logits observer env missing')
    return args


def preserved_failures(root, entries):
    require(isinstance(entries, list) and entries, 'explicit prior failure bindings required')
    for entry in entries:
        require(set(entry) == {'attempt', 'phase', 'failure_sha256', 'lifecycles'}, 'prior failure binding keys invalid')
        attempt = Path(entry['attempt'])
        require(attempt.is_absolute() and attempt.resolve().is_relative_to(root.resolve()), 'prior failure must stay inside evidence root')
        require(entry['phase'] in ('failed-before-create', 'failed-first-treatment-request'), 'prior failure phase invalid')
        require(sha(attempt / 'failure.json') == entry['failure_sha256'], 'prior failure fingerprint mismatch')
        require(entry['lifecycles'], 'prior failure lifecycle evidence required')
        for item in entry['lifecycles']:
            require(set(item) == {'path', 'sha256'}, 'prior lifecycle binding keys invalid')
            path = Path(item['path'])
            require(path.resolve().is_relative_to(attempt.resolve()) and sha(path) == item['sha256'],
                    'prior lifecycle fingerprint/path mismatch')
            lifecycle = read(path)
            require(not lifecycle['teardown_errors'], 'prior failure teardown incomplete')
            if entry['phase'] == 'failed-before-create':
                require(not lifecycle['containers'] and not lifecycle['owned_names'] and not lifecycle['started'],
                        'prior first failure was not before create')
        require(not (attempt / 'receipt.json').exists(), 'preserved failure has a complete receipt')
    return copy.deepcopy(entries)


def _fresh(config):
    root, prep = Path(config['evidence_root']).resolve(), Path(config['prep_root']).resolve()
    require(root.is_dir() and prep.is_relative_to(root) and not prep.exists(), 'fresh prep inside existing evidence root required')
    require(driver.ATTEMPT_NAME.fullmatch(config['attempt']) is not None, 'filename safe attempt required')
    return root, prep


def _tooling(prep, phase):
    target = prep / 'tooling'
    target.mkdir()
    for name in ('collect_logits.py', 'parity_protocol.py', 'attest_runtime_maps.py'):
        shutil.copyfile(REPO / 'tests/exl3_validation' / name, target / name)
    if phase is not None:
        shutil.copyfile(phase, target / 'collect_phase_logits.py')
    return {name: sha(target / name) for name in ('collect_logits.py', 'parity_protocol.py', 'attest_runtime_maps.py')}


def _remote(template, prep, image, ids, pins):
    result = copy.deepcopy(template)
    result['image_ids'] = {rank: image for rank in ('0', '1')}
    result['request_ids'] = ids
    result['remote_harness_dir'] = str(prep / 'tooling')
    result['remote_harness_sha256'] = {name: sha(prep / 'tooling' / name)
                                       for name in ('collect_logits.py', 'parity_protocol.py', 'attest_runtime_maps.py')}
    checker = ['sudo', '-n', 'python3', str(prep / 'tooling/attest_runtime_maps.py'),
               '{container}', '--pins', str(pins)]
    result['provider_checks'] = {rank: {provider: list(checker) for provider in ('cublas', 'cublaslt')}
                                  for rank in ('0', '1')}
    RemoteAdapter(result)  # validate the complete concrete configuration without executing it
    return result


def _schedule(suites, prefix):
    rows = []
    for name, cases in suites:
        for case in cases:
            for step in range(len(case['continuation_ids']) + 1):
                tokens = case['prompt_ids'] + case['continuation_ids'][:step]
                rows.append({'suite': name, 'case': case['id'], 'step': step,
                             'schedule_id': f'{prefix}-req-{len(rows):02d}',
                             'prefix_sha256': hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest(),
                             'request_id_policy': 'explicit-header' if name == 'short16' else 'frozen-phase-collector-managed'})
    return rows


def prepare_neutrality(config):
    require(set(config) == NEUTRAL_KEYS and config['schema'] == 'atlas-c2-neutrality-preparation-config-v1', 'explicit neutrality config keys/schema required')
    root, prep = _fresh(config)
    failures = preserved_failures(root, config['preserved_failures'])
    runtime = bind_runtime(config, read(config['build_manifest']))
    frozen = {name: load_artifact(config[key]) for name, key in (('short16', 'frozen_short'), ('long', 'frozen_long'))}
    require(all(sum(len(case['continuation_ids']) + 1 for case in data[0]['cases']) == 16
                for data in frozen.values()), 'full16 short and full16 long frozen suites required')
    provenance = read(config['baseline_provenance'])
    for manifest, _ in frozen.values():
        for key in ('serve_flags', 'execution', 'native_numerical_policy', 'checkpoint_revision', 'checkpoint_sha256', 'tokenizer_sha256', 'config_sha256'):
            require(provenance[key] == manifest['provenance'][key], 'baseline provenance differs from frozen source: ' + key)
    phase = Path(config['phase_collector'])
    require(sha(phase) == frozen['long'][0]['collector_sha256']
            == frozen['long'][0]['harness_sha256']['collect_phase_logits.py'],
            'phase collector must match the frozen long artifact source')
    require(phase.is_file() and phase.read_text().splitlines()[0] == '# SPDX-License-Identifier: AGPL-3.0-only', 'explicit licensed frozen phase collector required')
    require(Path(config['runtime_binary_path']).is_absolute(), 'explicit absolute runtime binary path required')
    require(frozen['short16'][0]['endpoint'] == frozen['long'][0]['endpoint']
            and frozen['short16'][0]['model'] == frozen['long'][0]['model'], 'frozen suite endpoint/model mismatch')
    prep.mkdir()
    try:
        _tooling(prep, phase)
        shutil.copyfile(config['pins_file'], prep / 'tooling/cublas131.sha256')
        shutil.copyfile(config['build_manifest'], prep / 'build-manifest.json')
        prior_policy_source = provenance['native_numerical_policy']['source_identity']
        provenance['engine_build'] = runtime
        provenance['native_numerical_policy']['source_identity'] = runtime['source_sha256']
        write(prep / 'provenance.json', provenance)
        capture = prep / config['attempt'] / 'runs/arm-a-repeat-00/logits'
        descriptors = {}
        profile = Path(config['profile_dir'])
        for rank in ('0', '1'):
            path = prep / f'candidate-rank{rank}-create-command.json'
            original = profile / f'cand-rank{rank}-create-command.json'
            write(path, rewrite_descriptor(read(original), config['image_tag'], capture))
            parsed = driver.parse_serve_descriptor(path)
            if rank == '0':
                require(parsed['command'] == provenance['serve_flags'], 'profile serve command differs from frozen numerical policy')
            descriptors[rank] = str(path)
        schedule = _schedule([(name, frozen[name][0]['cases']) for name in ('short16', 'long')], config['attempt'])
        write(prep / 'schedule.json', schedule)
        for name, (manifest, _) in frozen.items():
            write(prep / f'cases-{name}.json', manifest['cases'])
        require(frozen['short16'][0]['thresholds'] == frozen['long'][0]['thresholds'], 'suite thresholds differ')
        write(prep / 'thresholds.json', frozen['short16'][0]['thresholds'])
        write(prep / 'remote-config.json', _remote(read(config['remote_template']), prep, config['image_id'],
              [row['schedule_id'] for row in schedule], prep / 'tooling/cublas131.sha256'))
        plan = {'schema': 'atlas-c2-full-neutrality-plan-v1', 'preparation_root': str(prep), 'attempt': config['attempt'],
                'runtime': runtime, 'image_tag': config['image_tag'], 'rank_descriptors': descriptors,
                'runtime_binary_path': config['runtime_binary_path'],
                'prior_policy_source_identity': prior_policy_source,
                'runner_bindings': {'prepare.py': sha(__file__), 'run_neutrality.py': sha(Path(__file__).with_name('run_neutrality.py')),
                                    'c2_remote_adapter.py': sha(REPO / 'tests/exl3_validation/c2_remote_adapter.py')},
                'collector_inputs': {'endpoint': frozen['short16'][0]['endpoint'],
                                     'model': frozen['short16'][0]['model'],
                                     'timeouts': {name: frozen[name][0]['timeout_seconds'] for name in ('short16', 'long')}},
                'frozen': {name: {'path': str(Path(config[key]).resolve()), 'sha256': {
                    file: sha(Path(config[key]) / file) for file in ('manifest.json', 'logits.npz', 'pre_registration.json')}}
                           for name, key in (('short16', 'frozen_short'), ('long', 'frozen_long'))},
                'phase_collector_source': {'path': str(phase.resolve()), 'sha256': sha(phase)},
                'preserved_failures': failures, 'collection_order': ['short16', 'long'],
                'request_count': 32, 'file_sha256': {str(path.relative_to(prep)): sha(path) for path in sorted(prep.rglob('*')) if path.is_file()}}
        write(prep / 'neutrality-plan.json', plan)
        return plan
    except Exception as exc:
        write(prep / 'preparation-failure.json', {'exception': str(exc)})
        raise


def prepare_c2(config):
    require(set(config) == C2_KEYS and config['schema'] == 'atlas-c2-fresh-preparation-config-v1', 'explicit fresh C2 config keys/schema required')
    root, prep = _fresh(config)
    failures = preserved_failures(root, config['preserved_failures'])
    neutral = Path(config['neutrality_prep']).resolve()
    plan = read(neutral / 'neutrality-plan.json')
    require(plan['preparation_root'] == str(neutral), 'neutrality preparation root binding differs')
    require(Path(config['neutrality_proof']).resolve() == neutral / plan['attempt'] / 'neutrality-proof.json',
            'neutrality proof must use its declared lifecycle output role')
    proof = read(config['neutrality_proof'])
    require(proof['schema'] == 'atlas-c2-full-neutrality-proof-v1' and proof['passed'] is True
            and proof['runtime'] == plan['runtime'] and proof['neutrality_plan_sha256'] == sha(neutral / 'neutrality-plan.json')
            and proof['rows_total'] == 32 and proof['suite_order'] == ['short16', 'long']
            and proof['lifecycle_teardown_errors'] == [], 'completed full32-row bound neutrality proof required')
    require(proof['suite_results']['short16']['rows'] == proof['suite_results']['long']['rows'] == 16
            and all(item['passed'] for item in proof['suite_results'].values()), 'both full frozen suites must pass')
    from run_neutrality import verify_proof
    verify_proof(proof, plan, neutral / 'neutrality-plan.json')
    prior = Path(config['prior_c2_prep']).resolve()
    old = read(prior / 'execution-plan.json')
    for name in ('cases.json', 'thresholds.json', 'request-ids.json', 'schedule.json', 'provenance.json',
                 'candidate-rank0-create-command.json', 'candidate-rank1-create-command.json', 'run-config.json',
                 'reference-exact/manifest.json', 'v24-corrected/manifest.json'):
        require(sha(prior / name) == old['file_sha256'][name], 'prior C2 registered source fingerprint mismatch: ' + name)
    neutral_cases = {case['id']: case for case in load_artifact(plan['frozen']['short16']['path'])[0]['cases']}
    selected = read(prior / 'cases.json')
    require(all(neutral_cases.get(case['id']) == case for case in selected), 'C2 cases must be covered identically by full neutrality')
    prep.mkdir()
    try:
        for name in ('cases.json', 'thresholds.json', 'request-ids.json', 'schedule.json'):
            require(sha(prior / name) == old['file_sha256'][name], 'prior C2 input fingerprint mismatch')
            shutil.copyfile(prior / name, prep / name)
        for variant in ('reference-exact', 'v24-corrected'):
            make_norm_fixtures.validate_manifest(prior / variant)
            shutil.copytree(prior / variant, prep / variant)
        _tooling(prep, None)
        shutil.copyfile(neutral / 'tooling/cublas131.sha256', prep / 'tooling/cublas131.sha256')
        shutil.copyfile(neutral / 'tooling/cublas131.sha256', prep / 'cublas131.sha256')
        shutil.copyfile(neutral / 'build-manifest.json', prep / 'build-manifest.json')
        provenance = read(prior / 'provenance.json')
        provenance['engine_build'] = plan['runtime']
        provenance['native_numerical_policy']['source_identity'] = plan['runtime']['source_sha256']
        write(prep / 'provenance.json', provenance)
        descriptors = {}
        capture = prep / config['attempt'] / 'runs/arm-a-repeat-00/logits'
        for rank in ('0', '1'):
            path = prep / f'candidate-rank{rank}-create-command.json'
            write(path, rewrite_descriptor(read(prior / f'candidate-rank{rank}-create-command.json'), plan['image_tag'], capture))
            descriptors[rank] = str(path)
        run = read(prior / 'run-config.json')
        for field, file in (('cases_file', 'cases.json'), ('thresholds_file', 'thresholds.json'), ('provenance_file', 'provenance.json')):
            run[field] = str(prep / file)
        run.update(rank_descriptors=descriptors, capture_dir=str(prep / 'unused-capture'))
        write(prep / 'run-config.json', run)
        write(prep / 'remote-config.json', _remote(read(neutral / 'remote-config.json'), prep, plan['runtime']['image'],
              read(prep / 'request-ids.json'), prep / 'tooling/cublas131.sha256'))
        write(prep / 'preparation.json', {'schema': 'atlas-c2-postfix-preparation-v1', 'runtime': plan['runtime'],
              'neutrality_proof': str(Path(config['neutrality_proof']).resolve()), 'neutrality_proof_sha256': sha(config['neutrality_proof'])})
        binding = copy.deepcopy(old)
        binding.update(preparation_root=str(prep), runtime_source=plan['runtime']['commit'], runtime_image_id=plan['runtime']['image'],
                       runtime_neutrality_receipt=str(Path(config['neutrality_proof']).resolve()),
                       runtime_neutrality_sha256=sha(config['neutrality_proof']), preserved_failures=failures,
                       neutrality_runner_bindings=plan['runner_bindings'],
                       preparation_module_bindings={'prepare.py': sha(__file__), 'run_neutrality.py': sha(Path(__file__).with_name('run_neutrality.py'))},
                       file_sha256={str(path.relative_to(prep)): sha(path) for path in sorted(prep.rglob('*')) if path.is_file()})
        write(prep / 'execution-plan.json', binding)
        return binding
    except Exception as exc:
        write(prep / 'preparation-failure.json', {'exception': str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('neutrality', 'c2'), required=True)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    result = (prepare_neutrality if args.mode == 'neutrality' else prepare_c2)(read(args.config))
    print(json.dumps({'preparation_root': result['preparation_root']}))


if __name__ == '__main__':
    main()
