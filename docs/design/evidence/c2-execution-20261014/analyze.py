# SPDX-License-Identifier: AGPL-3.0-only
"""Read-only C2 evidence analysis; writes one fresh diagnostic-only report.

The full ref04..ref12 artifacts are validated before projecting identical
arithmetic/longer cases in memory. No artifact or scoring rule is rewritten.
Native reference own-output is authoritative; archived weights are never read.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / 'tests' / 'exl3_validation'))
import c2_ab_driver as driver
import c2_score
import make_norm_fixtures
from parity_protocol import token_sha, validate_cases
from statistical_gate import load_run, merge_run


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def project_run(source, selected_cases, source_binding):
    """Strict projection from an already validated one-suite full artifact."""
    selected_cases = validate_cases(selected_cases)
    require(len(source['manifests']) == 1, 'projection requires exactly one short16 suite')
    original = source['manifests'][0]
    lookup = {case['id']: case for case in original['cases']}
    require(all(lookup.get(case['id']) == case for case in selected_cases),
            'projection cases must exactly match the original token cases')
    manifest = copy.deepcopy(original)
    manifest['cases'] = copy.deepcopy(selected_cases)
    manifest['tokens_sha256'] = token_sha(selected_cases)
    manifest['generated_text'] = {case['id']: original['generated_text'][case['id']]
                                  for case in selected_cases}
    manifest['captures'] = [record for case in selected_cases
                            for record in original['captures'] if record['case'] == case['id']]
    manifest['diagnostic_projection'] = {
        'scope': 'in-memory identical-case projection; original artifact unchanged',
        'source': source_binding, 'original_tokens_sha256': original['tokens_sha256'],
        'original_logits_sha256': original['logits_sha256'],
        'selected_tokens_sha256': manifest['tokens_sha256']}
    arrays = {case['id']: source['rows'][(0, case['id'])].copy() for case in selected_cases}
    return merge_run([(manifest, arrays)])


def read_capture(path):
    """Validate each tensor declaration and exact byte count before comparison."""
    path = Path(path)
    meta = read_json(path)
    require(meta.get('schema') == 'atlas-exl3-ffn-v1' and meta.get('tp_world') == 2,
            'capture schema/world mismatch: ' + str(path))
    for field in ('layer', 'tp_rank', 'invocation'):
        require(type(meta.get(field)) is int and meta[field] >= 0,
                'capture identity invalid: ' + str(path))
    require(meta['tp_rank'] in (0, 1) and isinstance(meta.get('stage'), str),
            'capture stage/rank invalid')
    stem = f"ffn-L{meta['layer']}-rank{meta['tp_rank']}-{meta['invocation']:06d}-{meta['stage']}"
    require(path.name == stem + '.json', 'capture filename/metadata identity mismatch')
    sizes = {'bf16-le': 2, 'f32-le': 4, 'i32-le': 4}
    tensors = {}
    require(isinstance(meta.get('tensors'), list) and meta['tensors'], 'capture tensors absent')
    for tensor in meta['tensors']:
        name, dtype, shape = tensor['name'], tensor['dtype'], tensor['shape']
        require(isinstance(name, str) and name not in tensors and dtype in sizes,
                'capture tensor name/dtype invalid')
        require(isinstance(shape, list) and shape and all(type(x) is int and x > 0 for x in shape),
                'capture shape invalid')
        require(tensor['file'] == stem + '-' + name + '.bin', 'capture tensor filename mismatch')
        payload = path.parent / tensor['file']
        expected = math.prod(shape) * sizes[dtype]
        require(tensor['bytes'] == expected and payload.stat().st_size == expected,
                'capture tensor byte count mismatch')
        tensors[name] = {'dtype': dtype, 'shape': shape, 'sha256': sha(payload),
                         'path': str(payload), 'payload': payload.read_bytes()}
    return meta, tensors


def compare_native(capture_path, archive_dir, archive_name, binding_name, rank, invocation):
    """Compare BF16 bit patterns to a bound native archive, with exact shape."""
    meta, tensors = read_capture(capture_path)
    require(meta['layer'] == 0 and meta['tp_rank'] == rank and meta['invocation'] == invocation,
            'native comparison capture identity mismatch')
    tensor_name = 'output' if meta['stage'] == 'attention_local_output' else 'gated'
    require(meta['stage'] in ('attention_local_output', 'gated_norm')
            and set(tensors) == {tensor_name}, 'native comparison stage/tensor mismatch')
    archive_dir = Path(archive_dir)
    binding_path = archive_dir / binding_name
    binding = read_json(binding_path)
    require(binding['rank'] == rank and binding['invocation'] == invocation,
            'native archive rank/invocation mismatch')
    if meta['stage'] == 'gated_norm':
        require(binding['schema'] == 'norm-replay-fixtures-v1' and archive_name == 'expected'
                and binding_name == 'fixture.json', 'norm archive binding schema/role mismatch')
        entry = binding['fixtures'][archive_name]
    else:
        require(binding['schema'] == 'atlas-local-o-proj-binding-v1' and binding['engine'] == 'reference'
                and binding['layer'] == 0 and archive_name == 'own-output'
                and binding_name == 'binding.json' and binding['projection']['tp_pre_reduce'] is True,
                'native o_proj binding schema/role mismatch')
        entry = binding['files'][archive_name]
    archive_path = archive_dir / (archive_name + '.npy')
    require(entry['file'] == archive_path.name and entry['sha256'] == sha(archive_path),
            'native archive fingerprint mismatch')
    array = np.load(archive_path, allow_pickle=False)
    tensor = tensors[tensor_name]
    require(array.dtype == np.dtype('<u2') and entry['dtype'] == tensor['dtype'] == 'bf16-le',
            'native archive/capture dtype mismatch')
    require(list(array.shape) == entry['shape'], 'native archive declared shape mismatch')
    if meta['stage'] == 'gated_norm':
        norm = binding['candidate_binding']
        archive_shape = [norm['tokens'], norm['heads'], norm['dim']]
        capture_shape = [norm['tokens'], norm['heads'] * norm['dim']]
        require(entry['shape'] == archive_shape and tensor['shape'] == capture_shape,
                'native norm archive/capture shape mismatch')
    else:
        require(entry['shape'] == tensor['shape'], 'native archive/capture shape mismatch')
    native_bytes = array.astype('<u2', copy=False).tobytes(order='C')
    return {'rank': rank, 'invocation': invocation, 'capture': str(capture_path),
            'capture_sha256': sha(capture_path), 'payload_sha256': tensor['sha256'],
            'archive': str(archive_path), 'archive_sha256': sha(archive_path),
            'binding_sha256': sha(binding_path),
            'native_payload_sha256': hashlib.sha256(native_bytes).hexdigest(),
            'dtype': tensor['dtype'], 'shape': tensor['shape'], 'archive_shape': entry['shape'],
            'byte_equal': tensor['payload'] == native_bytes}


def capture_inventory(root, layers):
    root = Path(root)
    inventory = {}
    for path in sorted(root.glob('*.json')):
        meta, tensors = read_capture(path)
        require(meta['layer'] in layers and meta['invocation'] < 8, 'unexpected layer/invocation capture')
        key = (meta['layer'], meta['tp_rank'], meta['invocation'], meta['stage'])
        require(key not in inventory, 'duplicate capture identity')
        inventory[key] = (path, tensors)
    for layer in layers:
        for rank in (0, 1):
            for invocation in range(8):
                require(any(key[:3] == (layer, rank, invocation) for key in inventory),
                        f'missing capture pair L{layer}/r{rank}/i{invocation}: {root}')
    return inventory


def stage_differences(first_root, second_root, layers):
    first, second = capture_inventory(first_root, layers), capture_inventory(second_root, layers)
    require(set(first) == set(second), 'A/B capture stage inventories differ')
    comparisons = []
    for key in sorted(first):
        path_a, a = first[key]
        path_b, b = second[key]
        require(set(a) == set(b), 'A/B capture tensor inventories differ')
        for name in sorted(a):
            require(a[name]['dtype'] == b[name]['dtype'] and a[name]['shape'] == b[name]['shape'],
                    'A/B capture dtype/shape differ')
            comparisons.append({'layer': key[0], 'rank': key[1], 'invocation': key[2],
                                'stage': key[3], 'tensor': name, 'dtype': a[name]['dtype'],
                                'shape': a[name]['shape'], 'a_sidecar_sha256': sha(path_a),
                                'b_sidecar_sha256': sha(path_b),
                                'a_payload_sha256': a[name]['sha256'], 'b_payload_sha256': b[name]['sha256'],
                                'byte_equal': a[name]['payload'] == b[name]['payload']})
    return comparisons


def compare_final_logits(actual, expected):
    """Every selected final-vocabulary row, compared as original f32 bytes."""
    require(actual['cases'] == expected['cases'] and set(actual['rows']) == set(expected['rows']),
            'observer neutrality cases/rows differ')
    for field in c2_score.IDENTITY:
        require(actual['manifests'][0][field] == expected['manifests'][0][field],
                'observer neutrality identity differs: ' + field)
    rows = []
    for key in sorted(actual['cases']):
        a, b = actual['rows'][key], expected['rows'][key]
        require(a.shape == b.shape, 'observer neutrality vocabulary/row shapes differ')
        for row in range(a.shape[0]):
            first, second = a[row].astype('<f4').tobytes(), b[row].astype('<f4').tobytes()
            rows.append({'case': key[1], 'row': row, 'byte_equal': first == second,
                         'live_sha256': hashlib.sha256(first).hexdigest(),
                         'archived_sha256': hashlib.sha256(second).hexdigest()})
    require(len(rows) == 8, 'observer neutrality requires every eight diagnostic rows')
    return rows


def validate_live_registration(prep, binding):
    """Bind live arm data to the explicit pre-execution runtime/tooling profile."""
    registered = ['run-config.json', 'remote-config.json', 'provenance.json', 'preparation.json',
                  'candidate-rank0-create-command.json', 'candidate-rank1-create-command.json',
                  'thresholds.json', 'tooling/collect_logits.py', 'tooling/parity_protocol.py',
                  'tooling/attest_runtime_maps.py', 'tooling/cublas131.sha256']
    for name in registered:
        require(binding['file_sha256'][name] == sha(prep / name), 'registered file fingerprint mismatch: ' + name)
    run_config = driver.load_run_config(prep / 'run-config.json')
    remote = read_json(prep / 'remote-config.json')
    provenance = read_json(prep / 'provenance.json')
    require(provenance['engine_build']['commit'] == binding['runtime_source']
            and provenance['engine_build']['image'] == binding['runtime_image_id'],
            'registered runtime commit/image mismatch')
    require(all(remote['image_ids'][rank] == binding['runtime_image_id'] for rank in ('0', '1')),
            'remote image IDs differ from registered runtime')
    require(remote['remote_harness_dir'] == str(prep / 'tooling'), 'remote harness path registration mismatch')
    for name, digest in remote['remote_harness_sha256'].items():
        require(digest == binding['file_sha256']['tooling/' + name], 'remote module registration mismatch')
    descriptors = {}
    for rank in ('0', '1'):
        path = prep / f'candidate-rank{rank}-create-command.json'
        require(run_config['rank_descriptors'][rank] == str(path), 'registered descriptor path mismatch')
        descriptors[rank] = driver.parse_serve_descriptor(path)
        require(descriptors[rank]['source_snapshot'] == provenance['engine_build']['source_sha256'][:12],
                'registered descriptor source snapshot mismatch')
    # These immutable hashes were verified with git show at the named commits.
    driver_hash = '5e61f268b7dd1cacf715baa7ebf86a0372247ebf1347b7eed02be6acad9d9dc1'
    adapter_hash = '7949e770f5072103d484e2cc0f380cf77d99dd8e931a7eb5fa0ac62ef51c0aca'
    require(binding['harness_source_commit'] == '54fa51a4' and sha(driver.__file__) == driver_hash,
            'live driver differs from registered 54fa source')
    require(sha(REPO / 'tests/exl3_validation/c2_remote_adapter.py') == adapter_hash,
            'live adapter differs from separately bound 473 correction')
    return {'provenance': provenance, 'descriptors': descriptors, 'run_config': run_config,
            'collector_modules': {name: remote['remote_harness_sha256'][name]
                                  for name in ('collect_logits.py', 'parity_protocol.py')},
            'driver': {'commit': '54fa51a4', 'sha256': driver_hash},
            'adapter': {'commit': '473aa8bc', 'sha256': adapter_hash}}


def validate_preserved_failure_bindings(root, binding):
    """Keep failed historical attempts in their original roots and identities."""
    from prepare import preserved_failures
    entries = preserved_failures(Path(root), binding['preserved_failures'])
    return [dict(entry, failure=read_json(Path(entry['attempt']) / 'failure.json')) for entry in entries]


def validate_fresh_neutrality_binding(root, prep, binding, registration):
    from run_neutrality import verify_proof
    proof_path = Path(binding['runtime_neutrality_receipt'])
    require(proof_path.is_absolute() and proof_path.resolve().is_relative_to(Path(root).resolve())
            and sha(proof_path) == binding['runtime_neutrality_sha256'], 'registered full neutrality proof fingerprint/path mismatch')
    plan_path = proof_path.parent.parent / 'neutrality-plan.json'
    plan = read_json(plan_path)
    proof = read_json(proof_path)
    require(proof['runtime'] == registration['provenance']['engine_build']
            and binding['neutrality_runner_bindings'] == plan['runner_bindings'],
            'full neutrality runtime/runner differs from live C2 registration')
    modules = {'prepare.py': Path(__file__).with_name('prepare.py'),
               'run_neutrality.py': Path(__file__).with_name('run_neutrality.py')}
    require(set(binding['preparation_module_bindings']) == set(modules), 'explicit fresh C2 preparation module bindings required')
    for name, path in modules.items():
        require(binding['preparation_module_bindings'][name] == sha(path), 'fresh C2 preparation module changed: ' + name)
    verify_proof(proof, plan, plan_path)
    return {'path': str(proof_path), 'sha256': sha(proof_path), 'rows': proof['rows_total'], 'runtime': proof['runtime']}


def analyze(root, attempt):
    root, attempt = Path(root).resolve(), Path(attempt).resolve()
    prep = attempt.parent
    binding = read_json(prep / 'execution-plan.json')
    require(binding['preparation_root'] == str(prep) and prep.is_relative_to(root),
            'attempt must belong to its explicit registered preparation root')
    require(binding['reference_members'] == [f'ref{x:02d}' for x in range(4, 13)],
            'all nine frozen reference members required')
    require(binding['file_sha256']['cases.json'] == sha(prep / 'cases.json'),
            'registered diagnostic case fingerprint mismatch')
    registration = validate_live_registration(prep, binding)
    full_neutrality_binding = validate_fresh_neutrality_binding(root, prep, binding, registration)
    preserved_failures = validate_preserved_failure_bindings(root, binding)
    selected = validate_cases(read_json(prep / 'cases.json'))
    request_ids = read_json(prep / 'request-ids.json')
    require(binding['file_sha256']['request-ids.json'] == sha(prep / 'request-ids.json'),
            'registered request schedule fingerprint mismatch')
    require([case['id'] for case in selected] == ['arithmetic', 'longer']
            and sum(len(case['continuation_ids']) + 1 for case in selected) == 8,
            'registered eight-request diagnostic slice required')
    receipt = read_json(attempt / 'receipt.json')
    require(receipt['schema'] == 'atlas-c2-ab-receipt-v1' and receipt['repeats_per_arm'] == 2,
            'complete two-repeat driver receipt required')
    require(receipt['harness_sha256']['c2_ab_driver.py'] == registration['driver']['sha256'],
            'driver receipt harness hash differs from registered source')
    fixture_dir = prep / receipt['variant']
    require(binding['file_sha256'][receipt['variant'] + '/manifest.json'] == sha(fixture_dir / 'manifest.json'),
            'registered fixture manifest fingerprint mismatch')
    fixture = make_norm_fixtures.validate_manifest(fixture_dir)
    require(len(fixture['entries']) == 16, 'all16 rank/row fixtures required')
    arm_runs = {}
    for arm in ('a', 'b'):
        records = receipt['arms'][arm]
        require(len(records) == 2 and receipt['byte_determinism'][arm]['byte_deterministic'],
                'both arms must be byte deterministic over both repeats')
        arm_runs[arm] = []
        for repeat, record in enumerate(records):
            expected_name = f'arm-{arm}-repeat-{repeat:02d}'
            require(record['run'] == expected_name and record['arm'] == arm and record['repeat'] == repeat,
                    'driver repeat identity mismatch')
            require(record['gated_norm_capture_dir'] == str(attempt / 'gated_norm' / expected_name),
                    'driver upstream path binding mismatch')
            collection = attempt / 'runs' / expected_name / 'collection'
            require(record['collection']['destination'] == str(collection)
                    and record['collection']['manifest_file_sha256'] == sha(collection / 'manifest.json'),
                    'driver collection binding mismatch')
            require(record['fingerprints']['descriptors'] == registration['descriptors'],
                    'live receipt descriptors differ from registration')
            require(record['fingerprints']['harness_sha256'] == {
                'c2_ab_driver.py': registration['driver']['sha256'],
                'collect_logits.py': registration['collector_modules']['collect_logits.py']},
                'live receipt module hashes differ from registration')
            loaded = load_run(str(collection))
            manifest = loaded['manifests'][0]
            live_provenance = {key: value for key, value in manifest['provenance'].items() if key not in ('environment', 'rank_environments')}
            registered_provenance = {key: value for key, value in registration['provenance'].items() if key not in ('environment', 'rank_environments')}
            require(live_provenance == registered_provenance, 'live runtime provenance differs from registration')
            run_root = attempt / 'runs' / expected_name
            run_env = {'ATLAS_EXL3_LOGITS_DIR': str(run_root / 'logits'),
                       'ATLAS_EXL3_UPSTREAM_DUMP_DIR': str(attempt / 'gated_norm' / expected_name),
                       'ATLAS_EXL3_UPSTREAM_DUMP_LAYERS': ','.join(map(str, binding['capture_layers']['upstream'])),
                       'ATLAS_EXL3_FFN_DUMP_DIR': str(run_root / 'ffn'),
                       'ATLAS_EXL3_FFN_DUMP_LAYERS': ','.join(map(str, binding['capture_layers']['ffn']))}
            if arm == 'b':
                run_env.update({driver.KNOB_ENV: str(fixture_dir), driver.RECORD_ENV: str(attempt / 'records' / expected_name)})
            require(record['collection']['run_env'] == run_env, 'live injected environment differs from registration')
            expected_environment = dict(registration['provenance']['environment'])
            for key in (*driver.KNOB_ENV_KEYS, driver.UPSTREAM_ENV, driver.UPSTREAM_LAYERS_ENV):
                expected_environment.pop(key, None)
            expected_environment.update(registration['descriptors']['0']['env'])
            expected_environment.update(run_env)
            expected_ranks = {rank: dict(registration['descriptors'][rank]['env'], **run_env) for rank in ('0', '1')}
            require(manifest['provenance']['environment'] == expected_environment
                    and manifest['provenance']['rank_environments'] == expected_ranks,
                    'live rank environment differs from registered descriptor/injected profile')
            config = registration['run_config']
            require(manifest['endpoint'] == config['endpoint'] and manifest['model'] == config['model']
                    and manifest['timeout_seconds'] == config['timeout_seconds']
                    and manifest['thresholds'] == read_json(prep / 'thresholds.json'),
                    'live collector inputs differ from registered run configuration')
            require(manifest['harness_sha256'] == registration['collector_modules'],
                    'live collector/protocol modules differ from registration')
            require(loaded['manifests'][0]['cases'] == selected, 'arm cases differ from registered diagnostic slice')
            require(loaded['manifests'][0]['request_ids'] == request_ids
                    and [capture['request_id'] for capture in loaded['manifests'][0]['captures']] == request_ids,
                    'arm request IDs differ from registered ordered schedule')
            require(record['collection']['logits_sha256'] == sha(collection / 'logits.npz'),
                    'driver logits fingerprint mismatch')
            require(record['collection']['row_sha256'] == [row['row_sha256']
                    for row in loaded['manifests'][0]['captures']], 'driver row fingerprint mismatch')
            arm_runs[arm].append(loaded)
            require(not read_json(attempt / 'runs' / expected_name / 'lifecycle.json')['teardown_errors'],
                    'lifecycle teardown/evidence retrieval incomplete')
            if arm == 'b':
                located = driver.locate_substitution_records(attempt / 'records' / expected_name,
                                                            fixture['entries'], receipt['variant'])
                require(located['records_found'] == 16 and not located['post_hoc_gap']
                        and not located['unconsumed_entries'], 'not every fixture entry consumed')
                require(located == record['substitution_records'], 'substitution receipt inventory drift')
        require(driver._byte_determinism(records)['byte_deterministic'], 'driver determinism does not recompute')
    neutrality_source = root / 'gate-v2-qualification-20261013-luna1' / 'attempts' / 'v2cand01' / 'attempt-files' / 'short16'
    neutrality_binding = {'source': str(neutrality_source), 'sha256': {
        name: sha(neutrality_source / name) for name in ('manifest.json', 'logits.npz', 'pre_registration.json')}}
    archived_candidate = project_run(load_run(str(neutrality_source)), selected, neutrality_binding)
    observer_neutrality = [{'repeat': repeat, 'rows': compare_final_logits(run, archived_candidate)}
                          for repeat, run in enumerate(arm_runs['a'])]
    observer_ok = all(row['byte_equal'] for repeat in observer_neutrality for row in repeat['rows'])
    alignment = driver._pre_hash_alignment(receipt['arms']['a'], receipt['arms']['b'])
    require(alignment['aligned'] and not alignment['a_side_conflicts'] and len(alignment['entries']) == 16
            and alignment == receipt['pre_hash_alignment'], 'A/B pre-substitution hash alignment failed')
    baseline, first_measurement, differences = [], [], []
    for repeat in range(2):
        run_a, run_b = (f'arm-{arm}-repeat-{repeat:02d}' for arm in ('a', 'b'))
        upstream_a, upstream_b = (attempt / 'gated_norm' / run for run in (run_a, run_b))
        for rank in (0, 1):
            for row in range(8):
                prefix = f'ffn-L0-rank{rank}-{row:06d}-'
                archive = root / 'q2-norm-replay-20261011' / 'fixtures' / f'rank{rank}-row{row}'
                base = compare_native(upstream_a / (prefix + 'gated_norm.json'), archive,
                                      'expected', 'fixture.json', rank, row)
                baseline.append(dict(base, repeat=repeat))
                native = root / 'q2-c1-binding-2026013' / 'reference' / f'rank{rank}-row{row}'
                first = compare_native(upstream_b / (prefix + 'attention_local_output.json'), native,
                                       'own-output', 'binding.json', rank, row)
                first_measurement.append(dict(first, repeat=repeat))
        differences.append({'repeat': repeat,
                            'upstream': stage_differences(upstream_a, upstream_b, [0]),
                            'ffn': stage_differences(attempt / 'runs' / run_a / 'ffn',
                                                     attempt / 'runs' / run_b / 'ffn', [0, 3])})
    baseline_ok = all(row['byte_equal'] for row in baseline)
    first_ok = all(row['byte_equal'] for row in first_measurement)
    scores, ref_bindings = [], []
    if baseline_ok and first_ok and observer_ok:
        refs, ref_bindings = [], []
        for name in binding['reference_members']:
            source = Path(binding['reference_root']) / name / 'short16'
            require(source == root / 'statistical-gate-20261011' / 'runs' / name / 'short16',
                    'frozen reference source root mismatch')
            hashes = {file: sha(source / file) for file in ('manifest.json', 'logits.npz', 'pre_registration.json')}
            original = load_run(str(source))
            source_binding = {'source': str(source), 'sha256': hashes}
            refs.append(project_run(original, selected, source_binding))
            ref_bindings.append(source_binding)
        scores = []
        for repeat in range(2):
            result = c2_score.score(binding['reference_members'], refs, arm_runs['a'][repeat],
                                    arm_runs['b'][repeat], f'arm-a-repeat-{repeat:02d}', f'arm-b-repeat-{repeat:02d}')
            require(len(result['rows']) == 8 and result['inputs']['reference_count'] == 9,
                    'scorer did not cover every registered row/reference')
            scores.append(result)
    interpretation = ('diagnostic-intervention-bound' if baseline_ok and first_ok and observer_ok else
                      'STOP-reopen-C1' if not first_ok else
                      'STOP-observer-neutrality-mismatch' if not observer_ok else 'STOP-baseline-archive-mismatch')
    return {'schema': 'atlas-c2-execution-analysis-v1', 'scope': 'diagnostic-only; no Q2 gate or performance claim',
            'interpretation': {'status': interpretation, 'causal_interpretation_allowed': baseline_ok and first_ok and observer_ok,
                               'authority': 'outer measurement guards govern interpretation of nested SSOT score reports'},
            'attempt': str(attempt), 'receipt_sha256': sha(attempt / 'receipt.json'),
            'execution_binding': binding, 'execution_binding_sha256': sha(prep / 'execution-plan.json'),
            'live_registration': registration, 'preserved_failures': preserved_failures, 'full_neutrality_binding': full_neutrality_binding,
            'analyzer_sha256': sha(__file__), 'reference_projections': ref_bindings,
            'baseline_archive_checks': baseline,
            'observer_neutrality': {'scope': 'only the eight selected rows; historical image proof remains separate',
                                    'source': neutrality_binding, 'checks': observer_neutrality}, 'first_native_o_proj_measurement': first_measurement,
            'stage_byte_differences': differences, 'repeat_pair_scores': scores,
            'scoring_skipped_reason': (None if baseline_ok and first_ok and observer_ok else
                                       'native first measurement, baseline archive, or observer neutrality control failed')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('root', 'attempt', 'out'):
        parser.add_argument('--' + flag, type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), 'analysis output must be fresh')
    report = analyze(args.root, args.attempt)
    with args.out.open('x') as output:
        output.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'report': str(args.out), 'interpretation': report['interpretation']}))


if __name__ == '__main__':
    main()
