# SPDX-License-Identifier: AGPL-3.0-only
"""C2 A/B run driver: knob-off baseline versus substitution treatment (D3).

Orchestrates one C2 attempt under the durable evidence root (docs/plans/c2-tooling-spec.md
A2): arm A = knob off, arm B = knob on (--variant), both ranks, identical
serve flags and requests, >=2 repeats per arm with a byte-determinism check.
It wraps the existing collector (tests/exl3_validation/collect_logits.py) and
the registered collector-only command descriptors; it never re-implements
logit collection. The injected lifecycle/collector seams keep orchestration
CPU-testable; c2_remote_adapter.py supplies the explicit live lifecycle.

Contract (docs/plans/c2-tooling-spec.md "D3" + Lifecycle): per run the driver
creates a fresh records directory under the attempt root and exports
ATLAS_EXL3_NORM_SUBSTITUTE_RECORD_DIR for run B (run A leaves
ATLAS_EXL3_NORM_SUBSTITUTE_DIR and the record var unset entirely); repeats
never share a records dir. After each run B it locates the engine's
norm_substitution receipts (ffn-L0-rank{r}-{inv:06}-norm_substitution.json)
and reports manifest entries with no matching receipt as a post-hoc gap.
Record parsing follows the in-tree D1 producer exactly (the
atlas-exl3-ffn-v1 capture record from ffn_io.rs::capture_with_metadata with
the atlas-norm-substitution-v1 metadata nested under "binding"; see
norm_substitution.rs::PendingSubstitution::record).

A/B pre-hash alignment (spec C2): each run-B pre_substitution_sha256 must
equal the sha256 of run A's captured gated_norm payload for the same
(rank, invocation); run A emits no substitution records, so its
gated_norm capture files are the A-side source. A missing A-side capture is
reported as "a-side-capture-absent" and never silently replaced by
B-repeat-only evidence (that is b_repeat_pre_hash_determinism, reported
separately). Both arms capture gated_norm into a fresh per-run directory
exported as ATLAS_EXL3_UPSTREAM_DUMP_DIR (identically for both arms: the
knob is the one variable between them).
"""
import argparse
import copy
from contextlib import nullcontext
import datetime
import hashlib
import json
import math
import re
from pathlib import Path

import collect_logits
import make_norm_fixtures

KNOB_ENV = 'ATLAS_EXL3_NORM_SUBSTITUTE_DIR'
RECORD_ENV = 'ATLAS_EXL3_NORM_SUBSTITUTE_RECORD_DIR'
KNOB_ENV_KEYS = (KNOB_ENV, RECORD_ENV)
UPSTREAM_ENV = 'ATLAS_EXL3_UPSTREAM_DUMP_DIR'
UPSTREAM_LAYERS_ENV = 'ATLAS_EXL3_UPSTREAM_DUMP_LAYERS'
GRAPH_ENV = {'ATLAS_NO_DECODE_GRAPHS': '1', 'ATLAS_NO_DECODE_GRAPHS_MULTISEQ': '1',
             'ATLAS_DEBUG_NO_GRAPH': '1'}
RECORD_SCHEMA = 'atlas-norm-substitution-v1'
RECORD_STAGE = 'norm_substitution'
RECORD_PATTERN = re.compile(r'ffn-L0-rank(\d+)-(\d{6})-norm_substitution\.json')
GATED_NORM_CAPTURE = re.compile(r'ffn-L0-rank(\d+)-(\d{6})-gated_norm-gated\.bin')
FFN_SCHEMA = 'atlas-exl3-ffn-v1'
OUTER_KEYS = {'schema', 'layer', 'tp_rank', 'tp_world', 'invocation', 'stage',
              'stream', 'tensors', 'binding'}
BINDING_KEYS = {'schema', 'payload_stage', 'variant', 'request_id', 'invocation',
                'shape', 'fixture_file', 'fixture_sha256', 'pre_substitution_sha256'}
TENSOR_KEYS = {'name', 'dtype', 'shape', 'bytes', 'file'}
SOURCE_SNAPSHOT = re.compile(r'src-([0-9a-f]{6,})')
ATTEMPT_NAME = re.compile(r'[A-Za-z0-9_-]+')
HEX = re.compile(r'[0-9a-f]{64}')
CONFIG_KEYS = {'schema', 'endpoint', 'model', 'cases_file', 'thresholds_file',
               'provenance_file', 'capture_dir', 'timeout_seconds', 'rank_descriptors'}
CONFIG_SCHEMA = 'atlas-c2-ab-run-config-v1'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def parse_serve_descriptor(path):
    """Parse one registered collector-only create-command descriptor.

    Fingerprints recorded per rank: full image identity, its source snapshot
    (the src-* tag), and the complete serve command. The graph interlock env
    must be registered in the descriptor itself so both arms stay identical.
    """
    raw = json.loads(Path(path).read_text())
    _require(isinstance(raw, list) and all(isinstance(item, str) for item in raw),
             f'serve descriptor must be a string list: {path}')
    _require(raw[:2] == ['docker', 'create'], f'registered docker create descriptor required: {path}')
    _require(raw.count('serve') == 1, f'exactly one serve command required: {path}')
    serve_index = raw.index('serve')
    image = raw[serve_index - 1]
    _require(serve_index >= 2 and not image.startswith('-'),
             f'serve descriptor image identity missing: {path}')
    snapshot = SOURCE_SNAPSHOT.search(image)
    _require(snapshot is not None, f'serve descriptor image must carry a src-* source snapshot: {path}')
    env = {}
    for index, item in enumerate(raw[:serve_index]):
        if item == '--env':
            pair = raw[index + 1]
            _require('=' in pair, f'serve descriptor env must be KEY=VALUE: {path}')
            key, value = pair.split('=', 1)
            _require(key not in env or env[key] == value,
                     f'serve descriptor env keys must be unique: {path}')
            env[key] = value
    for key, value in GRAPH_ENV.items():
        _require(env.get(key) == value,
                 f'serve descriptor must register graph interlock {key}={value}: {path}')
    return {'file': str(path), 'sha256': _sha(path), 'image': image,
            'source_snapshot': snapshot.group(1), 'command': raw[serve_index:], 'env': env}


def _knob_env(engaged, substitute_dir, record_dir):
    if not engaged:
        return {}
    return {KNOB_ENV: str(substitute_dir), RECORD_ENV: str(record_dir)}


def arm_configurations(run_config, fixture_dir, variant):
    """Derive the two arm configurations; only the knob may differ."""
    base = {'endpoint': run_config['endpoint'], 'model': run_config['model'],
            'cases_file': run_config['cases_file'], 'thresholds_file': run_config['thresholds_file'],
            'provenance_file': run_config['provenance_file'],
            'capture_dir': run_config['capture_dir'],
            'timeout_seconds': run_config['timeout_seconds'],
            'descriptors': {rank: parse_serve_descriptor(path) for rank, path
                            in run_config['rank_descriptors'].items()}}
    arm_a = dict(base, arm='a', knob={'engaged': False, 'substitute_dir': None, 'variant': None})
    arm_b = dict(base, arm='b', knob={'engaged': True, 'substitute_dir': str(fixture_dir),
                                      'variant': variant})
    return arm_a, arm_b


def validate_arm_configurations(arm_a, arm_b):
    """Refuse mixed configurations between arms: identical serve flags,
    requests and provenance; the substitution knob is the one variable."""
    _require(arm_a['arm'] == 'a' and arm_b['arm'] == 'b', 'arm identities must be a then b')
    for field in ('endpoint', 'model', 'cases_file', 'thresholds_file', 'provenance_file',
                  'capture_dir', 'timeout_seconds'):
        _require(arm_a[field] == arm_b[field],
                 f'mixed configurations between arms: {field} differs')
    _require(set(arm_a['descriptors']) == set(arm_b['descriptors']),
             'mixed configurations between arms: rank descriptor sets differ')
    for rank in sorted(arm_a['descriptors']):
        first, second = arm_a['descriptors'][rank], arm_b['descriptors'][rank]
        for field in ('image', 'source_snapshot', 'command', 'sha256'):
            _require(first[field] == second[field],
                     f'mixed configurations between arms: serve descriptor {field} differs on rank {rank}')
        env_a = {key: value for key, value in first['env'].items() if key not in KNOB_ENV_KEYS}
        env_b = {key: value for key, value in second['env'].items() if key not in KNOB_ENV_KEYS}
        _require(env_a == env_b and set(first['env']) & set(KNOB_ENV_KEYS) == set()
                 and set(second['env']) & set(KNOB_ENV_KEYS) == set(),
                 'mixed configurations between arms: knob env must not be baked into descriptors')
    _require(not arm_a['knob']['engaged'], 'arm A must leave the substitution knob off')
    _require(arm_b['knob']['engaged'] and arm_b['knob']['substitute_dir'],
             'arm B must engage the substitution knob with a fixture directory')
    return True


def load_run_config(path):
    config = json.loads(Path(path).read_text())
    _require(isinstance(config, dict) and set(config) == CONFIG_KEYS
             and config['schema'] == CONFIG_SCHEMA,
             'run config must be an ' + CONFIG_SCHEMA + ' object')
    for field in ('endpoint', 'model', 'cases_file', 'thresholds_file', 'provenance_file',
                  'capture_dir'):
        _require(isinstance(config[field], str) and config[field],
                 f'run config {field} must be a nonempty string')
    _require(type(config['timeout_seconds']) in (int, float) and config['timeout_seconds'] > 0,
             'run config timeout_seconds must be positive')
    descriptors = config['rank_descriptors']
    _require(isinstance(descriptors, dict) and set(descriptors) == {'0', '1'},
             'run config rank_descriptors must cover ranks 0 and 1')
    return config


def locate_substitution_records(records_dir, entries, variant):
    """Post-hoc record accounting for one run B (spec verification contract).

    Manifest entries with no matching engine receipt are reported as a
    post-hoc gap, never dropped; a receipt that contradicts the manifest
    refuses the attempt.
    """
    root = Path(records_dir)
    by_key = {(entry['layer'], entry['tp_rank'], entry['invocation'], entry['tensor']): entry
              for entry in entries}
    records, matched = [], set()
    for path in sorted(root.iterdir()):
        match = RECORD_PATTERN.fullmatch(path.name)
        if not match:
            continue
        record = json.loads(path.read_text())
        _require(isinstance(record, dict) and set(record) == OUTER_KEYS
                 and record['schema'] == FFN_SCHEMA and record['stage'] == RECORD_STAGE,
                 f'substitution record schema/stage mismatch: {path}')
        rank, invocation = int(match.group(1)), int(match.group(2))
        _require(record['layer'] == 0 and type(record['tp_rank']) is int
                 and record['tp_rank'] == rank and type(record['invocation']) is int
                 and record['invocation'] == invocation,
                 f'substitution record identity mismatch: {path}')
        key = (record['layer'], record['tp_rank'], record['invocation'], 'gated')
        _require(key in by_key, f'substitution record without manifest entry: {path}')
        entry = by_key[key]
        tensors = record['tensors']
        _require(isinstance(tensors, list) and len(tensors) == 1
                 and isinstance(tensors[0], dict) and set(tensors[0]) == TENSOR_KEYS
                 and tensors[0]['name'] == 'gated' and tensors[0]['dtype'] == 'bf16-le'
                 and tensors[0]['shape'] == entry['shape']
                 and tensors[0]['bytes'] == math.prod(entry['shape']) * 2
                 and tensors[0]['file'] == path.stem + '-gated.bin',
                 f'substitution record payload declaration mismatch: {path}')
        payload = root / tensors[0]['file']
        _require(payload.is_file() and payload.stat().st_size == tensors[0]['bytes'],
                 f'substitution record payload bytes mismatch: {payload}')
        binding = record['binding']
        _require(isinstance(binding, dict) and set(binding) == BINDING_KEYS
                 and binding['schema'] == RECORD_SCHEMA
                 and binding['payload_stage'] == 'post-substitution',
                 f'substitution record payload_stage must be post-substitution: {path}')
        _require(binding['variant'] == variant and binding['request_id'] == entry['request_id']
                 and binding['invocation'] == invocation and binding['shape'] == entry['shape']
                 and binding['fixture_file'] == entry['file']
                 and binding['fixture_sha256'] == entry['sha256'],
                 f'substitution record disagrees with manifest entry: {path}')
        _require(isinstance(binding['pre_substitution_sha256'], str)
                 and HEX.fullmatch(binding['pre_substitution_sha256']),
                 f'substitution record pre-substitution sha256 missing: {path}')
        _require(_sha(payload) == entry['sha256'],
                 f'substitution record payload hash differs from fixture: {payload}')
        matched.add(key)
        records.append({'file': path.name, 'sha256': _sha(path),
                        'payload_file': tensors[0]['file'], 'payload_sha256': _sha(payload),
                        'tp_rank': rank, 'invocation': invocation,
                        'request_id': binding['request_id'],
                        'pre_substitution_sha256': binding['pre_substitution_sha256'],
                        'payload_stage': binding['payload_stage']})
    unconsumed = [{'file': by_key[key]['file'], 'tp_rank': key[1], 'invocation': key[2],
                   'request_id': by_key[key]['request_id']}
                  for key in sorted(set(by_key) - matched)]
    inventory = {path.name: _sha(path) for path in sorted(root.iterdir()) if path.is_file()}
    return {'records_found': len(records), 'records': records,
            'unconsumed_entries': unconsumed, 'post_hoc_gap': bool(unconsumed),
            'records_dir_inventory': inventory}


def _render_create_commands(arm, run_root, injected_env):
    """Per-run registered create commands with the run env injected: the
    substitution knob env for run B only, and the per-run gated_norm capture
    dir identically for both arms (it is capture bookkeeping, not a variable)."""
    rendered = {}
    for rank in sorted(arm['descriptors']):
        source = Path(arm['descriptors'][rank]['file'])
        args = json.loads(source.read_text())
        if injected_env:
            filtered, index = [], 0
            while index < len(args):
                if args[index] == '--env' and args[index + 1].split('=', 1)[0] in injected_env:
                    index += 2
                else:
                    filtered.append(args[index])
                    index += 1
            args = filtered
            serve_index = args.index('serve')
            image_index = serve_index - 1
            injected = []
            for key in sorted(injected_env):
                injected += ['--env', f'{key}={injected_env[key]}']
            args = args[:image_index] + injected + args[image_index:]
        out = run_root / f'create-command-rank{rank}.json'
        out.write_text(json.dumps(args) + '\n')
        rendered[rank] = {'file': out.name, 'sha256': _sha(out),
                          'image': arm['descriptors'][rank]['image'],
                          'source_snapshot': arm['descriptors'][rank]['source_snapshot'],
                          'command': arm['descriptors'][rank]['command'],
                          'injected_env': dict(injected_env)}
    return rendered


def _execute_run(collector, arm, run_root, records_dir, gated_norm_dir, lifecycle):
    """One collector invocation with full fingerprints; collection itself is
    entirely the wrapped collector's job (collect_logits.py)."""
    logits_dir = run_root / 'logits'
    logits_dir.mkdir()
    arm = dict(arm, capture_dir=str(logits_dir))
    knob_env = _knob_env(arm['knob']['engaged'], arm['knob']['substitute_dir'], records_dir)
    run_env = dict(knob_env)
    run_env[UPSTREAM_ENV] = str(gated_norm_dir)
    run_env['ATLAS_EXL3_LOGITS_DIR'] = str(logits_dir)
    run_env[UPSTREAM_LAYERS_ENV] = '0'
    ffn_dir = run_root / 'ffn'
    ffn_dir.mkdir()
    (ffn_dir / 'capture-enabled').write_text('')
    run_env['ATLAS_EXL3_FFN_DUMP_DIR'] = str(ffn_dir)
    run_env['ATLAS_EXL3_FFN_DUMP_LAYERS'] = '0,3'
    create_commands = _render_create_commands(arm, run_root, run_env)
    provenance = copy.deepcopy(arm['provenance'])
    environment = provenance.setdefault('environment', {})
    for key in (*KNOB_ENV_KEYS, UPSTREAM_ENV, UPSTREAM_LAYERS_ENV):
        environment.pop(key, None)
    environment.update(arm['descriptors']['0']['env'])
    environment.update(run_env)
    provenance['rank_environments'] = {
        rank: dict(arm['descriptors'][rank]['env'], **run_env)
        for rank in sorted(arm['descriptors'])}
    provenance_file = run_root / 'provenance.json'
    provenance_file.write_text(json.dumps(provenance, indent=2) + '\n')
    destination = run_root / 'collection'
    context = (nullcontext() if lifecycle is None else lifecycle(
        arm=arm, run_root=run_root, records_dir=records_dir,
        gated_norm_dir=gated_norm_dir, create_commands=create_commands))
    with context:
        collected = collector(endpoint=arm['endpoint'], model=arm['model'], cases=arm['cases'],
                              capture_dir=arm['capture_dir'], destination=destination,
                              provenance=provenance, timeout=arm['timeout_seconds'],
                              thresholds=arm['thresholds'])
    _require(isinstance(collected, dict) and isinstance(collected.get('logits_sha256'), str)
             and isinstance(collected.get('captures'), list),
             'collector must return the collect_logits manifest shape')
    row_hashes = []
    for record in collected['captures']:
        _require(isinstance(record, dict) and isinstance(record.get('row_sha256'), str),
                 'collector capture records must carry row_sha256')
        row_hashes.append(record['row_sha256'])
    manifest_file = destination / 'manifest.json'
    _require(manifest_file.is_file(), 'collector must write destination/manifest.json')
    return {'destination': str(destination), 'manifest_file_sha256': _sha(manifest_file),
            'provenance_file_sha256': _sha(provenance_file),
            'logits_sha256': collected['logits_sha256'], 'row_sha256': row_hashes,
            'knob_env': knob_env, 'run_env': run_env,
            'create_commands': create_commands}


def _byte_determinism(runs):
    baseline = runs[0]['collection']
    varying = []
    for run in runs[1:]:
        rows_differing = [index for index, (first, second)
                          in enumerate(zip(baseline['row_sha256'], run['collection']['row_sha256']))
                          if first != second]
        if (run['collection']['logits_sha256'] != baseline['logits_sha256']
                or rows_differing
                or len(run['collection']['row_sha256']) != len(baseline['row_sha256'])):
            varying.append({'run': run['run'], 'rows_differing': rows_differing,
                            'logits_sha256': run['collection']['logits_sha256']})
    return {'runs_compared': len(runs), 'byte_deterministic': not varying,
            'baseline_run': runs[0]['run'],
            'logits_sha256': [run['collection']['logits_sha256'] for run in runs],
            'varying_runs': varying}


def _b_repeat_pre_hash_determinism(runs):
    """B-repeat determinism, reported SEPARATELY from A/B alignment: the
    engine-recorded pre-substitution hashes must match across run B repeats
    for the same (rank, invocation)."""
    reference = {(item['tp_rank'], item['invocation']): item['pre_substitution_sha256']
                 for item in runs[0].get('substitution_records', {}).get('records', [])}
    mismatched = []
    for run in runs[1:]:
        current = {(item['tp_rank'], item['invocation']): item['pre_substitution_sha256']
                   for item in run.get('substitution_records', {}).get('records', [])}
        for key in sorted(set(reference) | set(current)):
            if reference.get(key) != current.get(key):
                mismatched.append({'run': run['run'], 'tp_rank': key[0], 'invocation': key[1]})
    return {'runs_compared': len(runs), 'byte_deterministic': not mismatched,
            'pre_substitution_sha256': {
                f'{rank}/{invocation}': digest
                for (rank, invocation), digest in sorted(reference.items())},
            'mismatched': mismatched}


def _a_side_gated_norm_sha256(run):
    """One run's captured gated_norm payloads keyed by (rank, invocation).
    Run A emits no substitution records; these capture files
    (ffn-L0-rank{r}-{inv:06}-gated_norm-gated.bin) are the A-side source."""
    root = Path(run['gated_norm_capture_dir'])
    found = {}
    if root.is_dir():
        for path in sorted(root.iterdir()):
            match = GATED_NORM_CAPTURE.fullmatch(path.name)
            if match:
                found[(int(match.group(1)), int(match.group(2)))] = _sha(path)
    return found


def _pre_hash_alignment(runs_a, runs_b):
    """Run A/B alignment evidence (spec C2): each run-B pre_substitution_sha256
    must equal the sha256 of run A's captured gated_norm payload for the same
    (rank, invocation). A missing A-side capture is reported as
    `a-side-capture-absent` — never silently replaced by B-repeat-only
    evidence (that is `_b_repeat_pre_hash_determinism`)."""
    a_side, a_side_conflicts = {}, []
    for run in runs_a:
        for key, digest in _a_side_gated_norm_sha256(run).items():
            if key in a_side and a_side[key] != digest:
                a_side_conflicts.append({'run': run['run'], 'tp_rank': key[0],
                                         'invocation': key[1]})
            else:
                a_side.setdefault(key, digest)
    b_pre = {}
    for run in runs_b:
        for item in run.get('substitution_records', {}).get('records', []):
            key = (item['tp_rank'], item['invocation'])
            b_pre.setdefault(key, {})[run['run']] = item['pre_substitution_sha256']
    entries = []
    for key in sorted(b_pre):
        hashes = b_pre[key]
        a_hash = a_side.get(key)
        if a_hash is None:
            status = 'a-side-capture-absent'
        elif all(digest == a_hash for digest in hashes.values()):
            status = 'match'
        else:
            status = 'mismatch'
        entries.append({'tp_rank': key[0], 'invocation': key[1], 'status': status,
                        'a_side_sha256': a_hash, 'b_pre_substitution_sha256': hashes})
    return {'source': 'run-a-gated-norm-captures', 'entries': entries,
            'aligned': bool(entries) and all(entry['status'] == 'match'
                                             for entry in entries),
            'a_side_conflicts': a_side_conflicts}


def validate_plan(plan):
    _require(plan['evidence_root'].is_dir(), 'durable evidence root must exist (docs/plans/c2-tooling-spec.md A2)')
    _require(isinstance(plan['attempt'], str) and ATTEMPT_NAME.fullmatch(plan['attempt']),
             'attempt name must be filename safe')
    _require(not (plan['evidence_root'] / plan['attempt']).exists(),
             'attempt directory must not exist: fresh attempt names only, never reuse or delete')
    _require(type(plan['repeats']) is int and plan['repeats'] >= 2,
             'at least 2 repeats per arm are required to confirm byte-determinism')
    _require(plan['variant'] in make_norm_fixtures.VARIANTS, 'unknown substitution variant')
    _require(isinstance(plan.get('cases'), list) and plan['cases'],
             'plan cases must be a nonempty list')
    for key in ('provenance', 'thresholds'):
        _require(isinstance(plan.get(key), dict), f'plan {key} must be an explicit mapping')
    fixture = make_norm_fixtures.validate_manifest(plan['fixture_dir'])
    _require(fixture['variant'] == plan['variant'],
             'fixture manifest variant differs from the requested arm variant')
    validate_arm_configurations(*plan['arm_configs'])
    return fixture


def run_ab(plan, collector=collect_logits.collect, *, lifecycle=None):
    """Execute one A/B attempt; every run is preserved, failures included."""
    fixture = validate_plan(plan)
    attempt_root = plan['evidence_root'] / plan['attempt']
    attempt_root.mkdir(parents=False, exist_ok=False)
    runs = {'a': [], 'b': []}
    try:
        (attempt_root / 'plan.json').write_text(json.dumps(
            {'schema': 'atlas-c2-ab-plan-v1', 'attempt': plan['attempt'],
             'variant': plan['variant'], 'repeats': plan['repeats'],
             'fixture_manifest_sha256': _sha(Path(plan['fixture_dir']) / 'manifest.json'),
             'fixture_entries': len(fixture['entries'])}, indent=2) + '\n')
        for arm in plan['arm_configs']:
            for repeat in range(plan['repeats']):
                run_name = f"arm-{arm['arm']}-repeat-{repeat:02d}"
                run_root = attempt_root / 'runs' / run_name
                records_dir = attempt_root / 'records' / run_name
                gated_norm_dir = attempt_root / 'gated_norm' / run_name
                run_root.mkdir(parents=True)
                records_dir.mkdir(parents=True)  # fresh per run; repeats never share
                gated_norm_dir.mkdir(parents=True)  # A-side alignment source; fresh per run
                # The engine's Exl3FfnRouter only captures when this marker
                # exists (ffn_io.rs::begin) — request gated_norm captures for
                # this run (run A: the pre-image source; run B: post-substitution).
                (gated_norm_dir / 'capture-enabled').write_text('')
                arm_run = dict(arm, provenance=plan['provenance'],
                               cases=plan['cases'], thresholds=plan['thresholds'])
                collection = _execute_run(collector, arm_run, run_root, records_dir,
                                          gated_norm_dir, lifecycle)
                receipt = {'run': run_name, 'arm': arm['arm'], 'repeat': repeat,
                           'gated_norm_capture_dir': str(gated_norm_dir),
                           'fingerprints': {
                               'descriptors': {rank: arm['descriptors'][rank]
                                               for rank in sorted(arm['descriptors'])},
                               'cases_sha256': hashlib.sha256(
                                   json.dumps(plan['cases'], sort_keys=True).encode()).hexdigest(),
                               'thresholds_sha256': _sha(arm['thresholds_file']),
                               'harness_sha256': {
                                   'c2_ab_driver.py': _sha(__file__),
                                   'collect_logits.py': _sha(Path(collect_logits.__file__))}},
                           'knob': dict(arm['knob'], record_dir=str(records_dir)),
                           'collection': collection}
                if arm['knob']['engaged']:
                    receipt['substitution_records'] = locate_substitution_records(
                        records_dir, fixture['entries'], plan['variant'])
                runs[arm['arm']].append(receipt)
        determinism = {arm: _byte_determinism(runs[arm]) for arm in ('a', 'b')}
        status = ('byte-determinism-confirmed'
                  if all(determinism[arm]['byte_deterministic'] for arm in ('a', 'b'))
                  else 'unresolved')
        receipt = {
            'schema': 'atlas-c2-ab-receipt-v1', 'attempt': plan['attempt'],
            'variant': plan['variant'], 'repeats_per_arm': plan['repeats'],
            'evidence_root': str(plan['evidence_root']),
            'arms': runs,
            'byte_determinism': determinism,
            'pre_hash_alignment': _pre_hash_alignment(runs['a'], runs['b']),
            'b_repeat_pre_hash_determinism': _b_repeat_pre_hash_determinism(runs['b']),
            'causality': {'status': status,
                          'reason': ('both arms byte-deterministic across repeats'
                                     if status != 'unresolved' else
                                     'variation across repeats: causality stays unresolved')},
            'harness_sha256': {'c2_ab_driver.py': _sha(__file__)},
            'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        (attempt_root / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        return receipt
    except Exception as exc:
        (attempt_root / 'failure.json').write_text(json.dumps(
            {'schema': 'atlas-c2-ab-failure-v1', 'exception_type': type(exc).__name__,
             'message': str(exc),
             'utc_time': datetime.datetime.now(datetime.timezone.utc).isoformat()},
            indent=2) + '\n')
        raise


def main(collector=collect_logits.collect):
    _require(collector is not collect_logits.collect,
             'live execution requires c2_remote_adapter.py and an explicit remote config')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-root', type=Path, required=True)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--variant', choices=make_norm_fixtures.VARIANTS, required=True)
    parser.add_argument('--fixture-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--repeats', type=int, required=True)
    args = parser.parse_args()
    run_config = load_run_config(args.config)
    arm_configs = arm_configurations(run_config, args.fixture_dir, args.variant)
    provenance = json.loads(Path(run_config['provenance_file']).read_text())
    cases = json.loads(Path(run_config['cases_file']).read_text())
    thresholds = json.loads(Path(run_config['thresholds_file']).read_text())
    plan = {'evidence_root': args.evidence_root, 'attempt': args.attempt,
            'variant': args.variant, 'fixture_dir': args.fixture_dir,
            'repeats': args.repeats, 'arm_configs': arm_configs,
            'provenance': provenance, 'cases': cases, 'thresholds': thresholds}
    receipt = run_ab(plan, collector)
    print(json.dumps({'attempt': args.attempt,
                      'causality': receipt['causality']['status'],
                      'receipt': str(args.evidence_root / args.attempt / 'receipt.json')}))


if __name__ == '__main__':
    main()
