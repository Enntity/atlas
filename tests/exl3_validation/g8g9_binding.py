# SPDX-License-Identifier: AGPL-3.0-only
"""Assemble the G8/G9 provenance binding receipt for EXL3 replay work.

G8 binds the effective checkpoint configuration, the per-rank runtime
overrides carried by the registered serve descriptors, and the selected
cuBLAS providers. G9 binds compiled binaries to source: the candidate build
manifest, hash inventories collected from live images, and the registered
reference source anchors (verified against the plan's source contract table,
the SSOT for those hashes).

Two subcommands:

  collect  hash the explicitly listed binaries inside an image (one ssh +
           ``docker run``; coreutils only) and write an inventory.
  bind     verify every explicit hash binding and emit the receipt. Nothing
           is inferred: a missing key, a manifest whose recorded
           config_sha256 differs, or an anchor that does not match the plan
           table fails the bind.

Usage:
    g8g9_binding.py collect --config collect.json --out inventory.json
    g8g9_binding.py bind --config bind.json --out binding.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

from image_cycle import SubprocessRunner, ssh_argv, sha, read, require, text

COLLECT_SCHEMA = 'atlas-exl3-g8g9-collect-v1'
INVENTORY_SCHEMA = 'atlas-exl3-g8g9-inventory-v1'
BIND_SCHEMA = 'atlas-exl3-g8g9-bind-v1'
BINDING_SCHEMA = 'atlas-exl3-g8g9-binding-v1'
ENTRY_COMMANDS = {'serve', '/start.sh'}
CONFIG_SHA_KEY = 'config.json'


def write_json(path, value):
    with Path(path).open('x') as target:
        target.write(text(value))


def _bound(spec, label):
    require(set(spec) == {'path', 'sha256'}, 'explicit path/sha256 required: ' + label)
    path = Path(spec['path'])
    require(path.is_absolute() and path.is_file(), 'missing bound input: ' + label)
    require(sha(path) == spec['sha256'], 'bound input hash differs: ' + label)
    return str(path)


def collect(config_path, out_path, runner=None):
    config = read(config_path)
    require(set(config) == {'schema', 'host', 'jump', 'image', 'entries'}
            and config['schema'] == COLLECT_SCHEMA,
            'explicit collect configuration keys/schema required')
    require(re.fullmatch(r'[^\s\'"]+', config['host']) is not None
            and re.fullmatch(r'[^\s\'"]+', config['jump']) is not None
            and re.fullmatch(r'[^\s\'"]+', config['image']) is not None,
            'host, jump and image must be shell-safe tokens')
    entries = config['entries']
    require(isinstance(entries, list) and entries, 'explicit binary entries required')
    labels, paths = set(), []
    for entry in entries:
        require(set(entry) == {'label', 'path'}, 'entry must be {label, path}')
        require(re.fullmatch(r'[a-z0-9][a-z0-9-]{0,55}', entry['label']) is not None,
                'filename-safe entry label required')
        require(entry['label'] not in labels, 'duplicate entry label: ' + entry['label'])
        require(re.fullmatch(r'/[A-Za-z0-9._/+:-]+', entry['path']) is not None,
                'entry path must be an absolute shell-safe path: ' + entry['path'])
        labels.add(entry['label'])
        paths.append(entry['path'])
    require(len(set(paths)) == len(paths), 'duplicate entry paths not allowed')
    script = 'for f in ' + ' '.join(paths) + '; do sha256sum "$f"; done'
    argv = ssh_argv(config['jump'], config['host'],
                    f"docker run --rm --entrypoint /bin/sh {config['image']} "
                    f"-c '{script}'")
    runner = runner or SubprocessRunner()
    result = runner.run(argv, env=None, timeout=600)
    require(result['rc'] == 0, 'in-image sha256sum failed: '
            + result['stderr'].decode(errors='replace')[-400:])
    found = {}
    for line in result['stdout'].decode().splitlines():
        match = re.fullmatch(r'([0-9a-f]{64})  (\S+)', line)
        require(match is not None, 'unparseable sha256sum line: ' + line)
        require(match.group(2) not in found, 'duplicate inventory path: ' + match.group(2))
        found[match.group(2)] = match.group(1)
    missing = sorted(set(paths) - set(found))
    require(not missing, 'binaries missing inside image: ' + ', '.join(missing))
    extra = sorted(set(found) - set(paths))
    require(not extra, 'unexpected binaries in image output: ' + ', '.join(extra))
    inventory = {'schema': INVENTORY_SCHEMA, 'image': config['image'],
                 'host': config['host'],
                 'entries': [{'label': entry['label'], 'path': entry['path'],
                              'sha256': found[entry['path']]}
                             for entry in entries],
                 'command_sha256': hashlib.sha256(argv[-1].encode()).hexdigest()}
    write_json(out_path, inventory)
    return inventory


def parse_descriptor(path, engine):
    """Per-rank effective environment and serve argv from a create descriptor."""
    require(engine in ('candidate', 'reference'), 'unknown descriptor engine')
    raw = read(path)
    require(isinstance(raw, list) and all(isinstance(item, str) for item in raw),
            'serve descriptor must be a string list: ' + str(path))
    require(raw[:2] == ['docker', 'create'], 'docker create descriptor required: ' + str(path))
    entries = [index for index, item in enumerate(raw)
               if item in ENTRY_COMMANDS and not item.startswith('-')]
    require(len(entries) == 1, 'exactly one serve/start entry required: ' + str(path))
    index = entries[0]
    image = raw[index - 1]
    require(not image.startswith('-'), 'descriptor image identity missing: ' + str(path))
    env = {}
    for position, item in enumerate(raw[:index]):
        if item in ('--env', '-e'):
            pair = raw[position + 1]
            require('=' in pair, 'descriptor env must be KEY=VALUE: ' + str(path))
            key, value = pair.split('=', 1)
            require(key not in env or env[key] == value,
                    'descriptor env keys must be unique: ' + str(path))
            env[key] = value
    return {'file': str(path), 'sha256': sha(path), 'image': image,
            'engine': engine, 'command': raw[index:], 'env': env}


def _plan_anchor_shas(plan_doc):
    """Parse the registered source-contract table: `label` | `sha256` rows."""
    table = {}
    pattern = re.compile(r'^\| `([^`]+)` \| `([0-9a-f]{64})` \|$', re.M)
    for label, digest in pattern.findall(Path(plan_doc).read_text()):
        table[label] = digest
    require(table, 'plan source-contract table not found: ' + str(plan_doc))
    return table


def bind(config_path, out_path):
    config = read(config_path)
    require(set(config) == {'schema', 'checkpoint_config', 'g8_fields',
                            'reference_manifests', 'descriptors', 'pins_file',
                            'provider_reports', 'inventories', 'build_manifest',
                            'source_anchors', 'plan_doc'}
            and config['schema'] == BIND_SCHEMA,
            'explicit bind configuration keys/schema required')
    inputs = {}

    checkpoint = _bound(config['checkpoint_config'], 'checkpoint_config')
    inputs['checkpoint_config'] = {'path': checkpoint,
                                   'sha256': config['checkpoint_config']['sha256']}
    checkpoint_json = read(checkpoint)
    fields = config['g8_fields']
    require(isinstance(fields, list) and fields
            and all(isinstance(key, str) and key for key in fields),
            'explicit g8_fields required')
    require(len(set(fields)) == len(fields), 'duplicate g8_fields not allowed')
    missing = [key for key in fields if key not in checkpoint_json]
    require(not missing, 'g8 fields missing from checkpoint config: ' + ', '.join(missing))
    config_digest = sha(checkpoint)

    manifests = []
    seen = set()
    require(isinstance(config['reference_manifests'], list)
            and config['reference_manifests'],
            'at least one reference manifest is required for the G8 '
            'config_sha256 cross-check')
    for position, spec in enumerate(config['reference_manifests']):
        path = _bound(spec, f'reference_manifests[{position}]')
        inputs[f'reference_manifest_{position}'] = {'path': path, 'sha256': spec['sha256']}
        manifest = read(path)
        recorded = manifest.get('provenance', {}).get('config_sha256')
        require(isinstance(recorded, dict) and recorded.get(CONFIG_SHA_KEY) == config_digest,
                'manifest config_sha256 differs from checkpoint config: ' + path)
        manifests.append(path)
        seen.add(path)
    require(len(seen) == len(config['reference_manifests']),
            'duplicate reference manifests not allowed')

    require(set(config['descriptors']) == {'candidate', 'reference'},
            'G8 binds both engines\' effective descriptors')
    effective = {}
    for engine, ranks in config['descriptors'].items():
        require(engine in ('candidate', 'reference') and set(ranks) == {'0', '1'},
                'descriptors must give both ranks per engine: ' + engine)
        effective[engine] = {}
        for rank, spec in ranks.items():
            path = _bound(spec, f'descriptors.{engine}.{rank}')
            parsed = parse_descriptor(path, engine)
            require(parsed['sha256'] == spec['sha256'], 'descriptor hash differs: ' + path)
            inputs[f'descriptor_{engine}_{rank}'] = {'path': path,
                                                     'sha256': spec['sha256']}
            effective[engine][rank] = parsed

    pins_path = _bound(config['pins_file'], 'pins_file')
    inputs['pins_file'] = {'path': pins_path, 'sha256': config['pins_file']['sha256']}
    pins = {}
    for line in Path(pins_path).read_text().splitlines():
        match = re.fullmatch(r'([0-9a-f]{64})  (/.+)', line)
        require(match is not None, 'unparseable pins line: ' + line)
        pins[Path(match.group(2)).name] = {'path': match.group(2),
                                           'sha256': match.group(1)}
    require({'libcublas.so.13', 'libcublasLt.so.13'} <= set(pins),
            'pins file must name both pinned providers')

    reports = []
    require(isinstance(config['provider_reports'], list)
            and config['provider_reports'],
            'at least one passing provider report is required for G8')
    for position, spec in enumerate(config['provider_reports']):
        path = _bound(spec, f'provider_reports[{position}]')
        inputs[f'provider_report_{position}'] = {'path': path, 'sha256': spec['sha256']}
        report = read(path)
        require(report.get('ok') is True and report.get('issues') == [],
                'provider report is not a passing report: ' + path)
        libraries = {item['name']: item for item in report.get('libraries', [])}
        require({'libcublas.so.13', 'libcublasLt.so.13'} <= set(libraries),
                'provider report must cover both providers: ' + path)
        for name, item in libraries.items():
            require(name in pins, 'provider report library not pinned: ' + name)
            require(item.get('ok') is True and item.get('sha256') == pins[name]['sha256'],
                    'provider report library differs from pins: ' + name)
        reports.append({'path': path, 'sha256': spec['sha256']})

    inventories = []
    for position, spec in enumerate(config['inventories']):
        path = _bound(spec, f'inventories[{position}]')
        inputs[f'inventory_{position}'] = {'path': path, 'sha256': spec['sha256']}
        inventory = read(path)
        require(inventory.get('schema') == INVENTORY_SCHEMA,
                'inventory schema differs: ' + path)
        for entry in inventory.get('entries', []):
            require(set(entry) == {'label', 'path', 'sha256'}
                    and re.fullmatch(r'[0-9a-f]{64}', entry['sha256']) is not None,
                    'inventory entry malformed: ' + path)
        inventories.append({'path': path, 'sha256': spec['sha256'],
                            'image': inventory.get('image'),
                            'entries': inventory['entries']})
    require(inventories, 'at least one binary inventory required for G9')

    candidate = None
    if config['build_manifest'] is not None:
        path = _bound(config['build_manifest'], 'build_manifest')
        inputs['build_manifest'] = {'path': path,
                                    'sha256': config['build_manifest']['sha256']}
        manifest = read(path)
        require(manifest.get('schema') == 'atlas-exl3-build-manifest-v1',
                'build manifest schema differs')
        require(re.fullmatch(r'[0-9a-f]{40}', manifest.get('git_commit', '')) is not None,
                'build manifest git commit missing')
        binaries = manifest.get('binaries', {})
        require('spark' in binaries and re.fullmatch(r'[0-9a-f]{64}',
                binaries['spark'].get('sha256', '')) is not None,
                'build manifest lacks the spark binary hash')
        candidate = {'path': path, 'sha256': config['build_manifest']['sha256'],
                     'git_commit': manifest['git_commit'],
                     'source_snapshot_sha256': manifest['source_snapshot_sha256'],
                     'spark_binary_sha256': binaries['spark']['sha256'],
                     'ptx_modules': len(manifest.get('ptx_modules', []))}

    anchors = {}
    if config['source_anchors']:
        plan_doc = _bound(config['plan_doc'], 'plan_doc')
        inputs['plan_doc'] = {'path': plan_doc, 'sha256': config['plan_doc']['sha256']}
        table = _plan_anchor_shas(plan_doc)
        source_anchors = config['source_anchors']
        require(isinstance(source_anchors, dict) and source_anchors,
                'source_anchors must be a non-empty label-to-path map')
        for label, path_value in sorted(source_anchors.items()):
            require(label in table, 'anchor not in plan source contract: ' + label)
            path = Path(path_value)
            require(path.is_absolute() and path.is_file(),
                    'missing anchor source: ' + str(path))
            require(sha(path) == table[label],
                    'anchor differs from plan source contract: ' + label)
            anchors[label] = {'path': str(path), 'sha256': table[label]}

    binding = {'schema': BINDING_SCHEMA,
               'g8': {'checkpoint': {'path': checkpoint, 'sha256': config_digest,
                                     'fields': {key: checkpoint_json[key]
                                                for key in fields}},
                      'manifest_crosscheck': {'manifests': len(manifests),
                                              'config_sha256': config_digest,
                                              'all_equal': True},
                      'effective': effective,
                      'providers': {'pins': pins, 'reports': reports}},
               'g9': {'candidate_build': candidate,
                      'inventories': inventories,
                      'source_anchors': anchors},
               'inputs': inputs}
    write_json(out_path, binding)
    return binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    for mode in ('collect', 'bind'):
        command = sub.add_parser(mode)
        command.add_argument('--config', required=True, type=Path)
        command.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.mode == 'collect':
        result = collect(args.config, args.out, SubprocessRunner())
        summary = {'mode': 'collect', 'out': str(args.out),
                   'entries': len(result['entries'])}
    else:
        result = bind(args.config, args.out)
        summary = {'mode': 'bind', 'out': str(args.out),
                   'inputs': len(result['inputs'])}
    print(text(summary), end='')


if __name__ == '__main__':
    main()
