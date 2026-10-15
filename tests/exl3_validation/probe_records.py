# SPDX-License-Identifier: AGPL-3.0-only
"""Probe pre-Spark records: attempt attestation, profiler binding, NCCL hashes.

Implements the three remaining registration items owed before either Spark
is obtained (``docs/design/exl3-runtime-mechanism-probe-registration.md``):

  attest   parse the registration's frozen attempt list and emit the
           attestation whose SHA256 is copied into every attempt receipt
           (``stamp``); the code-side ATTEMPTS constant must equal the
           registered list, so either side drifting fails the tool.
  profiler bind/unavailable   record the profiler selection (nsys), its
           version and binary hash when present, or the recorded tooling
           limitation when not — always with the exact registered invocation
           flags and their hash, before any attempt runs.
  nccl prior   bind the prior loaded-NCCL evidence (both ranks) with its
           stated limitation; nccl live re-binds loaded library paths from a
           running process's /proc during pre-attempt setup.

Usage:
    probe_records.py attest --registration P --out F
    probe_records.py stamp  --registration P --receipt DIR
    probe_records.py profiler bind --nsys PATH --out F
    probe_records.py profiler unavailable --reason TEXT --out F
    probe_records.py nccl prior --config C --out F
    probe_records.py nccl live --maps M --proc-root R --rank N --out F
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

from image_cycle import SubprocessRunner, sha, read, require, text

ATTEST_SCHEMA = 'atlas-probe-attempt-attestation-v1'
NCCL_BINDING_SCHEMA = 'atlas-probe-nccl-binding-v1'
NCCL_PRIOR_SCHEMA = 'atlas-probe-nccl-prior-config-v1'
PROFILER_SCHEMA = 'atlas-probe-profiler-binding-v1'
# Registered fixed order (three repeats per arm; arm is the only difference).
ATTEMPTS = (('off01', 'off'), ('on01', 'on'), ('on02', 'on'),
            ('off02', 'off'), ('off03', 'off'), ('on03', 'on'))
NSYS_TOOL = 'nsys'
NSYS_PROFILE_FLAGS = ('profile', '--trace=cuda,nvtx', '--sample=none',
                      '--cpuctxsw=none', '-o', '{output}')


def write_json(path, value):
    with Path(path).open('x') as target:
        target.write(text(value))


def invocation():
    return [NSYS_TOOL, *NSYS_PROFILE_FLAGS]


def invocation_sha256():
    return hashlib.sha256(
        json.dumps(list(invocation()), separators=(',', ':')).encode()).hexdigest()


def registered_attempts(registration_path):
    """Parse the frozen attempt list out of the registration section."""
    body = Path(registration_path).read_text()
    section = re.search(r'^## Attempt list.*?(?=^## )', body, re.S | re.M)
    require(section is not None, 'registration lacks the attempt-list section')
    rows = re.findall(r'^\s{4}([a-z0-9]+(?:,\s*[a-z0-9]+)+)\s*$',
                      section.group(0), re.M)
    require(len(rows) == 1, 'registration must carry exactly one attempt row')
    ids = tuple(item.strip() for item in rows[0].split(','))
    require(ids == tuple(name for name, _ in ATTEMPTS),
            'registered attempt list differs from the code-side freeze: '
            + str(ids))
    return ids


def attestation(registration_path):
    ids = registered_attempts(registration_path)
    return {'schema': ATTEST_SCHEMA,
            'attempts': [{'attempt': name, 'arm': arm} for name, arm in ATTEMPTS],
            'registration': {'path': str(Path(registration_path).resolve()),
                             'sha256': sha(registration_path)}}


def stamp(registration_path, receipt_dir):
    receipt_dir = Path(receipt_dir)
    require(receipt_dir.is_dir(), 'attempt receipt directory required')
    target = receipt_dir / 'attestation.json'
    require(not target.exists(), 'attempt receipt already carries an attestation')
    value = attestation(registration_path)
    write_json(target, value)
    return value


def profiler_bind(nsys_path, out_path, runner=None):
    path = Path(nsys_path)
    require(path.is_file(), 'nsys binary missing: ' + str(path))
    runner = runner or SubprocessRunner()
    result = runner.run([str(path), '--version'], env=None, timeout=120)
    require(result['rc'] == 0, 'nsys --version failed')
    first_line = result['stdout'].decode(errors='replace').splitlines()
    require(first_line, 'nsys --version produced no output')
    value = {'schema': PROFILER_SCHEMA, 'tool': NSYS_TOOL, 'available': True,
             'version': first_line[0].strip(), 'binary': {'path': str(path),
             'sha256': sha(path)},
             'invocation': list(invocation()),
             'invocation_sha256': invocation_sha256(),
             'limitation': None}
    write_json(out_path, value)
    return value


def profiler_unavailable(reason, out_path):
    require(isinstance(reason, str) and reason.strip(),
            'an explicit unavailability reason is required')
    value = {'schema': PROFILER_SCHEMA, 'tool': NSYS_TOOL, 'available': False,
             'version': None, 'binary': None,
             'invocation': list(invocation()),
             'invocation_sha256': invocation_sha256(),
             'limitation': reason.strip()}
    write_json(out_path, value)
    return value


def _parse_sha_lines(path):
    libraries = []
    for line in Path(path).read_text().splitlines():
        match = re.fullmatch(r'([0-9a-f]{64})  (/\S+)', line)
        require(match is not None, 'unparseable loaded-library line: ' + line)
        libraries.append({'sha256': match.group(1), 'path': match.group(2)})
    require(libraries, 'loaded-library evidence file is empty: ' + str(path))
    return libraries


def nccl_prior(config_path, out_path):
    config = read(config_path)
    require(config.get('schema') == NCCL_PRIOR_SCHEMA
            and isinstance(config.get('entries'), list) and config['entries'],
            'explicit prior-NCCL entries required')
    entries, ranks = [], set()
    for position, spec in enumerate(config['entries']):
        require(set(spec) == {'path', 'sha256', 'rank', 'kind'},
                'prior entry keys differ: ' + str(position))
        require(spec['kind'] in ('loaded-sha256', 'runtime-evidence'),
                'unknown prior entry kind: ' + str(spec['kind']))
        path = Path(spec['path'])
        require(path.is_file() and sha(path) == spec['sha256'],
                'bound prior evidence differs: ' + spec['path'])
        if spec['kind'] == 'loaded-sha256':
            require(spec['rank'] in ('0', '1'), 'loaded evidence needs a rank')
            ranks.add(spec['rank'])
            libraries = _parse_sha_lines(path)
        else:
            require(spec['rank'] == 'reference',
                    'runtime evidence is keyed reference')
            evidence = json.loads(path.read_text())
            require(evidence.get('status') == 0
                    and re.fullmatch(r'[0-9a-f]{64}', evidence.get('sha256', ''))
                    is not None and evidence.get('library'),
                    'runtime NCCL evidence malformed: ' + spec['path'])
            libraries = [{'sha256': evidence['sha256'],
                          'path': evidence['library'],
                          'version_code': evidence.get('version_code')}]
        entries.append({'rank': spec['rank'], 'kind': spec['kind'],
                        'path': str(path), 'sha256': spec['sha256'],
                        'libraries': libraries})
    require(ranks == {'0', '1'}, 'prior binding needs loaded evidence for both ranks')
    value = {'schema': NCCL_BINDING_SCHEMA,
             'scope': 'prior loaded-run evidence bound by hash; live re-bind '
                      'from running reference processes is owed during '
                      'pre-attempt setup (nccl live)',
             'entries': entries}
    write_json(out_path, value)
    return value


def nccl_live(maps_path, proc_root, rank, out_path):
    require(rank in ('0', '1'), 'rank must be 0 or 1')
    root = Path(proc_root)
    require(root.is_dir(), 'proc root directory required')
    paths = set()
    for line in Path(maps_path).read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        library = parts[-1]
        if library == '(deleted)':
            library = parts[-2] if len(parts) >= 2 else ''
        if 'libnccl' in library and library.startswith('/'):
            paths.add(library)
    require(paths, 'no loaded libnccl mapping found in ' + str(maps_path))
    libraries = []
    for library in sorted(paths):
        candidate = root / library.lstrip('/')
        require(candidate.is_file(),
                'mapped NCCL library missing under proc root: ' + library)
        libraries.append({'path': library, 'sha256': sha(candidate)})
    value = {'schema': NCCL_BINDING_SCHEMA, 'scope': 'live loaded-maps binding',
             'entries': [{'rank': rank, 'kind': 'loaded-maps',
                          'maps': {'path': str(Path(maps_path).resolve()),
                                   'sha256': sha(maps_path)},
                          'libraries': libraries}]}
    write_json(out_path, value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    attest = sub.add_parser('attest')
    attest.add_argument('--registration', required=True, type=Path)
    attest.add_argument('--out', required=True, type=Path)
    stamp_cmd = sub.add_parser('stamp')
    stamp_cmd.add_argument('--registration', required=True, type=Path)
    stamp_cmd.add_argument('--receipt', required=True, type=Path)
    profiler = sub.add_parser('profiler').add_subparsers(dest='profiler_mode',
                                                         required=True)
    bind = profiler.add_parser('bind')
    bind.add_argument('--nsys', required=True)
    bind.add_argument('--out', required=True, type=Path)
    missing = profiler.add_parser('unavailable')
    missing.add_argument('--reason', required=True)
    missing.add_argument('--out', required=True, type=Path)
    nccl = sub.add_parser('nccl').add_subparsers(dest='nccl_mode', required=True)
    prior = nccl.add_parser('prior')
    prior.add_argument('--config', required=True, type=Path)
    prior.add_argument('--out', required=True, type=Path)
    live = nccl.add_parser('live')
    live.add_argument('--maps', required=True, type=Path)
    live.add_argument('--proc-root', required=True, type=Path)
    live.add_argument('--rank', required=True, choices=('0', '1'))
    live.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.mode == 'attest':
        result = attestation(args.registration)
        write_json(args.out, result)
    elif args.mode == 'stamp':
        result = stamp(args.registration, args.receipt)
    elif args.mode == 'profiler':
        result = (profiler_bind(args.nsys, args.out)
                  if args.profiler_mode == 'bind'
                  else profiler_unavailable(args.reason, args.out))
    else:
        result = (nccl_prior(args.config, args.out)
                  if args.nccl_mode == 'prior'
                  else nccl_live(args.maps, args.proc_root, args.rank, args.out))
    print(text({'mode': args.mode, 'records': len(result.get('attempts',
          result.get('entries', [1])))}), end='')


if __name__ == '__main__':
    main()
