# SPDX-License-Identifier: AGPL-3.0-only
"""Assemble atlas-norm-substitution-fixtures-v1 fixture directories (C2 deliverable D2).

Input: reference-side layer-0 gated_norm captures (upstream reference package
dump layout, or the bound q2-norm-replay fixtures layout), a run-schedule JSON
(request order -> request_id) and --variant. Output: one immutable fixture
directory (manifest.json + payload .bin files + provenance.json) consumed by the
Rust substitution knob (docs/plans/c2-tooling-spec.md "Fixture manifest").

Fail-closed throughout, in the validation discipline of
gated_norm_ablation.validate_fixture: explicit identities, sha256 records for
every file read or emitted, dtype/shape/finiteness checks, and refusal to emit
on any request_id<->invocation mismatch. The variant sets stay separate even
when byte-identical (docs/plans/c2-tooling-spec.md A6); byte identity is reported as a fact in
provenance.json, never collapsed. Fixture layout is new contract
(docs/plans/c2-tooling-spec.md A10); manifest.json stays exactly the spec schema so the
parallel Rust consumer sees no drift.
"""
import argparse
import datetime
import hashlib
import json
import re
from pathlib import Path

import numpy as np

import gated_norm_ablation

MANIFEST_SCHEMA = 'atlas-norm-substitution-fixtures-v1'
PROVENANCE_SCHEMA = 'atlas-norm-substitution-provenance-v1'
SCHEDULE_SCHEMA = 'atlas-norm-run-schedule-v1'
VARIANTS = ('reference-exact', 'v24-corrected')
STAGE = 'gated_norm'
TENSOR = 'gated'
DTYPE = 'bf16-le'
SOURCE = 'installed-reference-forward-hook'
UPSTREAM_STAGE = 'kda-gated-norm'
MANIFEST_KEYS = {'schema', 'variant', 'source', 'entries'}
SOURCE_KEYS = {'capture_root', 'engine', 'schedule_sha256'}
ENTRY_KEYS = {'request_id', 'layer', 'tp_rank', 'tp_world', 'invocation', 'stage',
              'tensor', 'dtype', 'shape', 'file', 'sha256'}
_UPSTREAM_SIDECAR = re.compile(r'ffn-L(\d+)-rank(\d+)-(\d{6})-' + UPSTREAM_STAGE + r'\.json')
_HASH = re.compile(r'[0-9a-f]{64}')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_json(path, hashes):
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), f'Regular local artifact required: {path}')
    data = path.read_bytes()
    hashes[str(path)] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def _hex_sha(value, label):
    _require(isinstance(value, str) and _HASH.fullmatch(value) is not None,
             f'{label} must be a lowercase SHA256 digest')
    return value


def _payload(raw, shape, label):
    """Contiguous little-endian BF16 bits matching the recorded geometry."""
    _require(isinstance(shape, list) and len(shape) >= 1
             and all(type(dim) is int and dim > 0 for dim in shape),
             f'Invalid tensor geometry: {label}')
    _require(len(raw) == int(np.prod(shape)) * 2, f'Capture byte count does not match shape: {label}')
    value = np.frombuffer(raw, dtype='<u2')
    _require(not bool(np.any((value & 0x7f80) == 0x7f80)),
             f'Finite BF16 capture required: {label}')
    return raw


def load_schedule(path):
    """Validate the run schedule; returns request_ids in request order."""
    hashes = {}
    schedule = _read_json(path, hashes)
    _require(isinstance(schedule, dict) and set(schedule) == {'schema', 'requests'}
             and schedule['schema'] == SCHEDULE_SCHEMA,
             'run schedule must be an ' + SCHEDULE_SCHEMA + ' object')
    requests = schedule['requests']
    _require(isinstance(requests, list) and requests, 'run schedule requests must be a nonempty list')
    ids = []
    for position, entry in enumerate(requests):
        _require(isinstance(entry, dict) and set(entry) == {'order', 'request_id'},
                 'schedule entry requires exactly order/request_id')
        _require(type(entry['order']) is int and entry['order'] == position,
                 'schedule orders must run sequentially from zero')
        _require(isinstance(entry['request_id'], str) and entry['request_id'],
                 'schedule request_id must be a nonempty string')
        _require(entry['request_id'] not in ids, 'schedule request_id values must be unique')
        ids.append(entry['request_id'])
    return ids, hashes


def read_upstream_captures(capture_dirs, tp_world, hashes):
    """Upstream reference package dump layout: flat sidecars + raw BF16 bins.

    Sidecar identity is checked against its filename (layer/rank/invocation) and
    the request binding is taken from chunk_position; duplicate declared
    identities refuse even under distinct filenames.
    """
    records = {}
    for rank in range(tp_world):
        capture_dir = Path(capture_dirs[rank])
        _require(capture_dir.is_dir() and not capture_dir.is_symlink(),
                 f'capture directory required for rank {rank}')
        seen = set()
        for path in sorted(capture_dir.iterdir()):
            match = _UPSTREAM_SIDECAR.fullmatch(path.name)
            if not match:
                continue
            file_layer, file_rank, file_invocation = (int(match.group(1)), int(match.group(2)),
                                                      int(match.group(3)))
            side = _read_json(path, hashes)
            _require(isinstance(side, dict), f'Object sidecar required: {path}')
            declared = (side.get('layer'), side.get('rank'), side.get('invocation'))
            _require(all(type(value) is int for value in declared) and declared[0] == 0
                     and declared[1] == rank,
                     f'Tensor sidecar identity binding differs: {path}')
            key = (rank, declared[2])
            _require(key not in seen, f'duplicate (layer, rank, invocation, tensor) capture: {path}')
            seen.add(key)
            _require(declared == (file_layer, file_rank, file_invocation),
                     f'Tensor sidecar invocation/row identity mismatch: {path}')
            _require(side.get('stage') == UPSTREAM_STAGE and side.get('source') == SOURCE
                     and side.get('dtype') == DTYPE,
                     f'Tensor sidecar stage/source/dtype binding differs: {path}')
            chunk = side.get('chunk_position')
            _require(isinstance(chunk, dict), f'chunk_position binding required: {path}')
            request_id = chunk.get('request_id')
            _require(isinstance(request_id, str) and request_id,
                     f'missing request id in chunk_position: {path}')
            tokens = chunk.get('scheduled_tokens')
            _require(type(tokens) is int and tokens > 0,
                     f'scheduled_tokens identity required: {path}')
            filename = side.get('file')
            _require(isinstance(filename, str) and filename not in ('', '.', '..')
                     and Path(filename).name == filename,
                     f'tensor filename must be local basename: {path}')
            raw = _read_bytes(path.parent / filename, hashes)
            _payload(raw, side.get('shape'), str(path))
            records[key] = {'rank': rank, 'invocation': declared[2], 'tokens': tokens,
                            'request_id': request_id, 'raw': raw, 'sidecar': str(path),
                            'order': None}
        _require(seen, f'no layer-0 {UPSTREAM_STAGE} captures found in {capture_dir}')
    return records


def _read_bytes(path, hashes):
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), f'Regular local artifact required: {path}')
    data = path.read_bytes()
    hashes[str(path)] = hashlib.sha256(data).hexdigest()
    return data


def read_replay_fixture_captures(fixtures_root, tp_world, hashes):
    """Bound q2-norm-replay fixtures layout (rankN-rowM dirs).

    Row identity, file hashes, dtype/shape and operand equality are the
    reused validate_fixture seam; only the reference-side gated output is taken.
    """
    root = Path(fixtures_root)
    _require(root.is_dir() and not root.is_symlink(), 'replay fixtures root required')
    records = {}
    for rank in range(tp_world):
        rows = gated_norm_ablation.row_dirs(root, rank)
        _require(rows, f'fixture rows with fixture.json required for rank {rank}')
        for row, row_dir in rows:
            fixture, arrays = gated_norm_ablation.validate_fixture(row_dir, row, rank)
            hashes[str(row_dir / 'fixture.json')] = _sha(row_dir / 'fixture.json')
            request = fixture.get('request')
            _require(isinstance(request, dict) and type(request.get('index')) is int
                     and request['index'] >= 0,
                     f'fixture request order binding required: {row_dir}')
            key = (rank, row)
            _require(key not in records,
                     f'duplicate (layer, rank, invocation, tensor) capture: {row_dir}')
            value = arrays['reference-expected']
            _require(value.dtype.str == '<u2', f'BF16 fixture payload required: {row_dir}')
            hashes[str(row_dir / 'reference-expected.npy')] = _sha(row_dir / 'reference-expected.npy')
            records[key] = {'rank': rank, 'invocation': row, 'tokens': int(fixture['tokens']),
                            'request_id': None, 'raw': value.tobytes(), 'sidecar': str(row_dir),
                            'order': request['index']}
    return records


def bind_request_ids(records, schedule_ids):
    """request_id<->invocation binding from the schedule plus serial execution.

    Captures carrying their own request_id (upstream dump) must agree with the
    schedule in first-appearance order and completeness; bound fixtures map
    request order to invocation directly. Any mismatch refuses to emit.
    """
    _require(records, 'no captures to bind')
    invocations = sorted({key[1] for key in records})
    _require(invocations == list(range(len(invocations))),
             'invocations must run contiguously from zero')
    for rank, invocation in sorted(records):
        entry = records[(rank, invocation)]
        if entry['order'] is not None:
            _require(entry['order'] == invocation,
                     f'request order/invocation mismatch at rank {rank} invocation {invocation}')
            entry['request_id'] = schedule_ids[entry['order']]
        else:
            _require(entry['request_id'] in schedule_ids,
                     f'request_id not present in run schedule at invocation {invocation}')
    ordered = []
    for invocation in invocations:
        request_id = records[(0, invocation)]['request_id']
        for rank, inv in sorted(records):
            _require(records[(rank, inv)]['request_id'] == records[(0, inv)]['request_id'],
                     f'ranks disagree on request_id at invocation {inv}')
        if not ordered or ordered[-1] != request_id:
            _require(request_id not in ordered,
                     f'serial execution requires contiguous invocations per request: {request_id}')
            ordered.append(request_id)
    _require(ordered == schedule_ids,
             'schedule/invocation mismatch: captures do not cover the schedule in order')
    return ordered


def _flatten(raw, tokens, gated_width, label):
    _require(type(gated_width) is int and gated_width > 0, 'explicit positive --gated-width required')
    total = len(raw) // 2
    _require(total == tokens * gated_width,
             f'capture element count differs from tokens x gated width: {label}')
    return raw


def _v24_payload(replay_root, rank, invocation, tokens, gated_width, hashes):
    """v24-corrected values from the ablation replay outputs (provenance A6)."""
    row_dir = Path(replay_root) / f'v24-rank{rank}' / f'rank{rank}-row{invocation}'
    path = row_dir / 'v24-minimal.npy'
    _require(path.is_file(), f'v24 replay output required: {path}')
    hashes[str(path)] = _sha(path)
    receipt = row_dir / 'receipt.json'
    record = _read_json(receipt, hashes)
    _require(isinstance(record, dict) and record.get('schema') == 'atlas-gated-norm-ablation-v1'
             and record.get('rank') == rank and record.get('row') == invocation,
             f'v24 replay receipt identity differs: {receipt}')
    value = np.load(path, allow_pickle=False)
    _require(value.dtype.str == '<u2' and value.size == tokens * gated_width,
             f'v24 replay output geometry differs: {path}')
    _require(not bool(np.any((value & 0x7f80) == 0x7f80)),
             f'finite BF16 v24 replay output required: {path}')
    return value.tobytes()


def payload_name(invocation, rank):
    return f'L0-rank{rank}-{invocation:06d}-{TENSOR}.bin'


def validate_manifest(fixture_dir):
    """Fail-closed validation of an emitted atlas-norm-substitution-fixtures-v1
    directory: schema, entry identities, payload hashes/sizes and finiteness.
    Mirrors gated_norm_ablation.validate_fixture for the new manifest schema."""
    root = Path(fixture_dir)
    _require(root.is_dir() and not root.is_symlink(), f'fixture directory required: {root}')
    manifest = json.loads((root / 'manifest.json').read_text())
    _require(isinstance(manifest, dict) and set(manifest) == MANIFEST_KEYS
             and manifest['schema'] == MANIFEST_SCHEMA, 'fixture manifest schema mismatch')
    _require(manifest['variant'] in VARIANTS, 'fixture manifest variant mismatch')
    source = manifest['source']
    _require(isinstance(source, dict) and set(source) == SOURCE_KEYS
             and isinstance(source['capture_root'], str) and source['capture_root']
             and isinstance(source['engine'], str) and source['engine'],
             'fixture manifest source identity mismatch')
    _hex_sha(source['schedule_sha256'], 'schedule_sha256')
    entries = manifest['entries']
    _require(isinstance(entries, list) and entries, 'fixture manifest entries required')
    seen, files, ranks = set(), set(), {}
    for entry in entries:
        _require(isinstance(entry, dict) and set(entry) == ENTRY_KEYS,
                 'fixture entry key set mismatch')
        _require(type(entry['layer']) is int and entry['layer'] == 0, 'fixture entry layer must be 0')
        _require(type(entry['tp_rank']) is int and entry['tp_rank'] >= 0
                 and type(entry['tp_world']) is int and entry['tp_world'] > entry['tp_rank'],
                 'fixture entry TP identity mismatch')
        _require(type(entry['invocation']) is int and entry['invocation'] >= 0,
                 'fixture entry invocation identity mismatch')
        _require(entry['stage'] == STAGE and entry['tensor'] == TENSOR
                 and entry['dtype'] == DTYPE, 'fixture entry stage/tensor/dtype mismatch')
        _require(isinstance(entry['request_id'], str) and entry['request_id'],
                 'fixture entry request_id missing')
        shape = entry['shape']
        _require(isinstance(shape, list) and len(shape) == 2
                 and all(type(dim) is int and dim > 0 for dim in shape),
                 'fixture entry shape must be [tokens, p]')
        key = (entry['layer'], entry['tp_rank'], entry['invocation'], entry['tensor'])
        _require(key not in seen, 'duplicate (layer, tp_rank, invocation, tensor) entry')
        seen.add(key)
        _require(entry['file'] == payload_name(entry['invocation'], entry['tp_rank']),
                 'fixture entry file identity mismatch')
        _require(entry['file'] not in files, 'duplicate fixture payload file')
        files.add(entry['file'])
        ranks.setdefault(entry['tp_world'], set()).add(entry['tp_rank'])
        path = root / entry['file']
        _require(path.is_file() and not path.is_symlink(),
                 f'fixture payload required: {path}')
        _require(_sha(path) == _hex_sha(entry['sha256'], 'entry sha256'),
                 f'fixture payload hash mismatch: {path}')
        _payload(path.read_bytes(), shape, str(path))
    _require(len(ranks) == 1, 'fixture entries must share one tp_world')
    tp_world, covered = ranks.popitem()
    _require(covered == set(range(tp_world)), 'fixture entries must cover every rank')
    for rank in range(tp_world):
        _require({key[2] for key in seen if key[1] == rank}
                 == {key[2] for key in seen if key[1] == 0},
                 'fixture entries must cover identical invocations on every rank')
    return manifest


def assemble(capture_layout, capture_dirs, schedule_path, variant, out, engine,
             tp_world, gated_width, v24_replay_root=None):
    """Validate all inputs before creating the immutable output directory."""
    _require(variant in VARIANTS, 'variant must be one of ' + str(VARIANTS))
    _require(type(tp_world) is int and tp_world > 0, 'explicit positive tp_world required')
    _require(out and not Path(out).exists(), 'output directory must not exist')
    _require(isinstance(engine, str) and engine, 'explicit engine identity required')
    _require((v24_replay_root is None) == (variant != 'v24-corrected'),
             '--v24-replay-root is required exactly for variant v24-corrected')
    hashes = {}
    schedule_ids, schedule_hashes = load_schedule(schedule_path)
    hashes.update(schedule_hashes)
    if capture_layout == 'upstream-dump':
        records = read_upstream_captures(capture_dirs, tp_world, hashes)
    else:
        _require(capture_layout == 'replay-fixtures', 'unknown capture layout: ' + str(capture_layout))
        roots = {str(path) for path in capture_dirs.values()}
        _require(len(roots) == 1,
                 'replay-fixtures layout requires one shared fixtures root for every rank')
        records = read_replay_fixture_captures(next(iter(roots)), tp_world, hashes)
    bind_request_ids(records, schedule_ids)
    reference_payloads = {}
    for key in sorted(records):
        record = records[key]
        raw = _flatten(record['raw'], record['tokens'], gated_width, record['sidecar'])
        reference_payloads[key] = raw
    identity_rows, entries, payloads = [], [], {}
    for rank in range(tp_world):
        for invocation in sorted(key[1] for key in records if key[0] == rank):
            key = (rank, invocation)
            record = records[key]
            if variant == 'reference-exact':
                raw = reference_payloads[key]
            else:
                raw = _flatten(_v24_payload(v24_replay_root, rank, invocation,
                                            record['tokens'], gated_width, hashes),
                               record['tokens'], gated_width, f'v24 rank {rank} invocation {invocation}')
            identical = raw == reference_payloads[key]
            identity_rows.append({'tp_rank': rank, 'invocation': invocation,
                                  'byte_identical_to_reference_exact': identical})
            name = payload_name(invocation, rank)
            payloads[name] = raw
            entries.append({'request_id': record['request_id'], 'layer': 0, 'tp_rank': rank,
                            'tp_world': tp_world, 'invocation': invocation, 'stage': STAGE,
                            'tensor': TENSOR, 'dtype': DTYPE,
                            'shape': [record['tokens'], gated_width], 'file': name,
                            'sha256': hashlib.sha256(raw).hexdigest()})
    capture_root = (str(Path(capture_dirs[0])) if len(set(map(str, capture_dirs.values()))) == 1
                    else ','.join(f'rank{rank}={capture_dirs[rank]}' for rank in sorted(capture_dirs)))
    manifest = {'schema': MANIFEST_SCHEMA, 'variant': variant,
                'source': {'capture_root': capture_root, 'engine': engine,
                           'schedule_sha256': _sha(schedule_path)},
                'entries': entries}
    out = Path(out)
    out.mkdir(parents=False, exist_ok=False)  # immutable: fresh names only, never reuse
    for name, raw in payloads.items():
        (out / name).write_bytes(raw)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    provenance = {'schema': PROVENANCE_SCHEMA, 'variant': variant, 'engine': engine,
                  'capture_layout': capture_layout, 'capture_root': capture_root,
                  'schedule_file': str(schedule_path), 'schedule_sha256': _sha(schedule_path),
                  'schedule_request_ids': schedule_ids, 'tp_world': tp_world,
                  'gated_width': gated_width, 'v24_replay_root':
                      None if v24_replay_root is None else str(v24_replay_root),
                  'source_files': hashes,
                  'byte_identity_with_reference_exact':
                      None if variant == 'reference-exact' else identity_rows,
                  'harness_sha256': {'make_norm_fixtures.py': _sha(__file__),
                                     'gated_norm_ablation.py': _sha(Path(gated_norm_ablation.__file__))},
                  'assembled_utc': datetime.datetime.now(datetime.timezone.utc).strftime(
                      '%Y-%m-%dT%H:%M:%SZ')}
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    validate_manifest(out)  # re-validate what was emitted before reporting success
    return {'manifest': manifest, 'provenance': provenance, 'entries': len(entries)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-layout', choices=('upstream-dump', 'replay-fixtures'),
                        required=True)
    parser.add_argument('--capture-dir', action='append', required=True, metavar='RANK=PATH',
                        help='capture directory per rank (repeat for every rank)')
    parser.add_argument('--schedule', type=Path, required=True)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--engine', required=True)
    parser.add_argument('--tp-world', type=int, required=True)
    parser.add_argument('--gated-width', type=int, required=True)
    parser.add_argument('--v24-replay-root', type=Path)
    args = parser.parse_args()
    capture_dirs = {}
    for item in args.capture_dir:
        _require('=' in item, '--capture-dir requires RANK=PATH')
        rank_text, path = item.split('=', 1)
        _require(rank_text.isdigit() and int(rank_text) not in capture_dirs
                 and 0 <= int(rank_text) < args.tp_world,
                 '--capture-dir ranks must be unique and inside tp_world')
        capture_dirs[int(rank_text)] = Path(path)
    _require(set(capture_dirs) == set(range(args.tp_world)),
             '--capture-dir must cover every rank exactly once')
    summary = assemble(args.capture_layout, capture_dirs, args.schedule, args.variant,
                       args.out, args.engine, args.tp_world, args.gated_width,
                       args.v24_replay_root)
    print(json.dumps({'variant': args.variant, 'entries': summary['entries'],
                      'out': str(args.out)}))


if __name__ == '__main__':
    main()
