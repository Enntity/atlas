# SPDX-License-Identifier: AGPL-3.0-only
"""Cross-attempt byte reproducibility of L1 reference captures (O2, fused-off L1 observer study).

Compares rank{0,1}/diagnostic of two reference-observed windows: identical file sets,
byte-identical payloads, and identical sidecars apart from the registered per-run fields
(the opaque GPU request-ID suffix and the active-marker paths). Hook-source records are named
by the recording process ID, so they are compared as a multiset of contents instead.
Every compared file is first verified against its attempt's collection receipt.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re

from c2_remote_adapter import require

REQUEST_ID = re.compile(r'(cmpl-.+-0)-[0-9a-f]{8}')
MARKER = 'capture-enabled'
FILE_NAME = re.compile(r'ffn-L(\d+)-rank(\d)-(\d{6})-(.+)\.(bin|json)')
HOOK_SOURCE = re.compile(r'hook-source-(.+)-(\d+)\.json')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True)  # type-strict: true != 1, 1 != 1.0


def regular_files(directory):
    directory = Path(directory)
    require(directory.is_dir() and not directory.is_symlink(), 'capture directory required: '+str(directory))
    files = {}
    for path in directory.iterdir():
        require(path.is_file() and not path.is_symlink(), 'regular capture file required: '+str(path))
        files[path.name] = path
    return files


def verify_receipt(window, receipt):
    """Every rank diagnostic file is listed, hash-equal, in the attempt receipt, and vice versa."""
    window = Path(window)
    listed = json.loads(Path(receipt).read_text())['file_sha256']
    prefixes = tuple(window.name+'/rank'+rank+'/diagnostic/' for rank in ('0', '1'))
    expected = {key: digest for key, digest in listed.items() if key.startswith(prefixes)}
    actual = {window.name+'/rank'+rank+'/diagnostic/'+name: sha(path) for rank in ('0', '1')
              for name, path in regular_files(window/('rank'+rank)/'diagnostic').items()}
    require(expected and expected == actual, 'capture files differ from collection receipt: '+str(window))
    return len(actual)


def normalized_sidecar(path, root):
    """Sidecar with only the registered per-run fields reduced; anything else must match."""
    value = json.loads(Path(path).read_text())
    position = value.get('chunk_position')
    if position is not None:
        match = REQUEST_ID.fullmatch(position['request_id'])
        require(match is not None, 'unregistered request-id form: '+str(path))
        position['request_id'] = match.group(1)
        markers = position['active_markers']
        allowed = {str(Path(root)/d/MARKER): d+'/'+MARKER for d in ('logits', 'diagnostic')}
        require(isinstance(markers, list) and markers and all(m in allowed for m in markers),
                'active markers outside the compared window: '+str(path))
        # Window-relative, so the marker set itself (rank 0 also arms logits) must match.
        position['active_markers'] = sorted(allowed[m] for m in markers)
    return canonical(value)


def order(name):
    match = FILE_NAME.fullmatch(name)
    return (int(match.group(3)), int(match.group(1)), match.group(4), name) if match else (-1, -1, '', name)


def compare_rank(baseline, new, rank):
    left = regular_files(Path(baseline)/('rank'+rank)/'diagnostic')
    right = regular_files(Path(new)/('rank'+rank)/'diagnostic')
    sources = []
    for files in (left, right):
        names = [n for n in files if HOOK_SOURCE.fullmatch(n)]
        require(names, 'hook-source records required: rank'+rank)
        sources.append(collections.Counter(canonical(json.loads(files.pop(n).read_text())) for n in names))
    require(any(n.endswith('.bin') for n in left), 'baseline window has no payloads: rank'+rank)
    shared = sorted(set(left) & set(right), key=order)
    payload, sidecar = [], []
    for name in shared:
        require(name == MARKER or FILE_NAME.fullmatch(name) is not None, 'unregistered capture file: '+name)
        match = FILE_NAME.fullmatch(name)
        if match is not None:
            require(match.group(2) == rank, 'capture rank differs from directory: '+name)
        if name.endswith('.bin'):
            if sha(left[name]) != sha(right[name]):
                payload.append(name)
        elif name.endswith('.json'):
            if normalized_sidecar(left[name], baseline) != normalized_sidecar(right[name], new):
                sidecar.append(name)
    return {'files_compared': len(shared),
            'payloads_compared': sum(n.endswith('.bin') for n in shared),
            'only_in_baseline': sorted(set(left)-set(right), key=order),
            'only_in_new': sorted(set(right)-set(left), key=order),
            'payload_mismatches': payload, 'sidecar_mismatches': sidecar,
            'first_mismatch': next(iter(sorted(payload+sidecar, key=order)), None),
            'hook_source_records': [sum(sources[0].values()), sum(sources[1].values())],
            'hook_sources_equal': sources[0] == sources[1]}


def compare(baseline, new, baseline_receipt, new_receipt):
    verified = {'baseline': verify_receipt(baseline, baseline_receipt), 'new': verify_receipt(new, new_receipt)}
    ranks = {rank: compare_rank(baseline, new, rank) for rank in ('0', '1')}
    passed = all(r['hook_sources_equal'] and not (r['only_in_baseline'] or r['only_in_new']
                 or r['payload_mismatches'] or r['sidecar_mismatches']) for r in ranks.values())
    return {'schema': 'atlas-l1-capture-reproducibility-v2', 'baseline': str(baseline), 'new': str(new),
            'receipt_verified_files': verified,
            'per_run_fields': ['chunk_position.request_id random suffix', 'chunk_position.active_markers',
                               'hook-source record file-name process ID'],
            'ranks': ranks, 'passed': passed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, help='baseline reference-observed window directory')
    parser.add_argument('--baseline-receipt', required=True, help="baseline attempt's collection receipt.json")
    parser.add_argument('--new', required=True, help='new reference-observed window directory')
    parser.add_argument('--new-receipt', required=True, help="new attempt's collection receipt.json")
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    require(not out.exists(), 'report path must be fresh')
    try:
        report = compare(Path(args.baseline), Path(args.new), args.baseline_receipt, args.new_receipt)
    except Exception as exc:  # a refusal is an O2 fail, recorded rather than lost
        report = {'schema': 'atlas-l1-capture-reproducibility-v2', 'baseline': args.baseline, 'new': args.new,
                  'passed': False, 'refused': f'{type(exc).__name__}: {exc}'}
    report['comparator_sha256'] = sha(__file__)
    out.write_text(json.dumps(report, indent=2)+'\n')
    summary = {'passed': report['passed'], 'refused': report.get('refused')}
    for rank, result in report.get('ranks', {}).items():
        summary['rank'+rank] = {k: (len(v) if isinstance(v, list) and k != 'hook_source_records' else v)
                                for k, v in result.items()}
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
