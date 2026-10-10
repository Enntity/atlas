# SPDX-License-Identifier: AGPL-3.0-only
"""No-Torch candidate classic replay against immutable reference BF16 stages."""
import argparse
import ctypes
import json
import sys
from pathlib import Path
if any(name == 'torch' or name.startswith('torch.') for name in sys.modules):
    raise RuntimeError('Torch was imported before candidate-only replay; use Python -S and explicit package paths')
import numpy as np
from cublas_oracle import run_bf16
from cublaslt_oracle import file_hash
from differential_router import load_capture
from shared_reconstruction import byte_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input-json', 'gate-up-json', 'down-json', 'reference-stages',
                 'reference-report', 'candidate-cublas', 'candidate-cublaslt', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--workspace-bytes', type=int, required=True)
    args = parser.parse_args()
    if args.workspace_bytes < 0:
        parser.error('explicit nonnegative workspace required')
    # Bind candidate dependency before importing CuPy; never import Torch here.
    candidate_paths = [p.resolve(strict=True) for p in (args.candidate_cublas, args.candidate_cublaslt)]
    ctypes.CDLL(str(candidate_paths[1]), mode=ctypes.RTLD_GLOBAL)
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() != 1:
        raise RuntimeError('exactly one real GPU required')
    arrays, identities, hashes = [], [], {}
    for path in (args.input_json, args.gate_up_json, args.down_json):
        values, metadata, raw = load_capture(path)
        if metadata['dtype'] != 'bf16-le':
            raise ValueError('resident BF16 capture required')
        if metadata.get('sha256', file_hash(raw)) != file_hash(raw):
            raise ValueError('resident capture hash mismatch')
        arrays.append(cp.asarray((values.view(np.uint32) >> 16).astype(np.uint16)))
        identities.append((metadata['rank'], metadata['layer']))
        hashes.update({str(path): file_hash(path), str(raw): file_hash(raw)})
    if len(set(identities)) != 1:
        raise ValueError('input and weights have different rank/layer identities')
    x, gu, down = arrays
    binding = json.loads(args.reference_report.read_text())
    if tuple(binding['rank_layer']) != identities[0] or binding['output_sha256'] != file_hash(args.reference_stages):
        raise ValueError('reference report rank/layer or stage hash mismatch')
    if any(value not in binding['source_sha256'].values() for value in hashes.values()):
        raise ValueError('reference report does not bind the exact input and resident weight bytes')
    hashes[str(args.reference_report)] = file_hash(args.reference_report)
    with np.load(args.reference_stages) as stages:
        expected = {name: stages['reference_' + name].copy()
                    for name in ('gate_up', 'activation', 'down')}
    if expected['gate_up'].shape != (len(x), len(gu)) or expected['down'].shape != (len(x), len(down)):
        raise ValueError('reference stage geometry differs from captured matrices')
    act = cp.asarray(expected['activation'])
    results, outputs = {}, {}
    for reduced in (False, True):
        for name, input_bits, weight_bits in [('gate_up', x, gu), ('down', act, down)]:
            output, metadata = run_bf16(input_bits, weight_bits, candidate_paths[0],
                                        reduced_precision=reduced, compute=68, algorithm=99, workspace_bytes=args.workspace_bytes)
            key = str(reduced) + '_' + name
            outputs[key] = output
            results[key] = {'metrics': byte_metrics(expected[name], output), 'dispatch': metadata}
    maps = [line for line in Path('/proc/self/maps').read_text().splitlines() if 'libcublas' in line]
    loaded = {Path(line.split()[-1]).resolve(strict=True) for line in maps if line.split()[-1].startswith('/')}
    if loaded != set(candidate_paths):
        raise RuntimeError('classic replay loaded BLAS outside explicit candidate library pair')
    if any(name == 'torch' or name.startswith('torch.') for name in sys.modules):
        raise RuntimeError('candidate-only replay imported Torch')
    args.out.mkdir(exist_ok=False, parents=True)
    np.savez(args.out / 'stages.npz', **outputs)
    hashes[str(args.reference_stages)] = file_hash(args.reference_stages)
    for name in ('shared_classic_oracle.py', 'cublas_oracle.py', 'cublaslt_oracle.py',
                 'differential_router.py', 'shared_reconstruction.py'):
        path = Path(__file__).with_name(name); hashes[str(path)] = file_hash(path)
    report = {'scope': 'independent no-Torch component replay, no full-model qualification',
              'rank_layer': identities[0], 'metrics': results, 'source_sha256': hashes,
              'loaded_blas': {str(path): file_hash(path) for path in loaded},
              'loaded_library_maps': maps, 'cupy': cp.__version__, 'torch_imported': False}
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(results))
    return 0 if all(value['metrics']['exact'] for value in results.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
