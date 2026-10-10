# SPDX-License-Identifier: AGPL-3.0-only
"""Exact native-tail/legacy CUDA selector oracle; requires a real CUDA GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def oracle(scores, length, topk, pool, width, reserved):
    output = np.full(width, -1, np.int32)
    valid = (length > 0 and pool > 0 and topk > 0 and topk % pool == 0
             and width >= topk + pool - 1 and length <= len(scores) * pool
             and (not reserved or topk >= pool))
    threshold = topk - 1 if reserved else topk
    if not valid:
        return output, 0
    if length <= threshold:
        return output, length
    history = topk - pool if reserved else topk
    pools = length // pool
    count = min(history // pool, pools)
    cleaned = np.where(np.isnan(scores[:pools]), -np.inf, scores[:pools])
    negative_zero = np.signbit(cleaned) & (cleaned == 0)
    picked = np.sort(np.lexsort((np.arange(pools), negative_zero, -cleaned))[:count])
    output[:count * pool] = (picked[:, None] * pool + np.arange(pool)).reshape(-1)
    output[history:history + length % pool] = np.arange(pools * pool, length)
    return output, 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() != 1:
        raise RuntimeError('exactly one visible real CUDA GPU required')
    # NVRTC is freestanding in the reference image (no system math.h).
    # CUDA math intrinsics are provided by NVRTC; supply only the IEEE infinity macro.
    source = args.source.read_text().replace('#include <math.h>',
                '#define INFINITY (__int_as_float(0x7f800000))')
    module = cp.RawModule(code=source, options=('--std=c++17', '--fmad=false', '-I' + str(args.source.parent)))
    functions = [module.get_function('glm_index_topk_expand_dynamic' + suffix)
                 for suffix in ('', '_tail_reserved')]
    rng = np.random.default_rng(20261010)
    scores = rng.integers(-4, 5, 1024).astype(np.float32)
    scores[:5] = [np.nan, np.inf, -np.inf, -0., 0.]
    dscores = cp.asarray(scores)
    width, topk, pool = 2051, 2048, 4
    guarded = cp.full(width + 2, -923, cp.int32)
    dlength, ddense = cp.zeros(1, cp.uint32), cp.zeros(1, cp.uint32)
    cases, arrays = [], {}
    # Reuse the same buffers to prove long->short resets and repeated tail transitions.
    lengths = [4096, 1, 2047, 2048, 2049, 2050, 2051, 2052, 2053, 3000, 0, 4097, 2]
    for reserved, function in enumerate(functions):
        for length in lengths:
            guarded.fill(-923)
            ddense.fill(777)
            dlength.fill(length)
            function((1,), (256,), (dscores, guarded[1:-1], np.uint32(1), dlength,
                     np.uint32(len(scores)), np.uint32(topk), np.uint32(pool),
                     np.uint32(width), ddense), shared_mem=16)
            actual = cp.asnumpy(guarded)
            expected, dense = oracle(scores, length, topk, pool, width, bool(reserved))
            np.testing.assert_array_equal(actual[1:-1], expected)
            np.testing.assert_array_equal(actual[[0, -1]], [-923, -923])
            assert int(ddense.get()[0]) == dense
            name = ('native' if reserved else 'legacy') + '_' + str(length)
            arrays[name] = actual
            cases.append({'case': name, 'dense_length': dense, 'selected': int(np.sum(expected >= 0))})
    args.out.mkdir(parents=True, exist_ok=False)
    np.savez(args.out / 'outputs.npz', scores=scores, **arrays)
    report = {'pass': True, 'source_sha256': hashlib.sha256(args.source.read_bytes()).hexdigest(),
              'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'nvrtc_source_sha256': hashlib.sha256(source.encode()).hexdigest(),
              'nvrtc_math_header': 'CUDA builtin math intrinsics + explicit IEEE infinity macro',
              'device': cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
              'cupy': cp.__version__, 'cases': cases}
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
