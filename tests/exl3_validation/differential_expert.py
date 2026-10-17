# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded real-checkpoint thin/native expert diagnostic; never model parity.

Run inside the exact reference image with installed extension, Torch, CuPy,
safetensors, native CUDA sources, and an immutable checkpoint. Explicit missing
prerequisites fail. Output captures all stages even when numerical checks fail.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def validate_input(x):
    if x.dtype != np.float32 or x.ndim != 2 or not 1 <= len(x) <= 128:
        raise ValueError('input must be F32 [1..128, hidden] containing exact BF16 values')
    if not np.isfinite(x).all() or np.any(x.view(np.uint32) & 65535):
        raise ValueError('input must be finite and exactly BF16-representable')
    return np.ascontiguousarray(x)


def metrics(reference, candidate, absolute, relative):
    a, b = np.asarray(reference, np.float32), np.asarray(candidate, np.float32)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('stage shapes and finite values must match')
    error = np.abs(a - b)
    bad = error > absolute + relative * np.abs(a)
    return {'shape': list(a.shape), 'max_abs': float(error.max()),
            'rms': float(np.sqrt(np.mean(error.astype(np.float64) ** 2))),
            'violations': int(bad.sum()), 'elements': int(a.size)}


def load_parts(checkpoint, expert_prefix, rank, reference):
    from safetensors import safe_open
    index_path = checkpoint / 'model.safetensors.index.json'
    index = json.loads(index_path.read_text())['weight_map']
    parts, hashes = {}, {}
    for name, axis in (('gate', 'col'), ('up', 'col'), ('down', 'row')):
        pack = {}
        for suffix in ('trellis', 'suh', 'svh', 'mul1'):
            key = expert_prefix + '.' + name + '_proj.' + suffix
            shard = index[key]
            if Path(shard).name != shard:
                raise ValueError('index shard name must be a simple filename')
            with safe_open(checkpoint / shard, framework='pt', device='cpu') as source:
                full = source.get_tensor(key)
            hashes[key] = hashlib.sha256(full.numpy().tobytes()).hexdigest()
            fn = reference.shard_exl3_col if axis == 'col' else reference.shard_exl3_row
            pack[suffix] = fn(full, suffix, rank, 2).contiguous().cuda()
        parts[name] = pack
    return parts, hashes, index_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--expert-prefix', required=True)
    parser.add_argument('--reference-plugin', type=Path, required=True)
    parser.add_argument('--reference-plugin-sha256', required=True)
    parser.add_argument('--rank', type=int, choices=(0, 1), required=True)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--synthetic-seed', type=int)
    parser.add_argument('--synthetic-rows', type=int)
    parser.add_argument('--weight', type=float, required=True)
    parser.add_argument('--limit', type=float, required=True)
    parser.add_argument('--abs', dest='absolute', type=float, required=True)
    parser.add_argument('--rel', dest='relative', type=float, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--native-thin-cap', type=int,
                        help='explicitly enable new routed precision replay with actual reference scratch capacity')
    parser.add_argument('--native-mma-profile', nargs=4, type=int,
                        metavar=('TILE_K', 'TILE_N', 'SLICES', 'HALF_PARTIAL'))
    parser.add_argument('--native-fat-mma-profile', nargs=4, type=int,
                        metavar=('TILE_K', 'TILE_N', 'SLICES', 'HALF_PARTIAL'))
    args = parser.parse_args()
    if (args.native_mma_profile is None) != (args.native_fat_mma_profile is None):
        parser.error('production MMA requires explicit thin and fat profiles together')
    if args.native_fat_mma_profile is not None and tuple(args.native_fat_mma_profile) != (16, 128, 1, 0):
        parser.error('explicit production fat profile16/128/1/0 required')
    if args.native_mma_profile is not None and tuple(args.native_mma_profile) != (32, 256, 8, 1):
        parser.error('explicit thin MMA profile32/256/8/1 required')
    if args.native_mma_profile is not None and args.native_thin_cap is None:
        parser.error('MMA replay also requires explicit thin capacity')
    if not np.isfinite([args.weight, args.limit, args.absolute, args.relative]).all():
        parser.error('all numerical parameters must be finite')
    if min(args.limit, args.absolute, args.relative) < 0:
        parser.error('limit/tolerances must be nonnegative')
    if (args.input is None) == (args.synthetic_seed is None):
        parser.error('supply exactly one captured input or explicit synthetic seed')
    if args.input is not None and args.synthetic_rows is not None:
        parser.error('synthetic rows cannot accompany captured input')
    if args.synthetic_seed is not None and (args.synthetic_rows is None or not 1 <= args.synthetic_rows <= 128):
        parser.error('synthetic input requires explicit rows1..128')
    captured_input = validate_input(np.load(args.input, allow_pickle=False)) if args.input is not None else None
    if digest(args.reference_plugin) != args.reference_plugin_sha256:
        raise ValueError('installed reference plugin hash differs from frozen source')
    import torch
    import cupy as cp
    import exllamav3_ext
    if not torch.cuda.is_available() or cp.cuda.runtime.getDeviceCount() != 1:
        raise RuntimeError('runner requires exactly one visible real CUDA device')
    spec = importlib.util.spec_from_file_location('exl3_differential_reference', args.reference_plugin)
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)
    parts, hashes, index_path = load_parts(args.checkpoint, args.expert_prefix, args.rank, reference)
    hidden = parts['gate']['suh'].numel()
    intermediate = parts['gate']['svh'].numel()
    if args.input is not None:
        x = captured_input
        input_source = {'kind': 'captured', 'sha256': digest(args.input)}
    else:
        rng = np.random.default_rng(args.synthetic_seed)
        raw = rng.normal(0, 1, (args.synthetic_rows, hidden)).astype(np.float32)
        bits = raw.view(np.uint32)
        x = validate_input(((bits + 32767 + ((bits >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32))
        input_source = {'kind': 'synthetic_diagnostic_only', 'seed': args.synthetic_seed}
    if x.shape[1] != hidden or hidden % 128 or intermediate % 128:
        raise ValueError('input and expert H128 geometry differ')
    args.out.mkdir(parents=True, exist_ok=False)
    np.save(args.out / 'input.npy', x, allow_pickle=False)
    rows = len(x)
    dx = torch.from_numpy(x).cuda().to(torch.bfloat16)
    inners = [{name: reference.make_linear_exl3(p['trellis'], p['suh'], p['svh'], p['mul1'])
               for name, p in parts.items()}]
    layer = SimpleNamespace(w13_trellis=parts['gate']['trellis'],
                            _exl3_hidden_size=hidden, _exl3_intermediate_local=intermediate)
    reference.build_exl3_fused_state(layer, inners)
    layer._exl3_inners = inners
    if rows > layer._exl3_fused_temps[0].shape[1]:
        raise ValueError('input exceeds actual installed thin scratch capacity; select explicit rows before replay')
    ids = torch.zeros((rows, 1), dtype=torch.int64, device='cuda')
    weights = torch.full((rows, 1), args.weight, dtype=torch.float32, device='cuda')
    sorted_to_input = reference.map_topk_to_local(ids, 1, None).argsort()
    input_to_sorted = sorted_to_input.argsort()
    thin = reference.apply_exl3_fused_moe(dx, ids, weights, layer, inners, None, args.limit)
    torch.cuda.synchronize()
    arrays = {'input': x, 'reference_weighted_f32': thin.float().cpu().numpy(),
              'reference_weighted_bf16': thin.bfloat16().float().cpu().numpy(),
              'reference_sorted_to_input_row': sorted_to_input.cpu().numpy()}
    for name, position, width in (('up_xh', 1, hidden), ('up_yh', 3, intermediate),
                                  ('down_xh', 2, intermediate), ('down_yh', 0, hidden)):
        arrays['reference_' + name] = layer._exl3_fused_temps[position][0, :rows, :width].index_select(0, input_to_sorted).float().cpu().numpy()
    common = Path(__file__).resolve().parents[2] / 'kernels/gb10/glm-5.3-flash/exl3'
    diagnostic_source = r"""
extern "C" __global__ void diagnostic_f16_input_hadamard(
    const __nv_bfloat16* x, const __half* suh, float* xh,
    unsigned int rows, unsigned int k) {
    __shared__ float scratch[128];
    const unsigned int feature = blockIdx.x * 128u + threadIdx.x;
    const unsigned long long index = (unsigned long long)blockIdx.y * k + feature;
    const __half product = __hmul(__float2half_rn(__bfloat162float(x[index])), suh[feature]);
    xh[index] = __half2float(__float2half_rn(hadamard128(__half2float(product), scratch)));
}
"""
    module = cp.RawModule(code=(common / 'exl3.cu').read_text() + diagnostic_source,
                          options=('--std=c++17', '--fmad=false', '-I' + str(common)))
    funcs = {n: module.get_function('exl3_' + n) for n in
             ('input_hadamard', 'mul1_matmul', 'output_hadamard', 'output_hadamard_f32', 'silu_mul')}
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        cdx = cp.from_dlpack(dx.view(torch.uint16))
        compat_xh = cp.empty((rows, hidden), cp.float32)
        module.get_function('diagnostic_f16_input_hadamard')(
            (hidden // 128, rows), (128,),
            (cdx, cp.from_dlpack(parts['up']['suh']), compat_xh, np.uint32(rows), np.uint32(hidden)))
        arrays['diagnostic_f16_input_up_xh'] = cp.asnumpy(compat_xh)
        def linear(label, inp):
            p = parts[label]
            k, n = p['suh'].numel(), p['svh'].numel()
            bits = p['trellis'].shape[2] // 16
            xh, yh = cp.empty((rows, k), cp.float32), cp.empty((rows, n), cp.float32)
            out, f32 = cp.empty((rows, n), cp.uint16), cp.empty((rows, n), cp.float32)
            suh, svh, packed = (cp.from_dlpack(p[a]) for a in ('suh', 'svh', 'trellis'))
            funcs['input_hadamard']((k // 128, rows), (128,), (inp, suh, xh, np.uint32(rows), np.uint32(k)))
            funcs['mul1_matmul'](((n + 127) // 128, rows), (128,),
                                (xh, packed, yh, *map(np.uint32, (rows, n, k, bits))))
            for function, target in (('output_hadamard', out), ('output_hadamard_f32', f32)):
                funcs[function]((n // 128, rows), (128,), (yh, svh, target, np.uint32(rows), np.uint32(n)))
            arrays['native_' + label + '_xh'] = cp.asnumpy(xh)
            arrays['native_' + label + '_yh'] = cp.asnumpy(yh)
            arrays['native_' + label + '_f32_output_diagnostic'] = cp.asnumpy(f32)
            arrays['native_' + label + '_bf16'] = (cp.asnumpy(out).astype(np.uint32) << 16).view(np.float32)
            return out
        gate, up = linear('gate', cdx), linear('up', cdx)
        activation = cp.empty_like(gate)
        funcs['silu_mul'](((rows * intermediate + 255) // 256,), (256,),
                         (gate, up, activation, np.uint64(rows * intermediate), np.float32(args.limit), np.uint32(1)))
        arrays['native_activation_bf16'] = (cp.asnumpy(activation).astype(np.uint32) << 16).view(np.float32)
        linear('down', activation)
        # Hold the transformed input fixed to isolate packed GEMM arithmetic.
        for label in ('up', 'down'):
            p = parts[label]
            k, n = p['suh'].numel(), p['svh'].numel()
            bits = p['trellis'].shape[2] // 16
            fixed_xh = cp.asarray(arrays['reference_' + label + '_xh'])
            fixed_yh = cp.empty((rows, n), cp.float32)
            funcs['mul1_matmul'](((n + 127) // 128, rows), (128,),
                                (fixed_xh, cp.from_dlpack(p['trellis']), fixed_yh,
                                 *map(np.uint32, (rows, n, k, bits))))
            arrays['native_fixed_reference_' + label + '_yh_f32'] = cp.asnumpy(fixed_yh)
            arrays['native_fixed_reference_' + label + '_yh_f16'] = cp.asnumpy(fixed_yh).astype(np.float16).astype(np.float32)
    arrays['native_weighted_f32'] = arrays['native_down_bf16'] * np.float32(args.weight)
    raw = arrays['native_weighted_f32'].view(np.uint32)
    arrays['native_weighted_bf16'] = ((raw + 32767 + ((raw >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)
    precision_hashes = None
    if args.native_thin_cap is not None:
        if args.native_thin_cap != layer._exl3_fused_temps[0].shape[1]:
            raise ValueError('native thin cap differs from actual installed reference scratch geometry')
        from moe_precision import run_expert
        precision_source = common / 'exl3_moe_precision.cu'
        normalized = precision_source.read_text().replace('#include <math.h>', '')
        precision_module = cp.RawModule(code=normalized,
            options=('--std=c++17', '--fmad=false', '-I' + str(common)))
        mma_module = None if args.native_mma_profile is None else cp.RawModule(
            code=(common / 'exl3_moe_mma.cu').read_text() + '\n// header-hashes ' +
                 ' '.join(digest(common / name) for name in ('exl3_mma.cuh', 'exl3_common.cuh', 'exl3_expert.cuh')),
            options=('--std=c++17', '--fmad=false', '-I' + str(common)))
        for stage, value in run_expert(precision_module, dx, parts, args.weight, args.native_thin_cap, args.limit, mma_module, args.native_mma_profile, fat_mma_profile=args.native_fat_mma_profile).items():
            arrays['native_precision_' + stage] = value
        precision_hashes = {p.name: digest(p) for p in
            (precision_source, common / 'exl3_expert.cuh', common / 'exl3_common.cuh', Path(__file__).with_name('moe_precision.py'))}
        if args.native_mma_profile is not None:
            precision_hashes.update({name: digest(common / name) for name in ('exl3_moe_mma.cu', 'exl3_mma.cuh')})
    report = {'scope': 'single_expert_rank_local_diagnostic_not_model_parity',
              'expert_prefix': args.expert_prefix, 'input_shape': list(x.shape),
              'input_source': input_source, 'input_bytes_sha256': hashlib.sha256(x.tobytes()).hexdigest(),
              'checkpoint_index_sha256': digest(index_path), 'selected_tensor_sha256': hashes,
              'reference_plugin_sha256': digest(args.reference_plugin),
              'extension_sha256': digest(exllamav3_ext.__file__),
              'native_source_sha256': {p.name: digest(p) for p in (common / 'exl3.cu', common / 'exl3_common.cuh')},
              'harness_sha256': digest(__file__), 'rank': args.rank, 'tp': 2,
              'weight_f32': args.weight, 'weight_reference_f16': float(np.float16(args.weight)),
              'limit': args.limit, 'absolute': args.absolute, 'relative': args.relative,
              'device': torch.cuda.get_device_name(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
              'reference_scratch': {
                  'geometry': [list(t.shape) for t in layer._exl3_fused_temps],
                  'dtype': 'float16', 'group': 0, 'expert_count': 1,
                  'binding': {
                      'up_xh': {'buffer': 1, 'semantics': 'input H128 times up.suh, FP16'},
                      'up_yh': {'buffer': 3, 'semantics': 'up GEMM output before output H128, FP16'},
                      'down_xh': {'buffer': 2, 'semantics': 'fused gate/up output H128, activation, down input H128, FP16'},
                      'down_yh': {'buffer': 0, 'semantics': 'down GEMM output before output H128, FP16'}},
                  'native_matching_stages_dtype': 'float32',
                  'native_activation_dtype': 'bfloat16'},
              'stages': {}}
    for stage in ('up_xh', 'up_yh', 'down_xh', 'down_yh', 'weighted_f32', 'weighted_bf16'):
        report['stages'][stage] = metrics(arrays['reference_' + stage], arrays['native_' + stage], args.absolute, args.relative)
    report['f16_input_hadamard_diagnostic'] = metrics(
        arrays['reference_up_xh'], arrays['diagnostic_f16_input_up_xh'], args.absolute, args.relative)
    report['f16_input_hadamard_diagnostic']['exact_elements'] = int(np.sum(
        arrays['reference_up_xh'] == arrays['diagnostic_f16_input_up_xh']))
    report['fixed_reference_xh_gemm_diagnostic'] = {
        label + '_' + dtype: metrics(arrays['reference_' + label + '_yh'],
                                   arrays['native_fixed_reference_' + label + '_yh_' + dtype],
                                   args.absolute, args.relative)
        for label in ('up', 'down') for dtype in ('f32', 'f16')}
    report['pass'] = all(v['violations'] == 0 for v in report['stages'].values())
    if precision_hashes is not None:
        report['precision_native'] = {
            'source_sha256': precision_hashes, 'thin_cap': args.native_thin_cap,
            'mma_profile': args.native_mma_profile, 'fat_mma_profile': args.native_fat_mma_profile,
            'policy': ('explicit occupancy-selected tensor-core profiles' if args.native_mma_profile is not None
                       else 'occupancy-selected scalar GEMM diagnostic'),
            'stages': {stage: metrics(arrays['reference_' + stage], arrays['native_precision_' + stage], args.absolute, args.relative)
                for stage in ('up_xh', 'up_yh', 'down_xh', 'down_yh', 'weighted_f32', 'weighted_bf16')}}
        report['legacy_pass'] = report['pass']
        report['pass'] = all(v['violations'] == 0 for v in report['precision_native']['stages'].values())
    np.savez(args.out / 'stages.npz', **arrays)
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['stages'], indent=2))
    return 0 if report['pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
