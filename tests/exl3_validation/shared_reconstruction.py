# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded original-basis shared-weight diagnostic; strict byte gates, no model claim."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

import numpy as np
from differential_router import digest, load_capture


def shard_weight(value, projection, rank):
    if projection in ('gate', 'up'):
        if value.shape[0] % 2:
            raise ValueError('TP2 output width must divide evenly')
        return np.split(value, 2, axis=0)[rank].copy()
    if projection != 'down' or value.shape[1] % 2:
        raise ValueError('TP2 input width must divide evenly')
    return np.split(value, 2, axis=1)[rank].copy()


def byte_metrics(a, b):
    if a.shape != b.shape or a.dtype != np.uint16 or b.dtype != np.uint16:
        raise ValueError('matching BF16 bit-storage geometry required')
    av = (a.astype(np.uint32) << 16).view(np.float32)
    bv = (b.astype(np.uint32) << 16).view(np.float32)
    if not np.isfinite(av).all() or not np.isfinite(bv).all():
        raise ValueError('finite reconstruction required')
    return {'shape': list(a.shape), 'exact': bool(np.array_equal(a, b)),
            'unequal_bytes_elements': int(np.count_nonzero(a != b)),
            'max_abs': float(np.abs(av - bv).max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--shared-prefix', required=True)
    parser.add_argument('--reference-plugin', type=Path, required=True)
    parser.add_argument('--reference-plugin-sha256', required=True)
    parser.add_argument('--rank', type=int, choices=(0, 1), required=True)
    parser.add_argument('--gate-up-json', type=Path)
    parser.add_argument('--down-json', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    layer_match = re.search(r'\.layers\.(\d+)\.mlp\.shared_experts$', args.shared_prefix)
    if layer_match is None:
        parser.error('shared-only reconstruction required; routed unpack prohibited')
    if (args.gate_up_json is None) != (args.down_json is None):
        parser.error('both actual loaded shared captures are required together')
    if digest(args.reference_plugin) != args.reference_plugin_sha256:
        parser.error('actual installed plugin source hash changed')
    import torch
    import cupy as cp
    from safetensors import safe_open
    if not torch.cuda.is_available() or cp.cuda.runtime.getDeviceCount() != 1:
        raise RuntimeError('one visible real CUDA device required')
    spec = importlib.util.spec_from_file_location('exl3_shared_reference', args.reference_plugin)
    reference = importlib.util.module_from_spec(spec); sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)
    common = Path(__file__).resolve().parents[2] / 'kernels/gb10/common'
    source = common / 'exl3_shared.cu'
    headers = sorted(common.glob('exl3*.cuh'))
    bindings = {str(p): digest(p) for p in [source, *headers, args.reference_plugin, Path(__file__)]}
    # Include hashes in the compilation unit so header changes invalidate NVRTC caches.
    code = source.read_text() + '\n// included-source-hashes ' + json.dumps(bindings)
    module = cp.RawModule(code=code, options=('--std=c++17', '--fmad=false', '-I'+str(common)))
    kernels = {name: module.get_function('exl3_shared_'+name)
               for name in ('decode', 'left_hadamard', 'right_hadamard')}
    index_path = args.checkpoint / 'model.safetensors.index.json'
    index = json.loads(index_path.read_text())['weight_map']
    bindings[str(index_path)] = digest(index_path)
    args.out.mkdir(parents=True, exist_ok=False)
    metrics, hashes, native_local, ref_local = {}, {}, {}, {}
    loaded_metadata = {}
    guard = np.float32(12345)
    for projection in ('gate', 'up', 'down'):
        parts = {}
        for suffix in ('trellis', 'suh', 'svh', 'mul1'):
            key = args.shared_prefix + '.' + projection + '_proj.' + suffix
            shard = index[key]
            if Path(shard).name != shard:
                raise ValueError('simple indexed shard filename required')
            with safe_open(args.checkpoint / shard, framework='pt', device='cpu') as file:
                full = file.get_tensor(key)
            hashes[key] = hashlib.sha256(full.numpy().tobytes()).hexdigest()
            parts[suffix] = full.contiguous().cuda()
        k, n = parts['suh'].numel(), parts['svh'].numel()
        bits = parts['trellis'].shape[-1] // 16
        if k % 128 or n % 128 or not 1 <= bits <= 8:
            raise ValueError('supported H128 mul1 payload required')
        inner = reference.make_linear_exl3(parts['trellis'], parts['suh'], parts['svh'],
                                           parts['mul1'], out_dtype=torch.float16)
        ref_inner = inner.get_inner_weight_tensor().float().cpu().numpy()
        ref_weight = reference.glm53_dequant_packed_linear(parts, torch.device('cuda'))
        ref_bits = ref_weight.view(torch.uint16).cpu().numpy()
        arrays = [cp.full(k*n+64, guard, cp.float32) for _ in range(2)]
        weight = cp.full(k*n+64, np.uint16(0x42f6), cp.uint16)
        inner_native, temp = arrays
        packed, suh, svh = [cp.from_dlpack(parts[name]) for name in ('trellis', 'suh', 'svh')]
        kernels['decode'](((n+127)//128,k), (128,), (packed,inner_native,*map(np.uint32,(k,n,bits))))
        kernels['left_hadamard'](((n+127)//128,k), (128,), (inner_native,suh,temp,*map(np.uint32,(k,n))))
        kernels['right_hadamard'](((n+127)//128,k), (128,), (temp,svh,weight,*map(np.uint32,(k,n))))
        cp.cuda.get_current_stream().synchronize()
        for value in arrays:
            if not np.all(cp.asnumpy(value[k*n:]) == guard):
                raise RuntimeError('shared F32 reconstruction guard overwritten')
        if not np.all(cp.asnumpy(weight[k*n:]) == np.uint16(0x42f6)):
            raise RuntimeError('shared BF16 reconstruction guard overwritten')
        decoded = cp.asnumpy(inner_native[:k*n]).reshape(k,n)
        if not np.array_equal(decoded, ref_inner):
            raise RuntimeError('native packed decode differs from actual reference inner weight')
        native_bits = cp.asnumpy(weight[:k*n]).reshape(n,k)
        native_local[projection] = shard_weight(native_bits, projection, args.rank)
        ref_local[projection] = shard_weight(ref_bits, projection, args.rank)
        metrics[projection+'_full'] = byte_metrics(ref_bits, native_bits)
        for label, value in [('native', native_local[projection]), ('reference', ref_local[projection])]:
            path = args.out / (projection+'-'+label+'.npy'); np.save(path,value)
            bindings[str(path)] = digest(path)
        del inner, ref_inner, ref_weight, parts, arrays, weight, packed, suh, svh, inner_native, temp
        torch.cuda.empty_cache(); cp.get_default_memory_pool().free_all_blocks()
    if args.gate_up_json is not None:
        for capture, projection in [(args.gate_up_json, 'gate_up'), (args.down_json,'down')]:
            values, metadata, raw = load_capture(capture)
            if (metadata['dtype'] != 'bf16-le' or metadata['rank'] != args.rank
                    or metadata['layer'] != int(layer_match.group(1))):
                raise ValueError('actual BF16 shared capture rank mismatch')
            if metadata['sha256'] != digest(raw):
                raise ValueError('actual loaded shared weight payload hash changed')
            loaded_metadata[projection] = metadata
            actual = (values.view(np.uint32) >> 16).astype(np.uint16)
            for label, weights in [('reference',ref_local),('native',native_local)]:
                expected = np.concatenate([weights['gate'],weights['up']]) if projection == 'gate_up' else weights['down']
                metrics[label+'_loaded_'+projection] = byte_metrics(actual,expected)
            bindings.update({str(capture):digest(capture),str(raw):digest(raw)})
    report = {'scope':'shared-only weight reconstruction; no forward or full-model parity',
        'rank':args.rank,'shared_prefix':args.shared_prefix,'source_sha256':bindings,
        'checkpoint_selected_tensor_sha256':hashes,'metrics':metrics,
        'actual_loaded_weight_metadata':loaded_metadata,
        'has_actual_loaded_weight_oracle':args.gate_up_json is not None,
        'torch':{'version':torch.__version__,'cuda':torch.version.cuda,
            'allow_tf32':torch.backends.cuda.matmul.allow_tf32,
            'float32_matmul_precision':torch.get_float32_matmul_precision(),
            'gpu':torch.cuda.get_device_name(),'capability':list(torch.cuda.get_device_capability())},
        'cupy':{'version':cp.__version__,'driver':cp.cuda.runtime.driverGetVersion()}}
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(metrics))
    return 0 if all(item['exact'] for item in metrics.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
