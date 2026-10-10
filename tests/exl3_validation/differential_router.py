# SPDX-License-Identifier: AGPL-3.0-only
"""Actual-checkpoint router arithmetic diagnostic, never full-model parity."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_capture(path):
    metadata = json.loads(path.read_text())
    name = metadata['file']
    if Path(name).name != name:
        raise ValueError('capture must use a local filename')
    raw = path.parent / name
    width = {'bf16-le': 2, 'f32-le': 4}[metadata['dtype']]
    if raw.stat().st_size != int(np.prod(metadata['shape'])) * width:
        raise ValueError('capture byte size must match exact shape')
    if metadata['dtype'] == 'bf16-le':
        values = (np.fromfile(raw, '<u2').astype(np.uint32) << 16).view(np.float32)
    else:
        values = np.fromfile(raw, '<f4')
    values = values.reshape(metadata['shape'])
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('finite rank2 capture required')
    return values, metadata, raw


def validate_weights(values, metadata, path):
    if (values.dtype != np.float16 or values.ndim != 2 or
            list(values.shape) != metadata['shape'] or not np.isfinite(values).all()):
        raise ValueError('exact finite F16 checkpoint matrix required')
    if digest(path) != metadata['file_sha256']:
        raise ValueError('checkpoint weight file hash changed')
    if hashlib.sha256(values.tobytes()).hexdigest() != metadata['selected_tensor_sha256']:
        raise ValueError('checkpoint tensor payload hash changed')
    expanded = values.astype(np.float32)
    if np.any(expanded.view(np.uint32) & 65535):
        raise ValueError('this diagnostic requires exactly BF16-representable weights')
    return expanded


def compare(reference, candidate):
    if reference.shape != candidate.shape or not np.isfinite(candidate).all():
        raise ValueError('finite full-shape output required')
    error = np.abs(reference.astype(np.float64) - candidate)
    return {'shape': list(reference.shape), 'exact': bool(np.array_equal(reference, candidate)),
            'unequal': int(np.count_nonzero(reference != candidate)),
            'max_abs': float(error.max()), 'rms': float(np.sqrt(np.mean(error ** 2)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-json', type=Path, required=True)
    parser.add_argument('--logits-json', type=Path, required=True)
    parser.add_argument('--weight', type=Path, required=True)
    parser.add_argument('--weight-json', type=Path, required=True)
    parser.add_argument('--gate-source', type=Path, required=True)
    parser.add_argument('--gate-source-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cublaslt-library', type=Path)
    parser.add_argument('--runtime-wrapper', type=Path)
    args = parser.parse_args()
    if (args.cublaslt_library is None) != (args.runtime_wrapper is None):
        parser.error('cuBLASLt library and production wrapper must both be explicit')
    x, xm, xp = load_capture(args.input_json)
    logits, lm, lp = load_capture(args.logits_json)
    if xm['dtype'] != 'bf16-le' or lm['dtype'] != 'f32-le':
        parser.error('input must be BF16, logits F32 transport')
    for key in ['layer', 'rank', 'invocation']:
        if xm[key] != lm[key]:
            parser.error('input/logits capture identity differs')
    wm = json.loads(args.weight_json.read_text())
    w = validate_weights(np.load(args.weight, allow_pickle=False), wm, args.weight)
    if wm['key'] != f"model.language_model.layers.{xm['layer']}.mlp.gate.weight":
        parser.error('weight key must match captured layer')
    if x.shape[1] != w.shape[1] or logits.shape != (len(x), len(w)):
        parser.error('router exact matrix/output shapes differ')
    if digest(args.gate_source) != args.gate_source_sha256:
        parser.error('installed GateLinear source hash changed')
    import torch
    import cupy as cp
    if not torch.cuda.is_available() or cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError('real CUDA required')
    device = torch.cuda.current_device()
    xb = torch.tensor(x, device='cuda', dtype=torch.bfloat16)
    wb = torch.tensor(w, device='cuda', dtype=torch.bfloat16)
    actual = torch.nn.functional.linear(xb, wb).float().cpu().numpy()
    common = Path(__file__).resolve().parents[2] / 'kernels/gb10/common'
    source = common / 'exl3_moe.cu'
    module = cp.RawModule(code=source.read_text(),
        options=('--std=c++17', '--fmad=false', '-I' + str(common)))
    kernel = module.get_function('exl3_router')
    dx = cp.asarray((x.view(np.uint32) >> 16).astype(np.uint16))
    dw = cp.asarray(w)
    guard = np.float32(1.234375)
    out = cp.full(logits.size + 64, guard, cp.float32)
    kernel(((len(w) + 127) // 128, len(x)), (128,),
        (dx, dw, out, *map(np.uint32, (len(x), len(w), x.shape[1]))))
    cp.cuda.get_current_stream().synchronize()
    result = cp.asnumpy(out)
    if not np.all(result[logits.size:] == guard):
        raise RuntimeError('native router overwrote output guard')
    native = result[:logits.size].reshape(logits.shape)
    native_bf16 = torch.tensor(native).bfloat16().float().numpy()
    variants = {'torch_bf16_linear': actual, 'native_f32_router': native,
                'native_f32_router_bf16_cast': native_bf16, 'captured_logits': logits}
    lt_metadata = None
    if args.cublaslt_library is not None:
        from cublaslt_oracle import run_bf16
        bits, lt_metadata = run_bf16(dx, cp.asarray((w.view(np.uint32) >> 16).astype(np.uint16)),
                                     args.cublaslt_library)
        variants['atlas_cublaslt_bf16'] = (bits.astype(np.uint32) << 16).view(np.float32)
    args.out.mkdir(parents=True, exist_ok=False)
    output_hashes = {}
    for name, value in variants.items():
        path = args.out / (name + '.npy'); np.save(path, value)
        output_hashes[path.name] = digest(path)
    properties = torch.cuda.get_device_properties(device)
    inputs = [args.input_json, xp, args.logits_json, lp, args.weight,
              args.weight_json, args.gate_source, source, Path(__file__)]
    inputs += sorted(common.glob('exl3*.cuh'))
    if args.runtime_wrapper is not None:
        inputs += [args.runtime_wrapper, Path(__file__).with_name('cublaslt_oracle.py')]
    report = {'scope': 'same captured input/checkpoint router only; no full-model parity',
        'identity': {k: xm[k] for k in ['layer', 'rank', 'invocation']},
        'source_binding': {str(p): digest(p) for p in inputs}, 'weight_metadata': wm,
        'torch': {'version': torch.__version__, 'cuda': torch.version.cuda,
            'gpu': properties.name, 'capability': list(torch.cuda.get_device_capability(device)),
            'allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'allow_bf16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            'allow_fp16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'config': torch.__config__.show()},
        'cupy': {'version': cp.__version__, 'runtime': cp.cuda.runtime.runtimeGetVersion(),
                 'driver': cp.cuda.runtime.driverGetVersion(), 'compiler': 'NVRTC --fmad=false'},
        'captured_logits_all_bf16_representable': bool(not np.any(logits.view(np.uint32) & 65535)),
        'output_sha256': output_hashes,
        'atlas_cublaslt': lt_metadata,
        'metrics': {name: compare(logits, value) for name, value in variants.items()}}
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['metrics']))
    gates = ['torch_bf16_linear']
    if lt_metadata is not None:
        gates.append('atlas_cublaslt_bf16')
    return 0 if all(report['metrics'][name]['exact'] for name in gates) else 1


if __name__ == '__main__':
    raise SystemExit(main())
