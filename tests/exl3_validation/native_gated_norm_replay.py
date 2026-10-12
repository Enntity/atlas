# SPDX-License-Identifier: AGPL-3.0-only
"""Replay the candidate's attested embedded KDA PTX with explicit BF16 operands.

Inputs are .npy little-endian uint16 bit patterns: core [M,32,128], gate
[M,4096], resident weight [128], and captured output [M,32,128]. Epsilon
must be supplied explicitly from the bound invocation. Every launch receives
fresh allocations. No source compilation or mathematical substitute is used.
"""
import argparse
import json
import re
from pathlib import Path
import numpy as np
from native_ptx import sha, validate_binding, load_module

KDA_CONTRACT = {
    'kda': {'entries': ('kda_sigmoid_gated_rms_norm',),
            'argument_counts': {'kda_sigmoid_gated_rms_norm': 7},
            'sources': ('kda.cu', 'kda_recurrent_body.cuh', 'atlas_pdl.cuh')},
}


def validate_operands(core, gate, weight, epsilon):
    if (core.ndim != 3 or core.shape[0] <= 0 or core.shape[0] > 0xffffffff // 32
            or core.shape[1:] != (32, 128)
            or gate.shape != (core.shape[0], 4096) or weight.shape != (128,)):
        raise ValueError('explicit core [M,32,128], gate [M,4096], weight [128] required')
    for value in (core, gate, weight):
        if value.dtype.str != '<u2' or not value.flags.c_contiguous:
            raise ValueError('contiguous little-endian uint16 BF16 bits required')
        if np.any((value & 0x7f80) == 0x7f80):
            raise ValueError('finite BF16 operands required')
    if not np.isfinite(epsilon) or epsilon <= 0 or epsilon > float(np.finfo(np.float32).max):
        raise ValueError('finite positive representable explicit epsilon required')
    epsilon = np.float32(epsilon)
    if epsilon == 0: raise ValueError('epsilon underflows the runtime f32 ABI')
    return (core.shape[0] * 32,), (128,), epsilon


def replay(cp, kernel, core, gate, weight, epsilon):
    grid, block, epsilon = validate_operands(core, gate, weight, epsilon)
    # Keep raw BF16 bytes: CUDA interprets these uint16 allocations as BF16.
    operands = [cp.array(value.reshape(-1), dtype=cp.uint16, copy=True)
                for value in (core, gate, weight)]
    guard = np.uint16(0x5a5a)
    output = cp.full(core.size + 64, guard, dtype=cp.uint16)
    kernel(grid, block, (*operands, output, np.uint32(32), np.uint32(128), epsilon))
    cp.cuda.get_current_stream().synchronize()
    if not np.all(cp.asnumpy(output[core.size:]) == guard):
        raise RuntimeError('gated normalization output guard overwritten')
    for device, original in zip(operands, (core, gate, weight)):
        if not np.array_equal(cp.asnumpy(device), original.reshape(-1)):
            raise RuntimeError('gated normalization operand mutated')
    result = cp.asnumpy(output[:core.size]).reshape(core.shape)
    if np.any((result & 0x7f80) == 0x7f80):
        raise RuntimeError('gated normalization produced nonfinite output')
    return result


def perturbations(core, gate, weight):
    controls = {}
    for index, name in enumerate(('input', 'gate', 'weight')):
        values = [value.copy() for value in (core, gate, weight)]
        selected = values[index]
        if name in ('input', 'weight'):
            selected ^= np.uint16(0x8000)  # Flip signs while retaining RMS magnitude.
        else:
            floats = (selected.astype(np.uint32) << 16).view(np.float32) + np.float32(1)
            bits = floats.view(np.uint32)
            rounded = bits + np.uint32(0x7fff) + ((bits >> 16) & np.uint32(1))
            selected[:] = (rounded >> 16).astype(np.uint16)
        controls[name] = tuple(values)
    return controls


def device_properties(value):
    """Make CUDA properties JSON-compatible without losing binary identifiers."""
    if isinstance(value, bytes):
        return {'encoding': 'hex', 'value': value.hex()}
    if isinstance(value, dict):
        return {str(key): device_properties(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [device_properties(item) for item in value]
    if isinstance(value, np.generic):
        return device_properties(value.item())
    if isinstance(value, np.ndarray):
        return device_properties(value.tolist())
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError(f'Unsupported CUDA device property: {type(value)!r}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('core', 'gate', 'weight', 'expected', 'ptx-dir', 'ptx-binding', 'common', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--epsilon', type=float, required=True)
    args = parser.parse_args()
    core, gate, weight, expected = [np.load(path, allow_pickle=False) for path in
                                   (args.core, args.gate, args.weight, args.expected)]
    validate_operands(core, gate, weight, args.epsilon)
    if expected.shape != core.shape or expected.dtype.str != '<u2':
        raise ValueError('captured expected BF16 output must match core geometry')
    binding = validate_binding(args.ptx_dir, args.ptx_binding, args.common, KDA_CONTRACT)
    if not re.search(r'(?:^|@)sha256:[0-9a-f]{64}$', binding['image']):
        raise ValueError('immutable sha256 image identity required')
    ptx = (args.ptx_dir / binding['modules']['kda']['file']).read_text()
    signature = re.search(r'\.entry\s+kda_sigmoid_gated_rms_norm\s*\((.*?)\)', ptx, re.S)
    if not signature or re.findall(r'\.param\s+\.(\w+)', signature[1]) != ['u64'] * 4 + ['u32', 'u32', 'f32']:
        raise ValueError('runtime gated norm argument types differ from production ABI')
    # Immutable output directories preserve failures as well as successes.
    args.out.mkdir(parents=True, exist_ok=False)
    import cupy as cp
    kernel = load_module(cp, args.common, 'kda', args.ptx_dir, binding).get_function(
        'kda_sigmoid_gated_rms_norm')
    baseline = replay(cp, kernel, core, gate, weight, args.epsilon)
    np.save(args.out / 'baseline.npy', baseline)
    results = {'self_reproduction': bool(np.array_equal(baseline, expected)),
               'unequal_elements': int(np.count_nonzero(baseline != expected)),
               'epsilon_f32': float(np.float32(args.epsilon)),
               'grid': [core.shape[0] * 32], 'block': [128],
               'ptx_binding_sha256': sha(args.ptx_binding), 'binding': binding,
               'inputs': {name: {'file': str(path), 'sha256': sha(path)} for name, path in
                          zip(('core', 'gate', 'weight', 'expected'),
                              (args.core, args.gate, args.weight, args.expected))},
               'controls': {}}
    for name, operands in perturbations(core, gate, weight).items():
        output = replay(cp, kernel, *operands, args.epsilon)
        np.save(args.out / (name + '.npy'), output)
        results['controls'][name] = {'engaged': bool(np.any(output != baseline)),
                                    'unequal_elements': int(np.count_nonzero(output != baseline))}
    results['device'] = device_properties(
        cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id))
    results['runtime_version'] = cp.cuda.runtime.runtimeGetVersion()
    results['driver_version'] = cp.cuda.runtime.driverGetVersion()
    results['harness_sha256'] = sha(__file__)
    (args.out / 'receipt.json').write_text(json.dumps(results, indent=2) + '\n')
    if not results['self_reproduction'] or not all(v['engaged'] for v in results['controls'].values()):
        raise RuntimeError('self reproduction or operand engagement failed; receipt preserved')


if __name__ == '__main__': main()
