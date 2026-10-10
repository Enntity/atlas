# SPDX-License-Identifier: AGPL-3.0-only
"""Installed vLLM dense activation vs native CUDA on captured BF16 inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def captured_input(path, sidecar, expected_sha):
    if sha(path) != expected_sha:
        raise ValueError('captured gate/up bytes differ from pinned input')
    metadata = json.loads(Path(sidecar).read_text())
    shape = metadata['shape']
    if metadata['dtype'] != 'bf16-le' or len(shape) != 2 or min(shape) <= 0 or shape[1] % 2:
        raise ValueError('expected nonempty BF16 gate/up rows with even width')
    raw = np.fromfile(path, '<u2')
    if raw.size != shape[0] * shape[1]:
        raise ValueError('gate/up capture shape and byte count differ')
    values = (raw.astype(np.uint32) << 16).view(np.float32).reshape(shape)
    if not np.isfinite(values).all():
        raise ValueError('finite captured input required')
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-source', type=Path, required=True)
    parser.add_argument('--input-bin', type=Path, required=True)
    parser.add_argument('--input-sidecar', type=Path, required=True)
    parser.add_argument('--input-sha256', required=True)
    parser.add_argument('--reference-binary-sha256', required=True)
    parser.add_argument('--limit', type=float, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.limit) or args.limit <= 0:
        raise ValueError('finite positive clamp limit required')
    values = captured_input(args.input_bin, args.input_sidecar, args.input_sha256)
    import torch
    import cupy as cp
    import vllm._C_stable_libtorch as compiled_ops
    if sha(compiled_ops.__file__) != args.reference_binary_sha256:
        raise ValueError('installed reference CUDA binary differs from pinned source build')
    if not torch.cuda.is_available() or cp.cuda.runtime.getDeviceCount() != 1:
        raise RuntimeError('one visible real CUDA device required')
    module = cp.RawModule(code=args.native_source.read_text(),
        options=('--std=c++17', '--fmad=false', '-I' + str(args.native_source.parent)))
    kernel = module.get_function('exl3_silu_mul_dense')
    arrays, checks = {}, {}
    rng = np.random.default_rng(20261010)
    synthetic = rng.normal(size=(3, 1024)).astype(np.float32) * 20
    synthetic[0, :12] = [-100, -20, -10, -1, -0., 0., 1, 10, 20, 100, .5, -.5]
    synthetic[0, 512:524] = [100, -100, 20, -20, 10, -10, 1, -1, 0., -0., .5, -.5]
    for label, x in (('captured', values), ('saturation', synthetic)):
        dx = torch.from_numpy(x).cuda().bfloat16()
        gate, up = dx.chunk(2, dim=1)
        gate, up = gate.contiguous(), up.contiguous()
        count = gate.numel()
        arrays[label + '_input'] = dx.float().cpu().numpy()
        for clamp, limit in ((0, args.limit), (1, args.limit), (1, 1.)):
            reference = torch.empty_like(gate)
            if clamp:
                torch.ops._C.silu_and_mul_with_clamp(reference, dx, limit, 1., 0.)
            else:
                torch.ops._C.silu_and_mul(reference, dx)
            expected = reference.view(torch.uint16).cpu().numpy()
            with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
                for alias in (False, True):
                    guarded = cp.full(count + 2, np.uint16(0x42f6), cp.uint16)
                    output = guarded[1:-1].reshape(gate.shape)
                    dgate = cp.from_dlpack(gate.view(torch.uint16))
                    dup = cp.from_dlpack(up.view(torch.uint16))
                    if alias:
                        output[:] = dgate
                        dgate = output
                    kernel(((count + 255) // 256,), (256,),
                        (dgate, dup, output, np.uint64(count), np.float32(limit), np.uint32(clamp)))
                    actual = cp.asnumpy(output)
                    name = f'{label}_limit{limit}_clamp{clamp}_alias{int(alias)}'
                    arrays[name + '_native'], arrays[name + '_reference'] = actual, expected
                    checks[name] = {'pass': actual.tobytes() == expected.tobytes(),
                                    'mismatched_elements': int(np.sum(actual != expected))}
                    guards = cp.asnumpy(guarded)[[0, -1]]
                    checks[name + '_guards'] = {'pass': bool(np.all(guards == 0x42f6))}
            if label == 'saturation' and clamp:
                # A control must distinguish the old post-SiLU/one-round policy.
                g, u = gate.float(), up.float()
                wrong = (torch.nn.functional.silu(g).clamp(max=limit)
                         * u.clamp(-limit, limit)).bfloat16()
                mismatch = int((wrong.view(torch.uint16) != reference.view(torch.uint16)).sum().item())
                checks[f'old_policy_negative_control_limit{limit}'] = {'pass': mismatch > 0, 'mismatched_elements': mismatch}
    report = {'scope': 'installed_dense_activation_component_not_model_parity',
              'pass': all(item['pass'] for item in checks.values()), 'checks': checks,
              'input_sha256': sha(args.input_bin), 'sidecar_sha256': sha(args.input_sidecar),
              'native_sha256': sha(args.native_source), 'header_sha256': sha(args.native_source.parent / 'exl3_common.cuh'),
              'reference_binary_sha256': sha(compiled_ops.__file__), 'harness_sha256': sha(__file__),
              'limit': args.limit, 'alpha': 1., 'beta': 0., 'torch': torch.__version__,
              'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(), 'seed': 20261010}
    args.out.mkdir(parents=True, exist_ok=False)
    np.savez(args.out / 'stages.npz', **arrays)
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if report['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
