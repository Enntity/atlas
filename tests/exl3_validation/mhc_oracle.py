# SPDX-License-Identifier: AGPL-3.0-only
"""Capture installed mHC versus Atlas's existing pre-mix/standalone norm.

This records numerical differences without a parity pass/fail threshold. It
checks replay identity separately; it cannot qualify a full model.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_fused_geometry(rows, hidden, fx, residual, post, comb, fn, scale, base, weight):
    expected = ((rows, hidden), (rows, 4, hidden), (rows, 4, 1),
                (rows, 4, 4), (24, 4 * hidden), (3,), (24,), (hidden,))
    if rows <= 0 or hidden <= 0 or hidden % 128:
        raise ValueError('positive aligned mHC geometry required')
    if tuple(tuple(v.shape) for v in (fx, residual, post, comb, fn, scale, base, weight)) != expected:
        raise ValueError('fused mHC geometry differs from registered pre replay')


def native_mhc_compiler_options(fmad):
    if fmad not in ('true', 'false'):
        raise ValueError('Explicit true/false native mHC FMA policy required')
    return ('--std=c++17', '--fmad=' + fmad)


def rounded_split_count(rows,width,sms):
    if rows<=0 or width<=0 or width%64 or sms<=0:
        raise ValueError('positive explicit SMS/rows and aligned width required')
    return max(1,min(sms//((rows+63)//64),(width//64+3)//4))


def qualification_pass(replay_checks, corrected_checks):
    return all(check['byte_exact'] for check in replay_checks.values()) and all(
        check['byte_exact'] for check in corrected_checks.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-prefix', type=Path, required=True)
    parser.add_argument('--native-hc-source', type=Path, required=True)
    parser.add_argument('--native-norm-source', type=Path, required=True)
    parser.add_argument('--native-pdl-header', type=Path, required=True)
    parser.add_argument('--native-mhc-source', type=Path)
    parser.add_argument('--native-mhc-fmad', choices=('true', 'false'))
    parser.add_argument('--native-mhc-sms', type=int)
    parser.add_argument('--fused-capture-prefix', type=Path)
    parser.add_argument('--derived-fused-rows', type=int, nargs='+')
    parser.add_argument('--reference-source-sha256', required=True)
    parser.add_argument('--reference-kernels-sha256', required=True)
    parser.add_argument('--capture-manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if (args.native_mhc_source is None) != (args.native_mhc_fmad is None):
        parser.error('--native-mhc-source and explicit --native-mhc-fmad are required together')
    if (args.native_mhc_source is None)!=(args.native_mhc_sms is None) or (args.native_mhc_sms is not None and args.native_mhc_sms<=0):
        parser.error('positive explicit --native-mhc-sms required with native source')
    if args.derived_fused_rows is not None:
        if (args.fused_capture_prefix is None
                or len(set(args.derived_fused_rows)) != len(args.derived_fused_rows)
                or any(n <= 0 or n > 128 for n in args.derived_fused_rows)):
            raise ValueError('derived mHC cases require explicit unique row counts in 1..128 and a frozen fused input')
    frozen = json.loads(args.capture_manifest.read_text())
    import torch
    import cupy as cp
    from vllm.model_executor.kernels.mhc import tilelang as reference
    from vllm.model_executor.kernels.mhc import tilelang_kernels
    if sha(reference.__file__) != args.reference_source_sha256 or sha(tilelang_kernels.__file__) != args.reference_kernels_sha256:
        raise ValueError('installed mHC source differs from frozen source')
    if not torch.cuda.is_available() or cp.cuda.runtime.getDeviceCount() != 1:
        raise RuntimeError('one visible CUDA device required')
    files = {}
    def load(name, prefix=None):
        sidecar = Path(str(args.capture_prefix if prefix is None else prefix) + '-' + name + '.json')
        metadata = json.loads(sidecar.read_text())
        binary = sidecar.parent / metadata['file']
        for path in (sidecar, binary):
            actual = sha(path)
            if frozen.get(path.name) != actual:
                raise ValueError('capture bytes differ from registered manifest')
            files[path.name] = actual
        dtype = metadata['dtype']
        raw = np.fromfile(binary, {'bf16-le': '<u2', 'f32-le': '<f4'}[dtype]).reshape(metadata['shape'])
        if dtype == 'bf16-le':
            value = torch.from_numpy(raw.copy()).view(torch.bfloat16).cuda()
        else:
            value = torch.from_numpy(raw.copy()).cuda()
        if not torch.isfinite(value).all():
            raise ValueError('finite mHC input required')
        return value
    policy_path = Path(str(args.capture_prefix) + '-policy.json')
    if frozen.get(policy_path.name) != sha(policy_path):
        raise ValueError('mHC policy differs from frozen manifest')
    files[policy_path.name] = sha(policy_path)
    policy = json.loads(policy_path.read_text())['policy']
    x, fn, scale, base, weight = [load('input-' + name) for name in ('x', 'hc_fn', 'hc_scale', 'hc_base', 'norm_weight')]
    expected = [load(name) for name in ('post', 'comb', 'layer-input')]
    rows, streams, hidden = x.shape
    if streams != 4 or fn.shape != (24, streams * hidden) or weight.shape != (hidden,) or policy['hc_post_mult_value'] != 2.:
        raise ValueError('unsupported native mHC geometry/policy')
    replay = reference.mhc_pre_tilelang(x, fn, scale, base, policy['rms_eps'],
        policy['hc_pre_eps'], policy['hc_sinkhorn_eps'], policy['hc_post_mult_value'],
        policy['sinkhorn_repeat'], n_splits=1, norm_weight=weight, norm_eps=policy['norm_eps'])
    header = args.native_pdl_header.read_text()
    hc_source = args.native_hc_source.read_text().replace('#include "../../common/atlas_pdl.cuh"', header)
    norm_source = args.native_norm_source.read_text().replace('#include "atlas_pdl.cuh"', header)
    hc = cp.RawModule(code=hc_source, options=('--std=c++17', '--fmad=false'))
    norm = cp.RawModule(code=norm_source, options=('--std=c++17', '--fmad=false'))
    new_mhc = None
    if args.native_mhc_source is not None:
        new_mhc = cp.RawModule(code=args.native_mhc_source.read_text(),
            options=(*native_mhc_compiler_options(args.native_mhc_fmad), '-I' + str(args.native_mhc_source.parent)))
        new_pre = new_mhc.get_function('exl3_mhc_pre_norm')
    def carray(value):
        return cp.from_dlpack(value.view(torch.uint16) if value.dtype == torch.bfloat16 else value)
    def cpu(value):
        return value.detach().float().cpu().numpy() if isinstance(value, torch.Tensor) else cp.asnumpy(value)
    def bfvalues(value):
        return (cp.asnumpy(value).astype(np.uint32) << 16).view(np.float32)
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        highway = carray(x.float())
        collapsed, normalized = cp.empty((rows, hidden), cp.uint16), cp.empty((rows, hidden), cp.uint16)
        post, comb = cp.empty((rows, 4), cp.float32), cp.empty((rows, 4, 4), cp.float32)
        hc.get_function('hc_pre')((rows,), (256,),
            (highway, carray(fn), carray(scale), carray(base), collapsed, post, comb,
             np.uint32(hidden), np.uint32(streams), np.uint32(policy['sinkhorn_repeat']),
             np.float32(policy['rms_eps']), np.float32(policy['hc_pre_eps'])))
        norm.get_function('rms_norm_vanilla')((rows,), (min(hidden, 1024),),
            (collapsed, carray(weight), normalized, np.uint32(hidden), np.float32(policy['norm_eps'])))
        actual = [cp.asnumpy(post), cp.asnumpy(comb), bfvalues(normalized)]
        corrected = None
        if new_mhc is not None:
            new_pre((rows,), (256,),
                (highway, carray(fn), carray(scale), carray(base), carray(weight), normalized, post, comb,
                 np.uint32(hidden), np.uint32(policy['sinkhorn_repeat']), np.float32(policy['rms_eps']),
                 np.float32(policy['norm_eps']), np.float32(policy['hc_pre_eps']),
                 np.float32(policy['hc_post_mult_value']), np.uint32(0), np.uint32(rounded_split_count(rows,hidden*4,args.native_mhc_sms))))
            corrected = [cp.asnumpy(post), cp.asnumpy(comb), bfvalues(normalized)]
    arrays, differences, replay_checks = {}, {}, {}
    for name, native, ref, captured in zip(('post', 'comb', 'normalized'), actual, replay, expected):
        ref, captured = cpu(ref).reshape(native.shape), cpu(captured).reshape(native.shape)
        arrays[name + '_native'], arrays[name + '_reference'], arrays[name + '_captured'] = native, ref, captured
        differences[name] = {'max_abs': float(np.max(np.abs(native - ref))),
                             'rms': float(np.sqrt(np.mean((native.astype(np.float64) - ref) ** 2))),
                             'equal_elements': int(np.sum(native == ref)), 'elements': int(native.size)}
        replay_checks[name] = {'byte_exact': ref.tobytes() == captured.tobytes(),
                              'max_abs': float(np.max(np.abs(ref - captured)))}
    corrected_checks = {}
    if corrected is not None:
        for name, native, ref in zip(('post', 'comb', 'normalized'), corrected, replay):
            ref = cpu(ref).reshape(native.shape)
            arrays[name + '_corrected'] = native
            corrected_checks[name] = {'byte_exact': native.tobytes() == ref.tobytes(),
                'max_abs': float(np.max(np.abs(native - ref))),
                'mismatched_elements': int(np.sum(native != ref))}
    if args.fused_capture_prefix is not None:
        if new_mhc is None:
            raise ValueError('fused replay requires explicit native mHC source')
        new_post = new_mhc.get_function('exl3_mhc_post_unrounded')
        new_contract = new_mhc.get_function('exl3_mhc_contract')
        prefix = args.fused_capture_prefix
        fused_policy_path = Path(str(prefix) + '-policy.json')
        if frozen.get(fused_policy_path.name) != sha(fused_policy_path):
            raise ValueError('fused mHC policy differs from frozen manifest')
        files[fused_policy_path.name] = sha(fused_policy_path)
        fpolicy = json.loads(fused_policy_path.read_text())['policy']
        fx, residual, fpost, fcomb, ffn, fscale, fbase, fweight = [load('input-' + name, prefix)
            for name in ('x', 'residual', 'post', 'comb', 'hc_fn', 'hc_scale', 'hc_base', 'norm_weight')]
        fexpected = [load(name, prefix) for name in ('residual-output', 'post', 'comb', 'layer-input')]
        validate_fused_geometry(rows, hidden, fx, residual, fpost, fcomb, ffn, fscale, fbase, fweight)
        freplay = reference.mhc_fused_post_pre_tilelang(fx, residual, fpost, fcomb, ffn, fscale, fbase,
            fpolicy['rms_eps'], fpolicy['hc_pre_eps'], fpolicy['hc_sinkhorn_eps'],
            fpolicy['hc_post_mult_value'], fpolicy['sinkhorn_repeat'], n_splits=1, tile_n=1,
            norm_weight=fweight, norm_eps=fpolicy['norm_eps'])
        frows = rows
        with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
            fhighway = carray(residual.float())
            fnormed = cp.empty((frows, hidden), cp.uint16)
            new_post((frows,), (256,), (carray(fx), fhighway, carray(fpost), carray(fcomb), fhighway, np.uint32(hidden)))
            contracted = cp.empty((frows, hidden), cp.uint16)
            new_contract((frows,), (256,), (fhighway, contracted, np.uint32(hidden)))
            factual_contract = bfvalues(contracted)
            new_pre((frows,), (256,), (fhighway, carray(ffn), carray(fscale), carray(fbase), carray(fweight),
                fnormed, post, comb, np.uint32(hidden), np.uint32(fpolicy['sinkhorn_repeat']),
                np.float32(fpolicy['rms_eps']), np.float32(fpolicy['norm_eps']),
                np.float32(fpolicy['hc_pre_eps']), np.float32(fpolicy['hc_post_mult_value']), np.uint32(frows <= 16), np.uint32(rounded_split_count(frows,hidden*4,args.native_mhc_sms))))
            factual = [cp.asnumpy(fhighway), cp.asnumpy(post), cp.asnumpy(comb), bfvalues(fnormed)]
        reference_contract = cpu(freplay[0].mean(dim=1))
        arrays['fused_contract_native'] = factual_contract
        arrays['fused_contract_reference'] = reference_contract
        corrected_checks['fused_contract'] = {
            'byte_exact': factual_contract.tobytes() == reference_contract.tobytes(),
            'max_abs': float(np.max(np.abs(factual_contract - reference_contract))),
            'mismatched_elements': int(np.sum(factual_contract != reference_contract))}
        for name, native, ref, captured in zip(('residual', 'post', 'comb', 'normalized'), factual, freplay, fexpected):
            ref, captured = cpu(ref).reshape(native.shape), cpu(captured).reshape(native.shape)
            arrays['fused_' + name + '_native'], arrays['fused_' + name + '_reference'] = native, ref
            replay_checks['fused_' + name] = {'byte_exact': ref.tobytes() == captured.tobytes(),
                'max_abs': float(np.max(np.abs(ref - captured)))}
            corrected_checks['fused_' + name] = {'byte_exact': native.tobytes() == ref.tobytes(),
                'max_abs': float(np.max(np.abs(native - ref))), 'mismatched_elements': int(np.sum(native != ref))}
        # These cases change batch geometry by cyclically selecting frozen input
        # rows. They are component controls, not new captured model requests.
        for n in args.derived_fused_rows or ():
            indices = torch.arange(n, device=residual.device) % frows
            dx, dr, dp, dc = [v.index_select(0, indices) for v in (fx, residual, fpost, fcomb)]
            derived_reference = reference.mhc_fused_post_pre_tilelang(dx, dr, dp, dc, ffn, fscale, fbase,
                fpolicy['rms_eps'], fpolicy['hc_pre_eps'], fpolicy['hc_sinkhorn_eps'],
                fpolicy['hc_post_mult_value'], fpolicy['sinkhorn_repeat'], n_splits=1, tile_n=1,
                norm_weight=fweight, norm_eps=fpolicy['norm_eps'])
            with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
                dhighway = carray(dr.float())
                dnorm, dcontract = [cp.empty((n, hidden), cp.uint16) for _ in range(2)]
                dpost, dcomb = cp.empty((n, 4), cp.float32), cp.empty((n, 4, 4), cp.float32)
                new_post((n,), (256,), (carray(dx), dhighway, carray(dp), carray(dc), dhighway, np.uint32(hidden)))
                new_contract((n,), (256,), (dhighway, dcontract, np.uint32(hidden)))
                contracted_values = bfvalues(dcontract)
                new_pre((n,), (256,), (dhighway, carray(ffn), carray(fscale), carray(fbase), carray(fweight),
                    dnorm, dpost, dcomb, np.uint32(hidden), np.uint32(fpolicy['sinkhorn_repeat']),
                    np.float32(fpolicy['rms_eps']), np.float32(fpolicy['norm_eps']),
                    np.float32(fpolicy['hc_pre_eps']), np.float32(fpolicy['hc_post_mult_value']), np.uint32(n <= 16), np.uint32(rounded_split_count(n,hidden*4,args.native_mhc_sms))))
                actual = [cp.asnumpy(dhighway), cp.asnumpy(dpost), cp.asnumpy(dcomb), bfvalues(dnorm), contracted_values]
            targets = [*derived_reference, derived_reference[0].mean(dim=1)]
            for name, native, target in zip(('residual', 'post', 'comb', 'normalized', 'contract'), actual, targets):
                target = cpu(target).reshape(native.shape)
                label = 'derived_rows_' + str(n) + '_' + name
                arrays[label + '_native'], arrays[label + '_reference'] = native, target
                corrected_checks[label] = {'byte_exact': native.tobytes() == target.tobytes(),
                    'max_abs': float(np.max(np.abs(native - target))),
                    'mismatched_elements': int(np.sum(native != target))}
    report = {'corrected_checks': corrected_checks, 'scope': 'same-input mHC boundary diagnostic; no full-model parity claim',
              'differences': differences, 'installed_replay_identity': replay_checks,
              'capture_files': files, 'policy': policy, 'derived_fused_rows': args.derived_fused_rows, 'native_mhc_explicit_sms':args.native_mhc_sms,
              'rounded_split_counts': None if args.native_mhc_sms is None else {str(n):rounded_split_count(n,hidden*4,args.native_mhc_sms) for n in [rows,*(args.derived_fused_rows or [])]},
              'qualification_gate': 'all installed replay and corrected post/comb/normalized/residual/contract tensors byte exact',
              'derived_case_scope': 'cyclic frozen row selection; installed operator replay; no captured model parity claim',
              'sources': {str(p): sha(p) for p in (args.native_hc_source, args.native_norm_source,
                           args.native_pdl_header, Path(reference.__file__), Path(tilelang_kernels.__file__), Path(__file__))},
              'torch': torch.__version__, 'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(),
              'compiler_options': ['--std=c++17', '--fmad=false'],
              'note': 'Native stages jointly differ in mix arithmetic, Sinkhorn and normalization; no isolated causal attribution.'}
    if args.native_mhc_source is not None:
        report['sources'][str(args.native_mhc_source)] = sha(args.native_mhc_source)
        report['native_mhc_compiler_options'] = list(native_mhc_compiler_options(args.native_mhc_fmad))
    args.out.mkdir(parents=True, exist_ok=False)
    np.savez(args.out / 'stages.npz', **arrays)
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if qualification_pass(replay_checks, corrected_checks) else 1


if __name__ == '__main__':
    raise SystemExit(main())
