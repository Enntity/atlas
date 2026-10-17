# SPDX-License-Identifier: AGPL-3.0-only
"""Real installed GLM query/key quantizer differential; explicit CUDA prerequisites."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_e4m3_values():
    code = np.arange(127, dtype=np.uint8)
    exp, mantissa = (code >> 3).astype(np.int32), (code & 7).astype(np.float64)
    return np.where(exp == 0, mantissa * 2 ** -9,
                    (1 + mantissa / 8) * np.exp2(exp - 7)).astype(np.float32)


def scale_boundary_impulses(exponents):
    """Exact BF16 impulses at and around scale transitions AFTER normalized H128.

    One nonzero input has constant H magnitude. Account for F32 multiply and
    the reference's BF16 output rounding before selecting neighboring inputs.
    GPU results remain authoritative; these are explicit test fixtures.
    """
    if not exponents or any(not isinstance(e, int) or not -23 <= e <= 100 for e in exponents):
        raise ValueError('explicit finite scale exponents -23..100 required')
    norm = np.float32(1 / np.sqrt(128.))

    def round_bf16(value):
        bits = np.asarray(value, np.float32).view(np.uint32)
        rounded = bits + np.uint32(0x7fff) + ((bits >> 16) & 1)
        return (rounded & np.uint32(0xffff0000)).view(np.float32)

    vectors, metadata = [], []
    for exponent in exponents:
        boundary = np.float32(448. * 2. ** exponent)
        target = np.float32(boundary / norm)
        center = int(target.view(np.uint32) >> 16)
        candidates = (np.arange(center - 3, center + 4, dtype=np.uint32) << 16).view(np.float32)
        maxima = round_bf16(np.multiply(candidates, norm, dtype=np.float32))
        lower = np.flatnonzero(maxima < boundary)[-1]
        upper = np.flatnonzero(maxima > boundary)[0]
        amplitudes = candidates[[lower, upper]]
        offset = len(vectors)
        for value in (*amplitudes, *(-amplitudes)):
            x = np.zeros(128, np.float32)
            x[0] = value
            vectors.append(x)
        # Two adjacent BF16 impulses can reach a post-H BF16 maximum that a
        # single impulse skips. H128 outputs +/- (a+b) and +/- (a-b), so the
        # maximum depends on one F32 addition, normalization, then BF16 RN.
        half_center = int(np.float32(target / 2).view(np.uint32) >> 16)
        halves = (np.arange(half_center - 3, half_center + 4, dtype=np.uint32) << 16).view(np.float32)
        pair = next(((a, b) for a in halves for b in halves
                     if round_bf16(np.float32(np.float32(a + b) * norm)) == boundary), None)
        if pair is None:
            raise ValueError('no exact BF16 paired-impulse boundary fixture')
        for sign in (1, -1):
            x = np.zeros(128, np.float32)
            x[:2] = np.asarray(pair) * sign
            vectors.append(x)
        metadata.append({
            'exponent': exponent, 'boundary': float(boundary),
            'positive_rows': [offset, offset + 1], 'negative_rows': [offset + 2, offset + 3],
            'hadamard_maxima': maxima[[lower, upper]].tolist(),
            'equality_rows': [offset + 4, offset + 5],
            'equality_hadamard_maximum': float(boundary),
            'normalization': 'F32(1/sqrt(128)); normalized H then BF16 RN',
        })
    return np.stack(vectors), metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-source', type=Path, required=True)
    parser.add_argument('--reference-source', type=Path, required=True)
    parser.add_argument('--reference-sha256', required=True)
    parser.add_argument('--reference-attention-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if sha(args.reference_source) != args.reference_sha256:
        raise ValueError('installed query/key quantizer differs from frozen source')
    import torch
    import cupy as cp
    from vllm.models.glm5next.nvidia import attention as installed_attention
    if sha(installed_attention.__file__) != args.reference_attention_sha256:
        raise ValueError('installed compiled index leaves differ from frozen source')
    if not torch.cuda.is_available() or cp.cuda.runtime.getDeviceCount() != 1:
        raise RuntimeError('one visible real CUDA device required')
    spec = importlib.util.spec_from_file_location('installed_index_quant_reference', args.reference_source)
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)
    source = args.native_source.read_text().replace('#include <math.h>',
                    '#define INFINITY (__int_as_float(0x7f800000))')
    diagnostic_source = r"""
#include <cuda_fp8.h>
extern "C" __global__ void diagnostic_fp8_convert(
    const float* x, __nv_bfloat16* output, unsigned int count) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const __nv_fp8_e4m3 value(fminf(448.0f, fmaxf(-448.0f, x[i])));
    output[i] = __float2bfloat16_rn(static_cast<float>(value));
}
"""
    module = cp.RawModule(code=source + diagnostic_source, options=('--std=c++17', '--fmad=false', '-I' + str(args.native_source.parent)))
    functions = {name: module.get_function('glm_index_' + name) for name in
                 ('query_quant_bf16', 'kpool_finalize_quant_bf16', 'layernorm_f32_params_bf16',
                  'weight_gemv_f32', 'weight_scale_f32', 'logits_quant_bf16', 'logits_quant_bf16_dynamic')}
    rng = np.random.default_rng(20261010)
    def bf16(x):
        return torch.from_numpy(np.asarray(x, np.float32)).cuda().bfloat16()
    def carray(tensor):
        return cp.from_dlpack(tensor.view(torch.uint16) if tensor.dtype == torch.bfloat16 else tensor)
    def unpack(values):
        return (cp.asnumpy(values).astype(np.uint32) << 16).view(np.float32)
    arrays, report = {}, {'scope': 'installed_component_oracle_not_model_parity', 'checks': {}}
    def exact(name, actual, expected):
        arrays[name + '_native'], arrays[name + '_reference'] = actual, expected
        same = actual.shape == expected.shape and actual.dtype == expected.dtype and actual.tobytes() == expected.tobytes()
        report['checks'][name] = {'pass': same, 'exact_bytes': same, 'elements': int(actual.size)}
    def close(name, actual, expected, absolute, relative):
        arrays[name + '_native'], arrays[name + '_reference'] = actual, expected
        report['checks'][name] = {'pass': bool(np.allclose(actual, expected, atol=absolute, rtol=relative)),
                                 'absolute': absolute, 'relative': relative,
                                 'max_abs': float(np.max(np.abs(actual[np.isfinite(expected)] - expected[np.isfinite(expected)]))) if np.isfinite(expected).any() else 0.0}
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        values = finite_e4m3_values()
        midpoint = (values[:-1] + values[1:]) * np.float32(.5)
        positive = np.concatenate([values, midpoint, [432., 448., 464., 480.] ]).astype(np.float32)
        conversion = np.concatenate([positive, -positive])
        dconversion = cp.asarray(conversion)
        cout = cp.empty(len(conversion), cp.uint16)
        module.get_function('diagnostic_fp8_convert')(((len(conversion) + 255) // 256,), (256,),
                                  (dconversion, cout, np.uint32(len(conversion))))
        expected_conversion = torch.from_numpy(conversion).cuda().clamp(-448, 448).to(torch.float8_e4m3fn).float().cpu().numpy()
        exact('fp8_conversion_ties_clamp', unpack(cout), expected_conversion)
        query_cases = []
        for rows in (1, 7, 96):
            x = rng.normal(size=(rows, 128)).astype(np.float32)
            x[0] = 0
            if rows > 1:
                x[1] *= 1e-7
                x[2] *= 65536
                x[3] = np.tile([0., -0., .5, -.5], 32)
            query_cases.append(x)
        boundary_x, boundary_metadata = scale_boundary_impulses((-23, -22, -21, -1, 0, 1, 10, 100))
        query_cases.append(boundary_x)
        report['scale_boundary_inputs'] = boundary_metadata
        for x in query_cases:
            rows = len(x)
            dx = bf16(x)
            arrays['query_input_' + str(rows)] = dx.float().cpu().numpy()
            quant, scales = reference.fwht128_quant_fp8(dx)
            out, coutscale = cp.empty((rows, 128), cp.uint16), cp.empty(rows, cp.float32)
            functions['query_quant_bf16']((rows,), (128,),
                    (carray(dx), out, coutscale, np.uint32(rows)))
            exact('query_' + str(rows), unpack(out), quant.float().cpu().numpy())
            exact('query_scale_' + str(rows), cp.asnumpy(coutscale), scales.flatten().cpu().numpy())
            exact('query_dequant_' + str(rows), unpack(out) * cp.asnumpy(coutscale)[:, None],
                  (quant.float() * scales).bfloat16().float().cpu().numpy())
            alias_guard = cp.full(rows * 128 + 2, np.uint16(0x42f6), cp.uint16)
            alias = alias_guard[1:-1].reshape(rows, 128)
            alias[:] = carray(dx)
            scale_guard = cp.full(rows + 2, -999, cp.float32)
            functions['query_quant_bf16']((rows,), (128,),
                    (alias, alias, scale_guard[1:-1], np.uint32(rows)))
            exact('query_alias_' + str(rows), cp.asnumpy(alias), cp.asnumpy(out))
            exact('query_alias_scale_' + str(rows), cp.asnumpy(scale_guard[1:-1]), cp.asnumpy(coutscale))
            exact('query_alias_guard_' + str(rows), cp.asnumpy(alias_guard)[[0, -1]], np.full(2, 0x42f6, np.uint16))
            exact('query_scale_guard_' + str(rows), cp.asnumpy(scale_guard)[[0, -1]], np.full(2, -999, np.float32))
        pools, block, dim = 9, 8, 128
        keys = bf16(rng.normal(size=(pools, 4, dim)))
        gates = bf16(rng.normal(size=(pools, 4, dim)))
        keys[0].zero_()
        ape = torch.from_numpy(rng.normal(size=(4, dim)).astype(np.float32)).cuda()
        dummy_cache = torch.empty((1, 8, 132), dtype=torch.uint8, device='cuda')
        loc = torch.arange(pools, dtype=torch.int64, device='cuda')
        quant, scales = reference.kpool_compress_and_write_cache(
                dummy_cache, keys, gates, ape, loc, 4, head_dim=dim,
                round_scale=True, return_compressed=True, write_cache=False)
        blocks = (pools + 1) // 2
        mapping = np.arange(blocks - 1, -1, -1, dtype=np.uint32)
        tail = torch.zeros((blocks, 2, block, dim), dtype=torch.bfloat16, device='cuda')
        for p in range(pools):
            physical, start = p // 2, p % 2 * 4
            tail[int(mapping[physical]), 0, start:start + 4] = keys[p]
            tail[int(mapping[physical]), 1, start:start + 4] = gates[p]
        slots = cp.asarray(np.arange(pools, dtype=np.int64) * 4 + 3)
        cache = cp.full((blocks, block // 4, dim), np.uint16(0x42f6), cp.uint16)
        functions['kpool_finalize_quant_bf16']((pools,), (128,),
                (carray(tail), cp.asarray(mapping), carray(ape), cache, slots,
                 *map(np.uint32, (pools, block, 4, dim)),
                 np.uint64(2 * block * dim * 2), np.uint64(block // 4 * dim * 2)))
        arrays['key_input'] = keys.float().cpu().numpy()
        arrays['pool_gate_input'] = gates.float().cpu().numpy()
        arrays['ape_f32'] = ape.cpu().numpy()
        expected = (quant.float() * scales[:, None]).bfloat16().float().cpu().numpy()
        exact('key_paged', unpack(cache).reshape(-1, dim)[:pools], expected)
        exact('key_guard', cp.asnumpy(cache).reshape(-1, dim)[pools:], np.full((1, dim), 0x42f6, np.uint16))
        rows, heads, hidden = 3, 32, 256
        h, w = bf16(rng.normal(size=(rows, hidden))), bf16(rng.normal(size=(heads, hidden)))
        dweights = cp.empty((rows, heads), cp.float32)
        functions['weight_gemv_f32']((1, rows), (128,),
                (carray(h), carray(w), dweights, *map(np.uint32, (rows, heads, hidden))))
        expected_weights = (h.float() @ w.float().T).cpu().numpy()
        close('gate_unscaled', cp.asnumpy(dweights), expected_weights, 1e-4, 2e-5)
        arrays['gate_hidden_input'], arrays['gate_weight_input'] = h.float().cpu().numpy(), w.float().cpu().numpy()
        query_raw, qscale = reference.fwht128_quant_fp8(bf16(rng.normal(size=(rows * heads, 128))))
        norm = np.float32(128 ** -.5 * heads ** -.5)
        functions['weight_scale_f32']((1,), (256,),
                (dweights, carray(qscale), np.uint64(rows * heads), norm))
        expected_scaled = installed_attention._fused_indexer_weight_scale(
                torch.from_numpy(expected_weights).cuda(), qscale.reshape(rows, heads, 1), float(norm)).cpu().numpy()
        close('gate_scaled', cp.asnumpy(dweights), expected_scaled, 1e-5, 2e-5)
        # This independently checks the representations/scaling, not DeepGEMM order.
        dqraw = query_raw.bfloat16().reshape(rows, heads, 128)
        key_values = (quant.float() * scales[:, None]).bfloat16()
        dots = torch.einsum('rhd,pd->rhp', dqraw.float(), key_values.float()).relu()
        score_reference = (dots * torch.from_numpy(expected_scaled).cuda()[:, :, None]).sum(1).cpu().numpy()
        stride = blocks * 2
        logits = cp.empty((rows, stride), cp.float32)
        table = cp.arange(blocks, dtype=cp.uint32)
        functions['logits_quant_bf16'](((stride + 7) // 8, rows), (256,),
            (carray(dqraw), dweights, cache, logits, table,
             *map(np.uint32, (rows, 31, stride, heads, 128, 4, block)), np.uint64(block // 4 * 128 * 2)))
        mathematical = np.full((rows, stride), -np.inf, np.float32)
        for row in range(rows):
            live = (31 + row + 1) // 4
            mathematical[row, :live] = score_reference[row, :live]
        close('scorer_math_not_deepgemm', cp.asnumpy(logits), mathematical, 1e-4, 2e-5)
        dynamic = cp.empty(stride, cp.float32)
        dlength = cp.empty(1, cp.uint32)
        for length in (37, 31, 41, 32):
            dlength.fill(length)
            functions['logits_quant_bf16_dynamic'](((stride + 7) // 8,), (256,),
                (carray(dqraw), dweights, cache, dynamic, table, np.uint32(1), dlength,
                 *map(np.uint32, (stride, heads, 128, 4, block)), np.uint64(block // 4 * 128 * 2), np.uint32(31)))
            mathematical = np.full(stride, -np.inf, np.float32)
            if 31 < length <= stride * 4:
                mathematical[:length // 4] = score_reference[0, :length // 4]
            close('dynamic_scorer_' + str(length), cp.asnumpy(dynamic), mathematical, 1e-4, 2e-5)

        normx = bf16(rng.normal(size=(7, 128)))
        nweight = torch.from_numpy(rng.normal(size=128).astype(np.float32)).cuda()
        nbias = torch.from_numpy(rng.normal(size=128).astype(np.float32)).cuda()
        arrays['norm_input'], arrays['norm_weight_f32'], arrays['norm_bias_f32'] = normx.float().cpu().numpy(), nweight.cpu().numpy(), nbias.cpu().numpy()
        expected_norm = installed_attention._fused_indexer_k_norm(normx, nweight, nbias, 128, 1e-5).float().cpu().numpy()
        functions['layernorm_f32_params_bf16']((7,), (128,),
                (carray(normx), carray(nweight), carray(nbias), np.uint32(7), np.uint32(128), np.float32(1e-5)))
        actual_norm = normx.float().cpu().numpy()
        close('norm', actual_norm, expected_norm, .015625, .008)
    report.update({'pass': all(c['pass'] for c in report['checks'].values()), 'native_source_sha256': sha(args.native_source),
                   'native_header_sha256': {name: sha(args.native_source.parent / name)
                       for name in ('exl3_common.cuh', 'glm_indexer_score.cuh')},
                   'compiler_options': ['--std=c++17', '--fmad=false'],
                   'compiled_source_sha256': hashlib.sha256((source + diagnostic_source).encode()).hexdigest(),
                   'attention_source_sha256': sha(installed_attention.__file__),
                   'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                   'float32_matmul_precision': torch.get_float32_matmul_precision(),
                   'leaf_compile_backend': str(installed_attention._INDEXER_COMPILE['backend']),
                   'synthetic_seed': 20261010,
                   'reference_source_sha256': sha(args.reference_source), 'harness_sha256': sha(__file__),
                   'torch': torch.__version__, 'cupy': cp.__version__, 'device': torch.cuda.get_device_name()})
    args.out.mkdir(parents=True, exist_ok=False)
    np.savez(args.out / 'arrays.npz', **arrays)
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if report['pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
