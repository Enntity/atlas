# SPDX-License-Identifier: AGPL-3.0-only
"""Real installed thin/grouped-fat boundary replay for one checkpoint expert."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
from differential_expert import digest, load_parts, metrics
from moe_precision import run_expert


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--expert-prefix', required=True)
    parser.add_argument('--reference-plugin', type=Path, required=True)
    parser.add_argument('--reference-plugin-sha256', required=True)
    parser.add_argument('--reference-fat-extension-sha256', required=True)
    parser.add_argument('--rank', type=int, choices=(0, 1), required=True)
    parser.add_argument('--thin-cap', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--weight', type=float, required=True)
    parser.add_argument('--limit', type=float, required=True)
    parser.add_argument('--abs', dest='absolute', type=float, required=True)
    parser.add_argument('--rel', dest='relative', type=float, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--native-mma-profile', nargs=4, type=int,
                        metavar=('TILE_K', 'TILE_N', 'SLICES', 'HALF_PARTIAL'))
    parser.add_argument('--experimental-fat-mma-profile', nargs=4, type=int,
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
    if args.experimental_fat_mma_profile is not None and tuple(args.experimental_fat_mma_profile) not in ((16,128,1,0), (32,128,1,0)):
        parser.error('experimental fat profile16/128/1/0 or its K32 control required')
    if not 1 <= args.thin_cap < 128:
        raise ValueError('explicit bounded thin capacity required')
    if not np.isfinite([args.weight, args.limit, args.absolute, args.relative]).all() or min(args.limit, args.absolute, args.relative) < 0:
        raise ValueError('finite numerical policy with nonnegative limits/tolerances required')
    if digest(args.reference_plugin) != args.reference_plugin_sha256:
        raise ValueError('reference plugin differs from pinned source')
    import torch
    import cupy as cp
    import exllamav3_ext
    if not torch.cuda.is_available() or cp.cuda.runtime.getDeviceCount() != 1:
        raise RuntimeError('one visible real CUDA device required')
    spec = importlib.util.spec_from_file_location('installed_expert_boundary_reference', args.reference_plugin)
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)
    fat_extension = reference.load_fat_moe_ext()
    if fat_extension is None or int(fat_extension.exl3_fat_moe_abi()) != 2 or digest(fat_extension.__file__) != args.reference_fat_extension_sha256:
        raise ValueError('pinned ABI2 grouped reference extension required')
    if not reference.grouped_fat_enabled() or reference.fused_moe_row_tile_enabled():
        raise ValueError('installed reference must explicitly select grouped-fat, without row tiling')
    parts, hashes, index_path = load_parts(args.checkpoint, args.expert_prefix, args.rank, reference)
    hidden, intermediate = parts['gate']['suh'].numel(), parts['up']['svh'].numel()
    inners = [{name: reference.make_linear_exl3(p['trellis'], p['suh'], p['svh'], p['mul1'])
               for name, p in parts.items()}]
    layer = SimpleNamespace(w13_trellis=parts['gate']['trellis'],
        w13_suh=parts['gate']['suh'], w13_svh=parts['gate']['svh'],
        w2_trellis=parts['down']['trellis'], w2_suh=parts['down']['suh'], w2_svh=parts['down']['svh'],
        _exl3_hidden_size=hidden,
        _exl3_intermediate_local=intermediate, _exl3_inners=inners,
        _exl3_shared_w13_suh=torch.equal(parts['gate']['suh'], parts['up']['suh']))
    reference.build_exl3_fused_state(layer, inners)
    if args.thin_cap != int(layer._exl3_fused_temps[0].shape[1]):
        raise ValueError('thin cap differs from actual allocated reference workspace')
    reference._record_exl3_fat_resolution(layer)
    if layer._exl3_fat_effective_tier != 'grouped':
        raise ValueError('selected real checkpoint expert is not eligible for actual grouped-fat reference')
    common = Path(__file__).resolve().parents[2] / 'kernels/gb10/common'
    source = (common / 'exl3_moe_precision.cu').read_text().replace('#include <math.h>', '')
    module = cp.RawModule(code=source, options=('--std=c++17', '--fmad=false', '-I' + str(common)))
    mma_module = None if args.native_mma_profile is None else cp.RawModule(
        code=(common / 'exl3_moe_mma.cu').read_text() + '\n// header-hashes ' +
                 ' '.join(digest(common / name) for name in ('exl3_mma.cuh', 'exl3_common.cuh', 'exl3_expert.cuh')),
        options=('--std=c++17', '--fmad=false', '-I' + str(common)))
    probe_source = Path(__file__).with_name('mma_probe.cu')
    fat_module = None if args.experimental_fat_mma_profile is None else cp.RawModule(
        code=probe_source.read_text() + '\n// header-hashes ' +
             ' '.join(digest(common / name) for name in ('exl3_mma.cuh', 'exl3_common.cuh')),
        options=('--std=c++17', '--fmad=false', '-I' + str(common)))
    rng = np.random.default_rng(args.seed)
    input_values = rng.normal(size=(args.thin_cap + 1, hidden)).astype(np.float32)
    arrays, cases = {}, {}
    for rows in (args.thin_cap, args.thin_cap + 1):
        dx = torch.from_numpy(input_values[:rows]).cuda().bfloat16()
        ids = torch.zeros((rows, 1), dtype=torch.int64, device='cuda')
        weights = torch.full((rows, 1), args.weight, dtype=torch.float32, device='cuda')
        sorted_to_input = reference.map_topk_to_local(ids, 1, None).argsort()
        input_to_sorted = sorted_to_input.argsort()
        output = reference.apply_exl3_fused_moe(dx, ids, weights, layer, inners, None, args.limit)
        torch.cuda.synchronize()
        path = layer._exl3_last_fat_reason
        if path != ('no_fat_experts' if rows <= args.thin_cap else 'grouped_ok'):
            raise ValueError('reference did not engage the prescribed occupancy path')
        if rows <= args.thin_cap:
            up_xh = layer._exl3_fused_temps[1][0, :rows, :hidden]
            down_xh = layer._exl3_fused_temps[2][0, :rows, :intermediate]
        else:
            scratch = reference._grouped_scratch(dx.device, rows, hidden, intermediate)
            up_xh = scratch['h13'] if layer._exl3_shared_w13_suh else scratch['h13u']
            up_xh, down_xh = up_xh[:rows], scratch['h2'][:rows]
        expected = {'up_xh': up_xh.index_select(0, input_to_sorted).float().cpu().numpy(),
                    'down_xh': down_xh.index_select(0, input_to_sorted).float().cpu().numpy(),
                    'weighted_f32': output.float().cpu().numpy(), 'weighted_bf16': output.bfloat16().float().cpu().numpy()}
        native = run_expert(module, dx, parts, args.weight, args.thin_cap, args.limit,
                            mma_module, args.native_mma_profile, fat_module, args.experimental_fat_mma_profile, args.native_fat_mma_profile)
        comparisons = {}
        arrays[f'rows{rows}_input'] = dx.float().cpu().numpy()
        arrays[f'rows{rows}_reference_sorted_to_input_row'] = sorted_to_input.cpu().numpy()
        for stage, expected_value in expected.items():
            arrays[f'rows{rows}_reference_{stage}'] = expected_value
            arrays[f'rows{rows}_native_{stage}'] = native[stage]
            comparisons[stage] = metrics(expected_value, native[stage], args.absolute, args.relative)
        cases[str(rows)] = {'reference_path': path, 'occupancy': rows, 'stages': comparisons}
    report = {'scope': 'single_expert_thin_fat_boundary_component_not_model_parity',
        'pass': all(s['violations'] == 0 for c in cases.values() for s in c['stages'].values()),
        'mma_profile': args.native_mma_profile, 'fat_mma_profile': args.native_fat_mma_profile, 'experimental_fat_mma_profile': args.experimental_fat_mma_profile,
        'fat_probe_source_sha256': digest(probe_source),
        'cases': cases, 'thin_cap': args.thin_cap, 'seed': args.seed, 'rank': args.rank, 'tp': 2,
        'weight_f32': args.weight, 'limit': args.limit, 'abs': args.absolute, 'rel': args.relative,
        'expert_prefix': args.expert_prefix, 'selected_tensor_sha256': hashes,
        'checkpoint_index_sha256': digest(index_path), 'reference_plugin_sha256': digest(args.reference_plugin),
        'fat_extension_sha256': digest(fat_extension.__file__), 'fat_extension_abi': 2,
        'thin_extension_sha256': digest(exllamav3_ext.__file__), 'harness_sha256': digest(__file__),
        'native_source_sha256': {name: digest(common / name) for name in ('exl3_moe_precision.cu', 'exl3_common.cuh', 'exl3_expert.cuh', 'exl3_moe_mma.cu', 'exl3_mma.cuh')},
        'native_runner_sha256': digest(Path(__file__).with_name('moe_precision.py')),
        'reference_profile': {k: os.environ.get(k) for k in ('EXL3_TEMP_ROWS_FUSED', 'EXL3_FAT_GROUPED', 'EXL3_FAT_BATCHED', 'EXL3_MOE_ROW_TILE')},
        'torch': torch.__version__, 'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name()}
    args.out.mkdir(parents=True, exist_ok=False)
    np.savez(args.out / 'stages.npz', **arrays)
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if report['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
