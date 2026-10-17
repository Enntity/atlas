# SPDX-License-Identifier: AGPL-3.0-only
"""Fixed-reference-input packed MMA/split-K diagnostic; no production policy."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from differential_expert import digest, load_parts, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--expert-prefix', required=True)
    parser.add_argument('--reference-plugin', type=Path, required=True)
    parser.add_argument('--reference-plugin-sha256', required=True)
    parser.add_argument('--stage-report', type=Path, required=True)
    parser.add_argument('--stages-npz', type=Path, required=True)
    parser.add_argument('--stages-sha256', required=True)
    parser.add_argument('--launch-source-manifest', type=Path, required=True)
    parser.add_argument('--rank', type=int, choices=(0,1), required=True)
    parser.add_argument('--tile-k', type=int, choices=(16,32,48,64), required=True)
    parser.add_argument('--tile-n', type=int, required=True)
    parser.add_argument('--slices', type=int, required=True)
    parser.add_argument('--half-partial', type=int, choices=(0,1), required=True)
    parser.add_argument('--abs', dest='absolute', type=float, required=True)
    parser.add_argument('--rel', dest='relative', type=float, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.tile_n < 32 or args.tile_n % 32 or args.slices < 1:
        parser.error('explicit valid tile-N and positive slice count required')
    if not np.isfinite([args.absolute,args.relative]).all() or min(args.absolute,args.relative)<0:
        parser.error('finite nonnegative tolerances required')
    if digest(args.reference_plugin)!=args.reference_plugin_sha256 or digest(args.stages_npz)!=args.stages_sha256:
        parser.error('immutable input/plugin hash changed')
    old = json.loads(args.stage_report.read_text())
    if old['rank'] != args.rank or old['expert_prefix'] != args.expert_prefix:
        parser.error('captured stage rank/expert identity mismatch')
    import torch
    import cupy as cp
    if not torch.cuda.is_available() or cp.cuda.runtime.getDeviceCount()!=1:
        raise RuntimeError('exactly one visible real CUDA device required')
    spec = importlib.util.spec_from_file_location('mma_probe_reference',args.reference_plugin)
    reference = importlib.util.module_from_spec(spec); sys.modules[spec.name]=reference
    spec.loader.exec_module(reference)
    parts, hashes, index_path = load_parts(args.checkpoint,args.expert_prefix,args.rank,reference)
    if hashes != old['selected_tensor_sha256']:
        raise ValueError('selected checkpoint payload differs from frozen reference stages')
    common = Path(__file__).resolve().parents[2]/'kernels/gb10/glm-5.3-flash/exl3'
    source = Path(__file__).with_suffix('.cu')
    header = common/'exl3_common.cuh'
    mma_header = common/'exl3_mma.cuh'
    module = cp.RawModule(code=source.read_text()+'\n// header-sha256 '+digest(header)+'\n// mma-header-sha256 '+digest(mma_header),
                          options=('--std=c++17','--fmad=false','-I'+str(common)))
    kernel = module.get_function('exl3_mma_probe')
    torch.cuda.synchronize()
    captures = np.load(args.stages_npz,allow_pickle=False)
    arrays, comparisons = {}, {}
    for name in ('up','down'):
        xh = captures['reference_'+name+'_xh']
        expected = captures['reference_'+name+'_yh']
        rows,k = xh.shape; n=parts[name]['svh'].numel()
        if (expected.shape!=(rows,n) or rows>65535 or k%args.tile_k or n%args.tile_n
                or not np.isfinite(xh).all() or not np.isfinite(expected).all()
                or not np.array_equal(xh,xh.astype(np.float16).astype(np.float32))):
            raise ValueError('exact F16 reference geometry and valid explicit launch required')
        dx = cp.asarray(xh,dtype=cp.float32)
        packed = cp.from_dlpack(parts[name]['trellis'])
        bits = parts[name]['trellis'].shape[-1]//16
        sentinel=np.float32(12345)
        out=cp.full(rows*n+64,sentinel,cp.float32)
        kernel(((n+31)//32,rows),(128,),
               (dx,packed,out,*map(np.uint32,(rows,n,k,bits,args.tile_k,args.tile_n,args.slices,args.half_partial))))
        cp.cuda.get_current_stream().synchronize()
        value=cp.asnumpy(out)
        if not np.all(value[rows*n:]==sentinel):
            raise RuntimeError('MMA probe output guard overwritten')
        raw=value[:rows*n].reshape(rows,n)
        rounded=raw.astype(np.float16).astype(np.float32)
        arrays[name+'_raw_f32']=raw; arrays[name+'_f16']=rounded; arrays[name+'_reference']=expected
        comparisons[name]=metrics(expected,rounded,args.absolute,args.relative)
        comparisons[name]['exact_f16_bits']=bool(np.array_equal(expected.astype(np.float16).view(np.uint16),
                                                               rounded.astype(np.float16).view(np.uint16)))
        comparisons[name]['unequal_f16_bits']=int(np.count_nonzero(expected.astype(np.float16).view(np.uint16)!=
                                                                  rounded.astype(np.float16).view(np.uint16)))
        comparisons[name]['scalar_control']=metrics(expected,captures['native_fixed_reference_'+name+'_yh_f16'],
                                                    args.absolute,args.relative)
    args.out.mkdir(parents=True,exist_ok=False)
    np.savez(args.out/'stages.npz',**arrays)
    bindings=[source,header,mma_header,Path(__file__),args.stage_report,args.stages_npz,args.launch_source_manifest,
              args.reference_plugin,index_path]
    report={'scope':'fixed-reference-xh MMA/split-K diagnostic only; no full-model claim',
        'rank':args.rank,'expert_prefix':args.expert_prefix,'geometry':{'tile_k':args.tile_k,'tile_n':args.tile_n,
            'slices':args.slices,'half_partial':args.half_partial,'block':128,'output_columns_per_block':32},
        'absolute':args.absolute,'relative':args.relative,'metrics':comparisons,
        'source_sha256':{str(p):digest(p) for p in bindings},'selected_tensor_sha256':hashes,
        'output_sha256':digest(args.out/'stages.npz'),'torch':torch.__version__,'cupy':cp.__version__,
        'gpu':torch.cuda.get_device_name(),'capability':list(torch.cuda.get_device_capability()),
        'driver':cp.cuda.runtime.driverGetVersion()}
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(comparisons))
    return 0 if all(item['exact_f16_bits'] for item in comparisons.values()) else 1


if __name__=='__main__':
    raise SystemExit(main())
