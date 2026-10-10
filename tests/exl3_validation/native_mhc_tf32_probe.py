# SPDX-License-Identifier: AGPL-3.0-only
"""Same frozen BF16 inputs: installed DeepGEMM raw split partials versus native probe."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from differential_router import digest


def load_mhc_capture(path):
    metadata=json.loads(path.read_text())
    name=metadata['file']
    shape=metadata['shape']
    dtype=metadata['dtype']
    expected_rank={'bf16-le':3,'f32-le':2}.get(dtype)
    if (Path(name).name!=name or expected_rank is None or len(shape)!=expected_rank
            or any(type(v) is not int or v<=0 for v in shape)
            or (dtype=='bf16-le' and shape[1]!=4)):
        raise ValueError('local finite BF16 four-stream residual or F32 matrix required')
    raw=path.parent/name
    width=2 if dtype=='bf16-le' else 4
    if raw.stat().st_size!=int(np.prod(shape))*width:
        raise ValueError('capture byte size must match exact shape')
    if dtype=='bf16-le':
        values=(np.fromfile(raw,'<u2').astype(np.uint32)<<16).view(np.float32)
    else:values=np.fromfile(raw,'<f4')
    values=values.reshape(shape)
    if not np.isfinite(values).all():raise ValueError('finite capture required')
    return values,metadata,raw


def validate_case(rows,width,splits):
    if rows<=0 or width<=0 or width%64 or splits<=0 or splits>width//64:
        raise ValueError('positive rows, K64 fragments and valid explicit split count required')
    return ((rows+127)//128,splits)


def compare(a,b):
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise RuntimeError('finite matching raw partial shapes required')
    return {'shape':list(a.shape),'byte_exact':a.tobytes()==b.tobytes(),
            'unequal_bits':int(np.count_nonzero(a.view(np.uint32)!=b.view(np.uint32))),
            'max_abs':float(np.abs(a-b).max())}


_SCALAR = r'''
extern "C" __global__ void scalar_control(const __nv_bfloat16* x,const float* w,
    float* dots,float* squares,unsigned rows,unsigned width,unsigned splits){
    __shared__ float d[256],s[256];
    unsigned row=blockIdx.x,col=blockIdx.y,split=blockIdx.z,t=threadIdx.x;
    unsigned blocks=width/64,rem=blocks%splits;
    unsigned start=(split*(blocks/splits)+min(split,rem))*64;
    unsigned count=(blocks/splits+(split<rem))*64;
    float a=0,b=0;
    for(unsigned k=t;k<count;k+=256){float v=__bfloat162float(x[(unsigned long long)row*width+start+k]);
        a=__fmaf_rn(v,w[(unsigned long long)col*width+start+k],a);b=__fmaf_rn(v,v,b);}
    d[t]=a;s[t]=b;__syncthreads();
    for(unsigned step=128;step;step/=2){if(t<step){d[t]=__fadd_rn(d[t],d[t+step]);s[t]=__fadd_rn(s[t],s[t+step]);}__syncthreads();}
    if(!t){dots[((unsigned long long)split*rows+row)*24+col]=d[0];if(!col)squares[(unsigned long long)split*rows+row]=s[0];}
}'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('input-json','weight-json','source','reference-source','capture-manifest','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--reference-source-sha256',required=True)
    p.add_argument('--rows',nargs='+',type=int,required=True)
    p.add_argument('--splits',nargs='+',type=int,required=True)
    args=p.parse_args()
    if len(args.rows)!=len(args.splits) or len(set(args.rows))!=len(args.rows):
        p.error('one explicit split count per unique row case required')
    x,xmeta,xraw=load_mhc_capture(args.input_json);w,wmeta,wraw=load_mhc_capture(args.weight_json)
    frozen=json.loads(args.capture_manifest.read_text())
    for path in (args.input_json,xraw,args.weight_json,wraw):
        if frozen.get(path.name)!=digest(path):raise ValueError('input bytes differ from frozen capture manifest')
    if xmeta['dtype']!='bf16-le' or wmeta['dtype']!='f32-le' or x.ndim!=3:
        raise ValueError('frozen BF16 residual and F32 projection weights required')
    x=x.reshape(len(x),-1);width=x.shape[1]
    if w.shape!=(24,width):raise ValueError('frozen mHC 24-channel weight geometry required')
    if digest(args.reference_source)!=args.reference_source_sha256:
        raise ValueError('installed DeepGEMM wrapper source differs from registration')
    for rows,splits in zip(args.rows,args.splits):validate_case(rows,width,splits)
    import torch
    import cupy as cp
    import vllm.utils.deep_gemm as deep_gemm
    tf32_hc_prenorm_gemm,is_deep_gemm_supported=deep_gemm.tf32_hc_prenorm_gemm,deep_gemm.is_deep_gemm_supported
    if digest(deep_gemm.__file__)!=args.reference_source_sha256:
        raise ValueError('actual imported DeepGEMM wrapper source differs')
    if not torch.cuda.is_available() or cp.cuda.runtime.getDeviceCount()!=1 or not is_deep_gemm_supported():
        raise RuntimeError('one real CUDA device with actual DeepGEMM support required')
    source=args.source.read_text();options=('--std=c++17','--fmad=false')
    module=cp.RawModule(code=source,options=options)
    kernel=module.get_function('native_mhc_tf32_probe')
    control=cp.RawModule(code='#include <cuda_bf16.h>\n'+_SCALAR,options=options).get_function('scalar_control')
    tw=torch.tensor(w,device='cuda',dtype=torch.float32)
    cases={};arrays={};guard=np.float32(12345)
    for rows,splits in zip(args.rows,args.splits):
        selected=x[np.arange(rows)%len(x)].copy()
        tx=torch.tensor(selected,device='cuda',dtype=torch.bfloat16)
        rd=torch.empty((splits,rows,24),device='cuda',dtype=torch.float32)
        rs=torch.empty((splits,rows),device='cuda',dtype=torch.float32)
        tf32_hc_prenorm_gemm(tx,tw,rd,rs,splits)
        torch.cuda.synchronize()
        dx=cp.full(tx.numel()+64,np.uint16(0x42f6),cp.uint16)
        dw=cp.full(tw.numel()+64,guard,cp.float32)
        dx[:tx.numel()]=cp.from_dlpack(tx.view(torch.uint16)).reshape(-1)
        dw[:tw.numel()]=cp.from_dlpack(tw).reshape(-1)
        nd,ns=cp.full(rd.numel()+64,guard,cp.float32),cp.full(rs.numel()+64,guard,cp.float32)
        cd,cs=cp.full_like(nd,guard),cp.full_like(ns,guard)
        vals=tuple(np.uint32(v) for v in (rows,width,splits))
        kernel(validate_case(rows,width,splits),(256,),(dx,dw,nd,ns,*vals))
        control((rows,24,splits),(256,),(dx,dw,cd,cs,*vals))
        cp.cuda.get_current_stream().synchronize()
        if (not np.all(cp.asnumpy(dx[tx.numel():])==np.uint16(0x42f6))
                or not np.all(cp.asnumpy(dw[tw.numel():])==guard)):
            raise RuntimeError('projection input/weight guard overwritten')
        for value,size in ((nd,rd.numel()),(ns,rs.numel()),(cd,rd.numel()),(cs,rs.numel())):
            if not np.all(cp.asnumpy(value[size:])==guard):raise RuntimeError('projection guard overwritten')
        a,b=rd.cpu().numpy(),rs.cpu().numpy()
        c,d=cp.asnumpy(nd[:rd.numel()]).reshape(a.shape),cp.asnumpy(ns[:rs.numel()]).reshape(b.shape)
        e,f=cp.asnumpy(cd[:rd.numel()]).reshape(a.shape),cp.asnumpy(cs[:rs.numel()]).reshape(b.shape)
        cases[str(rows)]={'splits':splits,'dots':compare(a,c),'squares':compare(b,d),
            'scalar_f32_tree_negative_dots':compare(a,e),'scalar_f32_tree_negative_squares':compare(b,f)}
        for label,v in (('reference_dots',a),('reference_squares',b),('native_dots',c),('native_squares',d),('scalar_dots',e),('scalar_squares',f)):
            arrays[f'rows{rows}_{label}']=v
    args.out.mkdir(parents=True,exist_ok=False);np.savez(args.out/'stages.npz',**arrays)
    report={'scope':'raw rounded mHC TF32 projection component; cyclic frozen rows; no model parity',
        'cases':cases,'source_sha256':{str(q):digest(q) for q in (args.input_json,xraw,args.weight_json,wraw,args.source,args.reference_source,args.capture_manifest,Path(__file__))},
        'actual_fn_tf32_representable_elements':int(np.count_nonzero((w.view(np.uint32)&0x1fff)==0)),
        'actual_fn_elements':int(w.size),'guard_passed':True,
        'compiler_options':list(options),'negative_control':'native-style 256-thread F32 FMA/tree within each balanced split; not execution of legacy full mHC kernel',
        'torch':torch.__version__,'cuda':torch.version.cuda,'cupy':cp.__version__,'gpu':torch.cuda.get_device_name(),
        'allow_tf32':torch.backends.cuda.matmul.allow_tf32,'float32_matmul_precision':torch.get_float32_matmul_precision(),
        'stages_sha256':digest(args.out/'stages.npz')}
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(cases,indent=2))
    return 0 if all(v['dots']['byte_exact'] and v['squares']['byte_exact'] for v in cases.values()) else 1


if __name__=='__main__':raise SystemExit(main())
