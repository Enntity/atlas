# SPDX-License-Identifier: AGPL-3.0-only
"""Strict same-input resident KDA auxiliary arithmetic controls, not model parity."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np


def extra_slices(heads,dim):
    if type(heads) is not int or type(dim) is not int or min(heads,dim)<=0:
        raise ValueError('explicit positive heads/dim required')
    return [(0,heads),(heads,heads+dim),(heads+dim,heads+2*dim)]


def validate_geometry(op,rows,k,n,heads,dim):
    if min(rows,k,n,heads,dim)<=0 or dim!=128:
        raise ValueError('positive H128 shape required')
    if op=='extra':
        if k!=4096 or n!=heads+2*dim:raise ValueError('extra must be merged b/fa/ga')
        return max(rows,3)
    if op in ('fb','gb'):
        if k!=dim or n!=heads*dim:raise ValueError('fb/gb must be full local projection')
        return rows
    raise ValueError('explicit extra/fb/gb operation required')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capture(path):
    m=json.loads(path.read_text())
    if m.get('dtype')!='bf16-le' or len(m.get('shape',[]))!=2 or min(m['shape'])<=0:
        raise ValueError('rank2 positive exact BF16 reference capture required')
    name=m['file']
    if Path(name).name!=name:raise ValueError('local capture filename required')
    p=path.parent/name;a=np.fromfile(p,'<u2')
    if a.size!=np.prod(m['shape']):raise ValueError('capture byte count mismatch')
    a=a.reshape(m['shape'])
    if not np.isfinite((a.astype(np.uint32)<<16).view(np.float32)).all():raise ValueError('nonfinite capture')
    return a,m,p


def exact(a,b):
    if a.shape!=b.shape:raise ValueError('full output shape differs')
    af=(a.astype(np.uint32)<<16).view(np.float32)
    bf=(b.astype(np.uint32)<<16).view(np.float32)
    return {'byte_exact':a.tobytes()==b.tobytes(),'elements_different':int(np.count_nonzero(a!=b)),
            'max_abs_error':float(np.max(np.abs(af.astype(np.float64)-bf.astype(np.float64))))}


def validate_ptx(gemv,gemm,binding,common,prefill):
    from native_ptx import validate_binding
    if gemv.parent!=gemm.parent:raise ValueError('one immutable PTX directory required')
    kernel='dense_gemm_bf16_pipelined' if prefill=='pipelined' else 'dense_gemm_bf16'
    contract={'dense_gemv_bf16':{'entries':['dense_gemv_bf16'],'sources':['dense_gemv_bf16.cu'],'argument_counts':{'dense_gemv_bf16':5}},
              'dense_gemm_bf16':{'entries':[kernel],'sources':['dense_gemm_bf16.cu'],'argument_counts':{kernel:6}}}
    value=validate_binding(gemv.parent,binding,common,required=contract)
    if value['modules']['dense_gemv_bf16']['file']!=gemv.name or value['modules']['dense_gemm_bf16']['file']!=gemm.name:
        raise ValueError('explicit PTX paths differ from attested modules')
    return value


def validate_replay_policy(policy,bf16,tf32):
    if policy=='disabled':
        if bf16 is not None or tf32 is not None:raise ValueError('disabled Torch replay must not supply flags')
        return False
    if policy=='requested-flags-diagnostic' and bf16 in ('on','off') and tf32 in ('on','off'):
        return True
    raise ValueError('explicit Torch replay policy and both diagnostic flags required')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('input-json','output-json','weight-json','common-dir','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--operation',choices=['extra','fb','gb'],required=True)
    p.add_argument('--heads',type=int,required=True);p.add_argument('--dim',type=int,required=True)
    p.add_argument('--torch-replay',choices=['disabled','requested-flags-diagnostic'],required=True)
    p.add_argument('--bf16-reduced-precision',choices=['on','off'])
    p.add_argument('--tf32',choices=['on','off'])
    p.add_argument('--prefill-kernel',choices=['scalar','pipelined'],required=True)
    p.add_argument('--module-mode',choices=['nvrtc-diagnostic','image-ptx'],required=True)
    p.add_argument('--gemv-ptx',type=Path);p.add_argument('--gemm-ptx',type=Path)
    p.add_argument('--ptx-binding',type=Path)
    args=p.parse_args()
    x,xm,xp=capture(args.input_json);y,ym,yp=capture(args.output_json);w,wm,wp=capture(args.weight_json)
    for key in ('layer','rank'):
        if len({m[key] for m in (xm,ym,wm)})!=1:raise ValueError('rank/layer capture identity mismatch')
    if xm['invocation']!=ym['invocation'] or xm['layer']!=0 or wm['invocation']!=0:
        raise ValueError('source-bound L0 input/output and first-invocation resident weight required')
    expected_weight={'extra':'kda-resident-extra-weight','fb':'kda-resident-f_b_proj-weight','gb':'kda-resident-g_b_proj-weight'}[args.operation]
    if wm['stage']!=expected_weight:raise ValueError('weight operation identity differs')
    expected_x={'extra':'kda-qkvbfg-input0','fb':'kda-g1-input0','gb':'kda-g2-input0'}[args.operation]
    expected_y={'extra':'kda-qkvbfg','fb':'kda-g1','gb':'kda-g2'}[args.operation]
    if xm['stage']!=expected_x or ym['stage']!=expected_y:raise ValueError('input/output operation stage mismatch')
    rows,k=x.shape;n,wk=w.shape
    padded=validate_geometry(args.operation,rows,k,n,args.heads,args.dim)
    if wk!=k:raise ValueError('weight/input K mismatch')
    if args.operation=='extra':
        packed_width=3*args.heads*args.dim
        if y.shape!=(rows,packed_width+n):raise ValueError('fused QKV+extra shape mismatch')
        y=y[:,packed_width:]
    elif y.shape!=(rows,n):raise ValueError('output shape mismatch')
    if (args.module_mode=='image-ptx') != bool(args.gemv_ptx and args.gemm_ptx and args.ptx_binding):
        raise ValueError('image PTX requires explicit gemv and gemm modules; diagnostic must not supply them')
    binding=validate_ptx(args.gemv_ptx,args.gemm_ptx,args.ptx_binding,args.common_dir,args.prefill_kernel) if args.module_mode=='image-ptx' else None
    if args.module_mode=='nvrtc-diagnostic' and any((args.gemv_ptx,args.gemm_ptx,args.ptx_binding)):
        raise ValueError('diagnostic source mode must not accept PTX options')
    replay=validate_replay_policy(args.torch_replay,args.bf16_reduced_precision,args.tf32)
    if not replay and any(name=='torch' or name.startswith('torch.') for name in sys.modules):
        raise RuntimeError('Disabled Torch replay requires a process without Torch imports')
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount()!=1:raise RuntimeError('exactly one real GPU required')
    reference_raw=None
    if replay:
        import torch
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=args.bf16_reduced_precision=='on'
        torch.backends.cuda.matmul.allow_tf32=args.tf32=='on'
        torch_x=torch.from_numpy(x.copy()).view(torch.bfloat16).cuda()
        torch_w=torch.from_numpy(w.copy()).view(torch.bfloat16).cuda()
        padded_x=torch.nn.functional.pad(torch_x,(0,0,0,padded-rows))
        reference=torch.nn.functional.linear(padded_x,torch_w)[:rows]
        reference_raw=reference.contiguous().view(torch.uint16).cpu().numpy()
    kernel='dense_gemm_bf16_pipelined' if args.prefill_kernel=='pipelined' else 'dense_gemm_bf16'
    source_names=['dense_gemv_bf16.cu','dense_gemm_bf16.cu']
    source_hashes={name:digest(args.common_dir/name) for name in source_names}
    if args.module_mode=='image-ptx':
        modules=[cp.RawModule(path=str(path)) for path in (args.gemv_ptx,args.gemm_ptx)]
    else:
        modules=[cp.RawModule(code=(args.common_dir/name).read_text(),options=('--std=c++17','--fmad=false','-I'+str(args.common_dir))) for name in source_names]
    gemv=modules[0].get_function('dense_gemv_bf16');gemm=modules[1].get_function(kernel)
    def native(matrix):
        nn=len(matrix);xx=cp.asarray(x);ww=cp.asarray(matrix)
        guards=cp.full(rows*nn+128,np.uint16(0xdead));out=guards[64:-64].reshape(rows,nn)
        if rows==1:
            gemv(((nn+3)//4,1,1),(256,1,1),(xx,ww,out,np.uint32(nn),np.uint32(k)))
        else:
            grid=((nn+127)//128,(rows+127)//128,1) if args.prefill_kernel=='pipelined' else ((nn+15)//16,(rows+15)//16,1)
            block=(256,1,1) if args.prefill_kernel=='pipelined' else (16,16,1)
            gemm(grid,block,(xx,ww,out,np.uint32(rows),np.uint32(nn),np.uint32(k)))
        cp.cuda.runtime.deviceSynchronize()
        raw=cp.asnumpy(guards)
        if not np.all(raw[:64]==0xdead) or not np.all(raw[-64:]==0xdead):raise RuntimeError('native guard corrupted')
        return cp.asnumpy(out)
    if args.operation=='extra':
        actual=np.concatenate([native(w[a:b]) for a,b in extra_slices(args.heads,args.dim)],axis=1)
        control_raw=None
        if replay:
            shape_control=torch.cat([torch.nn.functional.linear(torch_x,torch_w[a:b]) for a,b in extra_slices(args.heads,args.dim)],dim=1)
            control_raw=shape_control.contiguous().view(torch.uint16).cpu().numpy()
    else:
        actual=native(w);control_raw=reference_raw
    report={'scope':'actual resident BF16 weight, same captured BF16 input; original qualification baseline untouched',
            'operation':args.operation,'physical_rows':rows,'reference_padded_rows':padded,'geometry':[rows,n,k],
            'torch_matches_capture':exact(y,reference_raw) if replay else {'status':'SKIPPED','reason':'live worker math flags are not captured'},'native_matches_capture':exact(y,actual),
            'separate_shape_control':exact(y,control_raw) if replay else {'status':'SKIPPED'},'module_mode':args.module_mode,'prefill_kernel':args.prefill_kernel,
            'source_sha256':source_hashes,'capture_sha256':{str(v):digest(v) for v in (args.input_json,args.output_json,args.weight_json,xp,yp,wp)},
            'torch_replay':args.torch_replay,'torch':torch.__version__ if replay else None,'capability':cp.cuda.Device().compute_capability,
            'math_flags':{'matmul_allow_tf32':args.tf32,'allow_bf16_reduced_precision_reduction':args.bf16_reduced_precision,
                          'scope':'requested diagnostic flags; no live worker flag attestation' if replay else 'no Torch replay'},
            'harness_sha256':digest(__file__)}
    if args.module_mode=='image-ptx':
        report['ptx_binding']=binding
        report['ptx_sha256']={str(v):digest(v) for v in (args.gemv_ptx,args.gemm_ptx,args.ptx_binding)}
    if not replay and any(name=='torch' or name.startswith('torch.') for name in sys.modules):
        raise RuntimeError('Disabled replay unexpectedly imported Torch')
    args.out.mkdir(parents=True,exist_ok=False)
    arrays={'captured':y,'native':actual}
    if replay:arrays.update(torch=reference_raw,separate=control_raw)
    np.savez(args.out/'outputs.npz',**arrays)
    report['outputs_sha256']=digest(args.out/'outputs.npz')
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['native_matches_capture']))
    return 0 if (not replay or report['torch_matches_capture']['byte_exact']) and report['native_matches_capture']['byte_exact'] else 1
if __name__=='__main__':raise SystemExit(main())
