# SPDX-License-Identifier: AGPL-3.0-only
"""Candidate-only classic BF16 KDA auxiliary replay; no Torch/interposition."""
import argparse
import ctypes
import json
from pathlib import Path
import sys
if any(name=='torch' or name.startswith('torch.') for name in sys.modules):
    raise RuntimeError('Use Python -S with explicit package paths; Torch must not be imported')
import numpy as np
from cublas_oracle import run_bf16
from cublaslt_oracle import file_hash
from kda_aux import capture, exact, validate_geometry


def validate_identity(operation,x,y,w):
    stages={'extra':('kda-qkvbfg-input0','kda-qkvbfg','kda-resident-extra-weight'),
            'fb':('kda-g1-input0','kda-g1','kda-resident-f_b_proj-weight'),
            'gb':('kda-g2-input0','kda-g2','kda-resident-g_b_proj-weight')}
    if operation not in stages:raise ValueError('explicit extra/fb/gb operation required')
    if tuple(m['stage'] for m in (x,y,w))!=stages[operation]:raise ValueError('operation stages differ')
    if any(m['layer']!=0 for m in (x,y,w)) or len({m['rank'] for m in (x,y,w)})!=1:
        raise ValueError('actual layer/rank identity differs')
    if x['invocation']!=y['invocation'] or w['invocation']!=0:
        raise ValueError('input/output invocation or first resident weight identity differs')
    if not x.get('chunk_position') or x['chunk_position']!=y.get('chunk_position'):
        raise ValueError('actual prepared request/chunk identity differs')


def reference_output(operation,output,rows,k,n,heads,dim):
    validate_geometry(operation,rows,k,n,heads,dim)
    if operation=='extra':
        packed=3*heads*dim
        if output.shape!=(rows,packed+n):raise ValueError('fused output full rows/columns differ')
        return output[:,packed:]
    if output.shape!=(rows,n):raise ValueError('full output geometry differs')
    return output


def validate_capture_inventory(binding,files):
    names=[p.name for p in files]
    if len(files)!=6 or len(set(names))!=6 or set(binding['files'])!=set(names):
        raise ValueError('exact six distinct capture metadata/raw filenames required')
    if any(binding['files'][p.name]!=file_hash(p) for p in files):
        raise ValueError('capture source inventory differs')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('input-json','output-json','weight-json','capture-binding','candidate-cublas','candidate-cublaslt','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--operation',choices=['extra','fb','gb'],required=True)
    parser.add_argument('--heads',type=int,required=True);parser.add_argument('--dim',type=int,required=True)
    parser.add_argument('--workspace-bytes',type=int,required=True)
    args=parser.parse_args()
    if args.workspace_bytes!=32*1024*1024:raise ValueError('explicit pinned classic workspace must be32MiB')
    x,xm,xp=capture(args.input_json);y,ym,yp=capture(args.output_json);w,wm,wp=capture(args.weight_json)
    validate_identity(args.operation,xm,ym,wm)
    rows,k=x.shape;n,wk=w.shape
    if wk!=k:raise ValueError('captured resident weight/input K differs')
    padded=validate_geometry(args.operation,rows,k,n,args.heads,args.dim)
    expected=reference_output(args.operation,y,rows,k,n,args.heads,args.dim)
    binding=json.loads(args.capture_binding.read_text())
    files=(args.input_json,args.output_json,args.weight_json,xp,yp,wp)
    hashes={str(p.resolve()):file_hash(p) for p in files}
    if binding.get('schema')!='atlas-kda-aux-capture-binding-v1':raise ValueError('explicit capture/source binding schema required')
    validate_capture_inventory(binding,files)
    if xm['chunk_position']['runner_source_sha256']!=binding['runner_source_sha256']:
        raise ValueError('actual runner source differs from binding')
    if not binding.get('hook_package_binding_sha256'):raise ValueError('actual frozen hook package binding required')
    candidate=[p.resolve(strict=True) for p in (args.candidate_cublas,args.candidate_cublaslt)]
    ctypes.CDLL(str(candidate[1]),mode=ctypes.RTLD_GLOBAL)
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount()!=1:raise RuntimeError('exactly one real GPU required')
    xx=cp.asarray(np.pad(x,((0,padded-rows),(0,0)),constant_values=0));ww=cp.asarray(w)
    outputs={};metrics={}
    for reduced in (False,True):
        output,dispatch=run_bf16(xx,ww,candidate[0],reduced_precision=reduced,
                               compute=68,algorithm=99,workspace_bytes=args.workspace_bytes)
        key='math16' if not reduced else 'math0_control'
        outputs[key]=output[:rows]
        metrics[key]={'metrics':exact(expected,outputs[key]),'dispatch':dispatch}
    maps=[line for line in Path('/proc/self/maps').read_text().splitlines() if 'libcublas' in line]
    loaded={Path(line.split()[-1]).resolve(strict=True) for line in maps if line.split()[-1].startswith('/')}
    if loaded!=set(candidate):raise RuntimeError('loaded BLAS differs from exact candidate-only pair')
    if any(name=='torch' or name.startswith('torch.') for name in sys.modules):raise RuntimeError('candidate-only replay imported Torch')
    args.out.mkdir(parents=True,exist_ok=False)
    np.savez(args.out/'outputs.npz',captured=expected,**outputs)
    hashes[str(args.capture_binding.resolve())]=file_hash(args.capture_binding)
    for name in ('kda_classic_oracle.py','kda_aux.py','cublas_oracle.py','cublaslt_oracle.py'):
        p=Path(__file__).with_name(name);hashes[str(p.resolve())]=file_hash(p)
    report={'scope':'same-input actual resident KDA auxiliary component only; no whole-model qualification',
        'operation':args.operation,'rank':xm['rank'],'layer':0,'invocation':xm['invocation'],
        'request_chunk':xm['chunk_position'],'physical_rows':rows,'reference_operator_rows':padded,
        'geometry_m_n_k':[rows,n,k],'metrics':metrics,'source_sha256':hashes,
        'loaded_blas':{str(p):file_hash(p) for p in loaded},'loaded_library_maps':maps,
        'torch_imported':False,'cupy':cp.__version__,'outputs_sha256':file_hash(args.out/'outputs.npz'),
        'capture_binding':binding}
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(metrics))
    return 0 if metrics['math16']['metrics']['byte_exact'] else 1

if __name__=='__main__':raise SystemExit(main())
