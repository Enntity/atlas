# SPDX-License-Identifier: AGPL-3.0-only
"""Candidate-only first-heuristic Lt BF16 KDA auxiliary diagnostic; no Torch."""
import argparse
import ctypes
import json
from pathlib import Path
import sys
if any(name=='torch' or name.startswith('torch.') for name in sys.modules):
    raise RuntimeError('Use Python -S with explicit package paths; Torch must not be imported')
import numpy as np
from cublaslt_oracle import run_bf16
from kda_classic_oracle import validate_identity, reference_output, validate_capture_inventory
from cublaslt_oracle import file_hash
from kda_aux import capture, exact, validate_geometry


def validate_workspace(value):
    if value != 64*1024*1024:
        raise ValueError('explicit existing Lt workspace must be 64MiB')


def validate_pair(loaded, candidate):
    if loaded != set(candidate):
        raise RuntimeError('loaded BLAS differs from exact candidate-only pair')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('input-json','output-json','weight-json','capture-binding','candidate-cublas','candidate-cublaslt','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--operation',choices=['extra','fb','gb'],required=True)
    parser.add_argument('--heads',type=int,required=True);parser.add_argument('--dim',type=int,required=True)
    parser.add_argument('--workspace-bytes',type=int,required=True)
    args=parser.parse_args()
    validate_workspace(args.workspace_bytes)
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
    pair_handles=[ctypes.CDLL(str(p),mode=ctypes.RTLD_GLOBAL) for p in (candidate[1],candidate[0])]
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount()!=1:raise RuntimeError('exactly one real GPU required')
    xx=cp.asarray(np.pad(x,((0,padded-rows),(0,0)),constant_values=0));ww=cp.asarray(w)
    output,dispatch=run_bf16(xx,ww,candidate[1])
    if dispatch['workspace_bytes'] != args.workspace_bytes:
        raise RuntimeError('Lt implementation workspace differs from explicit request')
    outputs={'lt_first_heuristic':output[:rows]}
    metrics={'lt_first_heuristic':{'metrics':exact(expected,outputs['lt_first_heuristic']),
                                   'dispatch':dispatch}}
    maps=[line for line in Path('/proc/self/maps').read_text().splitlines() if 'libcublas' in line]
    loaded={Path(line.split()[-1]).resolve(strict=True) for line in maps if line.split()[-1].startswith('/')}
    validate_pair(loaded,candidate)
    if any(name=='torch' or name.startswith('torch.') for name in sys.modules):raise RuntimeError('candidate-only replay imported Torch')
    args.out.mkdir(parents=True,exist_ok=False)
    np.savez(args.out/'outputs.npz',captured=expected,**outputs)
    hashes[str(args.capture_binding.resolve())]=file_hash(args.capture_binding)
    for name in ('kda_lt_oracle.py','kda_classic_oracle.py','kda_aux.py','cublas_oracle.py','cublaslt_oracle.py'):
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
    return 0 if metrics['lt_first_heuristic']['metrics']['byte_exact'] else 1

if __name__=='__main__':raise SystemExit(main())
