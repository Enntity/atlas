# SPDX-License-Identifier: AGPL-3.0-only
"""Captured router versus explicitly selected cuBLASLt pair; no Torch import."""
import argparse
import ctypes
import json
from pathlib import Path
import sys
import numpy as np
from differential_router import load_capture,validate_weights,digest
from cublaslt_oracle import run_bf16


def validate_pair(x,y,xm,ym):
    if xm.get('dtype')!='bf16-le' or ym.get('dtype')!='f32-le':
        raise ValueError('explicit BF16 input/F32 router transport required')
    if xm.get('stage')!='input' or ym.get('stage')!='router-logits':
        raise ValueError('actual FFN input/router stage required')
    if xm.get('layer')!=3 or ym.get('layer')!=3:
        raise ValueError('this weight contract explicitly binds layer3')
    for key in ('rank','invocation','chunk_position'):
        if xm.get(key)!=ym.get(key):raise ValueError('captured rank/invocation/chunk differs')
    if not xm.get('chunk_position') or xm['rank'] not in (0,1):
        raise ValueError('actual prepared request and TP rank required')
    if x.shape[1:]!=(4096,) or y.shape!=(len(x),288) or len(x) not in (1,5,10,128):
        raise ValueError('explicit actual router M1/5/10/128 K4096/N288 required')
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(y.view(np.uint32)&65535):
        raise ValueError('finite exact BF16 producer transport required')


def preload_pair(libs,load=ctypes.CDLL):
    # Load the exact pair explicitly; Lt alone need not map classic BLAS.
    load(str(libs[1]),mode=ctypes.RTLD_GLOBAL)
    load(str(libs[0]),mode=ctypes.RTLD_GLOBAL)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('input-json','logits-json','weight','weight-json','capture-binding','cublas','cublaslt','runtime-wrapper','out'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if any(n=='torch' or n.startswith('torch.') for n in sys.modules):raise RuntimeError('no-Torch process required')
    x,xm,xp=load_capture(a.input_json);y,ym,yp=load_capture(a.logits_json);validate_pair(x,y,xm,ym)
    wm=json.loads(a.weight_json.read_text());w=validate_weights(np.load(a.weight,allow_pickle=False),wm,a.weight)
    if wm['key']!='model.language_model.layers.3.mlp.gate.weight' or w.shape!=(288,4096):raise ValueError('actual layer3 checkpoint gate required')
    files=(a.input_json,xp,a.logits_json,yp,a.weight,a.weight_json)
    binding=json.loads(a.capture_binding.read_text())
    if binding.get('schema')!='atlas-router-capture-binding-v1' or binding.get('files')!={v.name:digest(v) for v in files}:
        raise ValueError('exact input/output/checkpoint capture inventory differs')
    if binding.get('request_chunk')!=xm['chunk_position'] or not binding.get('parent_evidence_sha256'):
        raise ValueError('actual request/parent source binding required')
    libs=[v.resolve(strict=True) for v in (a.cublas,a.cublaslt)]
    preload_pair(libs)
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount()!=1:raise RuntimeError('one explicit real GPU required')
    dx=cp.asarray((x.view(np.uint32)>>16).astype(np.uint16));dw=cp.asarray((w.view(np.uint32)>>16).astype(np.uint16))
    bits,dispatch=run_bf16(dx,dw,libs[1]);expected=(y.view(np.uint32)>>16).astype(np.uint16)
    maps=[v for v in Path('/proc/self/maps').read_text().splitlines() if 'libcublas' in v]
    loaded={Path(v.split()[-1]).resolve(strict=True) for v in maps if v.split()[-1].startswith('/')}
    if loaded!=set(libs):raise RuntimeError('actual mapped BLAS pair differs')
    if any(n=='torch' or n.startswith('torch.') for n in sys.modules):raise RuntimeError('Torch imported during replay')
    diff=int(np.count_nonzero(bits!=expected));a.out.mkdir(parents=True,exist_ok=False)
    np.savez(a.out/'outputs.npz',captured=expected,replayed=bits)
    sources=(*files,a.capture_binding,a.runtime_wrapper,Path(__file__),Path(__file__).with_name('cublaslt_oracle.py'),Path(__file__).with_name('differential_router.py'))
    report={'scope':'same captured router input/output; no full-model parity','identity':{k:xm[k] for k in ('layer','rank','invocation','chunk_position')},'bf16_elements_different':diff,'byte_exact':diff==0,'dispatch':dispatch,'loaded_blas_sha256':{str(v):digest(v) for v in loaded},'loaded_library_maps':maps,'source_sha256':{str(v.resolve()):digest(v) for v in sources},'outputs_sha256':digest(a.out/'outputs.npz'),'torch_imported':False}
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));return int(diff!=0)

if __name__=='__main__':raise SystemExit(main())
