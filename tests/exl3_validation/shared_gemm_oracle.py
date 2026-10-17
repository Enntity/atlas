# SPDX-License-Identifier: AGPL-3.0-only
"""Candidate-library shared BF16 GEMM/activation on actual resident weights/input."""
import argparse
import json
from pathlib import Path

import numpy as np
from cublaslt_oracle import run_bf16
from differential_router import digest, load_capture
from shared_reconstruction import byte_metrics
from shared_gemm_controls import independent_gemm_controls, reduction_controls


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('input-json','gate-up-json','down-json','cublaslt-library','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--reference-binary-sha256',required=True)
    p.add_argument('--limit',type=float,required=True)
    p.add_argument('--diagnostic-reference-cublaslt-library',type=Path)
    p.add_argument('--blas-trace-library',type=Path)
    p.add_argument('--diagnostic-candidate-cublas-library',type=Path)
    args=p.parse_args()
    if not np.isfinite(args.limit) or args.limit<=0:
        p.error('explicit positive finite shared clamp required')
    captures={}; bindings={}; identities=[]; resident_metadata={}
    for name,path in [('input',args.input_json),('gate_up_weight',args.gate_up_json),('down_weight',args.down_json)]:
        values,metadata,raw=load_capture(path)
        if metadata['dtype']!='bf16-le' or ('sha256' in metadata and metadata['sha256']!=digest(raw)):
            raise ValueError('exact BF16 capture bytes required')
        identities.append((metadata['rank'],metadata['layer']))
        captures[name]=values; bindings.update({str(path):digest(path),str(raw):digest(raw)})
        if name!='input': resident_metadata[name]=metadata
    if len(set(identities))!=1:
        raise ValueError('input/resident shared weight identity differs')
    x,gu,down=[captures[n] for n in ('input','gate_up_weight','down_weight')]
    mid=gu.shape[0]//2
    if gu.shape[0]%2 or gu.shape[1]!=x.shape[1] or down.shape!=(x.shape[1],mid):
        raise ValueError('exact merged/shared down geometry required')
    from blas_dispatch_trace import BlasDispatchTrace
    trace = BlasDispatchTrace(args.blas_trace_library) if args.blas_trace_library else None
    def dispatch(name, operation):
        return trace.run(name, operation) if trace else operation()
    import torch
    import cupy as cp
    import vllm._C_stable_libtorch as extension
    if digest(extension.__file__)!=args.reference_binary_sha256:
        raise ValueError('compiled installed activation extension changed')
    if not torch.cuda.is_available() or cp.cuda.runtime.getDeviceCount()!=1:
        raise RuntimeError('one visible real CUDA device required')
    for metadata in resident_metadata.values():
        if (metadata['allow_tf32']!=torch.backends.cuda.matmul.allow_tf32
                or metadata['torch_matmul_precision']!=torch.get_float32_matmul_precision()
                or metadata['allow_bf16_reduced_precision_reduction']!=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction):
            raise RuntimeError('Torch diagnostic flags differ from actual resident-weight capture')
    tx,tgu,td=[torch.tensor(v,device='cuda',dtype=torch.bfloat16) for v in (x,gu,down)]
    expected_gu=dispatch('reference_gate_up', lambda: torch.nn.functional.linear(tx,tgu))
    expected_act=torch.empty((len(x),mid),device='cuda',dtype=torch.bfloat16)
    torch.ops._C.silu_and_mul_with_clamp(expected_act,expected_gu,args.limit,1.,0.)
    expected_down=dispatch('reference_independent_down', lambda: torch.nn.functional.linear(expected_act,td))
    arrays={name:value.view(torch.uint16).cpu().numpy() for name,value in
            [('reference_gate_up',expected_gu),('reference_activation',expected_act),('reference_down',expected_down)]}
    bits=lambda v:(v.view(np.uint32)>>16).astype(np.uint16)
    dx,dgu,dd=[cp.asarray(bits(v)) for v in (x,gu,down)]
    lt_metadata={}
    merged,lt_metadata['merged']=run_bf16(dx,dgu,args.cublaslt_library)
    arrays['native_merged_gate_up']=merged
    gate,lt_metadata['separate_gate']=run_bf16(dx,dgu[:mid],args.cublaslt_library)
    up,lt_metadata['separate_up']=run_bf16(dx,dgu[mid:],args.cublaslt_library)
    arrays['native_separate_gate_up']=np.concatenate([gate,up],axis=1)
    common=Path(__file__).resolve().parents[2]/'kernels/gb10/glm-5.3-flash/exl3'
    source=common/'exl3.cu'; header=common/'exl3_common.cuh'
    module=cp.RawModule(code=source.read_text()+'\n// header-sha '+digest(header),
                        options=('--std=c++17','--fmad=false','-I'+str(common)))
    activation=module.get_function('exl3_silu_mul_dense')
    # Mirrors Rust's row-strided merged split into contiguous gate/up arenas.
    dgate=cp.asarray(merged[:,:mid].copy()); dup=cp.asarray(merged[:,mid:].copy())
    activation(((dgate.size+255)//256,),(256,),
               (dgate,dup,dgate,np.uint64(dgate.size),np.float32(args.limit),np.uint32(1)))
    cp.cuda.get_current_stream().synchronize()
    arrays['native_activation']=cp.asnumpy(dgate)
    arrays['native_down'],lt_metadata['down']=run_bf16(dgate,dd,args.cublaslt_library)
    comparisons={name:byte_metrics(arrays[reference],arrays[candidate]) for name,reference,candidate in
        [('merged_gate_up','reference_gate_up','native_merged_gate_up'),
         ('separate_gate_up_control','reference_gate_up','native_separate_gate_up'),
         ('activation','reference_activation','native_activation'),('down','reference_down','native_down')]}
    diagnostics = None
    if args.diagnostic_reference_cublaslt_library is not None:
        reference_act_bits = cp.asarray(arrays['reference_activation'])
        control_arrays, control_metadata = independent_gemm_controls(dx, dgu, reference_act_bits, dd,
            {'candidate': args.cublaslt_library, 'reference': args.diagnostic_reference_cublaslt_library}, run_bf16)
        arrays.update({'control_' + name: value for name, value in control_arrays.items()})
        control_metrics = {name: byte_metrics(arrays['reference_gate_up' if name.endswith('gate_up') else 'reference_down'], value)
                           for name, value in control_arrays.items()}
        def run_torch_control(flag):
            gu = dispatch('torch_reduced_' + str(flag) + '_gate_up', lambda: torch.nn.functional.linear(tx, tgu))
            down_result = dispatch('torch_reduced_' + str(flag) + '_independent_down', lambda: torch.nn.functional.linear(expected_act, td))
            result = {'gate_up': gu.view(torch.uint16).cpu().numpy(),
                      'independent_down': down_result.view(torch.uint16).cpu().numpy()}
            arrays.update({'torch_reduced_' + str(flag) + '_' + name: value for name, value in result.items()})
            return {name: byte_metrics(arrays['reference_gate_up' if name == 'gate_up' else 'reference_down'], value)
                    for name, value in result.items()}
        torch_controls = reduction_controls(
            lambda: torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            lambda value: setattr(torch.backends.cuda.matmul, 'allow_bf16_reduced_precision_reduction', value),
            run_torch_control)
        diagnostics = {'scope': 'diagnostic one-variable controls; qualification metrics unchanged',
                       'independent_input': 'reference_activation', 'libraries': control_metadata,
                       'metrics': control_metrics, 'torch_reduced_precision_controls': torch_controls,
                       'torch_preferred_blas_backend': str(torch.backends.cuda.preferred_blas_library()),
                       'torch_git_version': torch.version.git_version}
    classic_diagnostics = None
    if args.diagnostic_candidate_cublas_library is not None:
        from cublas_oracle import run_bf16 as run_classic
        classic_diagnostics = {}
        reference_act_bits = cp.asarray(arrays['reference_activation'])
        for reduced in (False, True):
            for name, activation_input, weight_input, reference_name in [
                    ('gate_up', dx, dgu, 'reference_gate_up'),
                    ('independent_down', reference_act_bits, dd, 'reference_down')]:
                values, metadata = run_classic(activation_input, weight_input,
                    args.diagnostic_candidate_cublas_library, reduced_precision=reduced,
                    compute=68, algorithm=99, workspace_bytes=0)
                key = str(reduced) + '_' + name
                arrays['classic_' + key] = values
                classic_diagnostics[key] = {'metrics': byte_metrics(arrays[reference_name], values),
                                            'dispatch': metadata}
    bindings.update({str(p):digest(p) for p in [Path(__file__),source,header,Path(__file__).with_name('cublas_oracle.py'), Path(__file__).with_name('cublaslt_oracle.py'),
                                              Path(__file__).with_name('shared_gemm_controls.py'), Path(__file__).with_name('blas_dispatch_trace.py'), Path(__file__).with_name('blas_dispatch_trace.cpp')]})
    args.out.mkdir(parents=True,exist_ok=False); np.savez(args.out/'stages.npz',**arrays)
    report={'scope':'actual resident shared weights/input, merged BF16 candidate Lt and native activation; no full-model parity',
        'candidate_classic_diagnostics': classic_diagnostics,
        'blas_dispatch_trace': None if trace is None else {'library': str(trace.path), 'sha256': digest(trace.path), 'operators': trace.records}, 'diagnostic_controls': diagnostics, 'rank_layer':list(identities[0]),'metrics':comparisons,'cublaslt':lt_metadata,
        'source_sha256':bindings,'activation_binary_sha256':digest(extension.__file__),'limit':args.limit,
        'resident_parameter_metadata':resident_metadata,
        'torch':torch.__version__,'cupy':cp.__version__,'gpu':torch.cuda.get_device_name(),
        'allow_tf32':torch.backends.cuda.matmul.allow_tf32,'allow_bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        'output_sha256':digest(args.out/'stages.npz')}
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(comparisons))
    return 0 if all(comparisons[n]['exact'] for n in ('merged_gate_up','activation','down')) else 1


if __name__=='__main__':
    raise SystemExit(main())
