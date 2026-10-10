# SPDX-License-Identifier: AGPL-3.0-only
"""Explicit traced classic BF16 cuBLAS profile; diagnostic only."""
import ctypes as c
from pathlib import Path
from cublaslt_oracle import file_hash, symbol_binding


def classic_arguments(rows, columns, inner, compute, algorithm):
    if any(not isinstance(value, int) or value <= 0 or value >= 2**31
           for value in (rows, columns, inner)):
        raise ValueError('explicit positive i32 GEMM geometry required')
    if (compute, algorithm) != (68, 99):
        raise ValueError('only actual traced compute68/algorithm99 profile is supported')
    return (1, 0, columns, rows, inner, 14, 14, 14,
            inner, inner, columns, compute, algorithm)


def run_bf16(act, weight, library, *, reduced_precision, compute, algorithm, workspace_bytes):
    import cupy as cp
    if type(workspace_bytes) is not int or workspace_bytes < 0:
        raise ValueError('explicit nonnegative workspace bytes required')
    if type(reduced_precision) is not bool:
        raise ValueError('explicit reduced precision flag required')
    if act.dtype != cp.uint16 or weight.dtype != cp.uint16:
        raise ValueError('BF16 bit-storage matrices required')
    if act.ndim != 2 or weight.ndim != 2 or act.shape[1] != weight.shape[1]:
        raise ValueError('matching [M,K] and [N,K] required')
    if not act.flags.c_contiguous or not weight.flags.c_contiguous:
        raise ValueError('contiguous matrices required')
    m, k = act.shape; n = weight.shape[0]
    arguments = classic_arguments(m, n, k, compute, algorithm)
    path = Path(library).resolve(strict=True)
    lib = c.CDLL(str(path))
    p, i = c.c_void_p, c.c_int
    signatures = {
        'cublasCreate_v2': [c.POINTER(p)], 'cublasDestroy_v2': [p],
        'cublasSetStream_v2': [p, p], 'cublasSetMathMode': [p, i],
        'cublasSetWorkspace_v2': [p, p, c.c_size_t],
        'cublasGetVersion_v2': [p, c.POINTER(i)],
        'cublasGemmEx': [p,i,i,i,i,i,p,p,i,i,p,i,i,p,p,i,i,i,i],
    }
    for name, signature in signatures.items():
        function = getattr(lib, name); function.argtypes = signature; function.restype = i
    binding = symbol_binding(lib.cublasGemmEx)
    if binding['sha256'] != file_hash(path):
        raise RuntimeError('classic GEMM binding differs from explicitly selected library')
    version_binding = symbol_binding(lib.cublasGetVersion_v2)
    if version_binding['sha256'] != binding['sha256']:
        raise RuntimeError('classic version and GEMM bind different libraries')
    def call(name, *args):
        status = getattr(lib, name)(*args)
        if status:
            raise RuntimeError(f'{name} failed status {status}')
    handle = p(); call('cublasCreate_v2', c.byref(handle))
    guard = 0x42f6
    storage = cp.full(m*n + 128, guard, cp.uint16)
    output = storage[64:64+m*n]
    workspace = cp.empty(workspace_bytes, cp.uint8) if workspace_bytes else None
    try:
        call('cublasSetStream_v2', handle, p(cp.cuda.get_current_stream().ptr))
        if workspace is not None:
            call('cublasSetWorkspace_v2', handle, p(workspace.data.ptr), workspace.nbytes)
        # PyTorch BF16 helper: default math, or disallow reduced-precision reduction.
        math_mode = 0 if reduced_precision else 16
        call('cublasSetMathMode', handle, math_mode)
        alpha, beta = c.c_float(1), c.c_float(0)
        call('cublasGemmEx', handle, 1, 0, n, m, k, c.byref(alpha),
             p(weight.data.ptr), 14, k, p(act.data.ptr), 14, k,
             c.byref(beta), p(output.data.ptr), 14, n, compute, algorithm)
        cp.cuda.get_current_stream().synchronize()
        if not bool(cp.all(storage[:64] == guard)) or not bool(cp.all(storage[64+m*n:] == guard)):
            raise RuntimeError('classic cuBLAS overwrote output guards')
        version = i(); call('cublasGetVersion_v2', handle, c.byref(version))
        return cp.asnumpy(output).reshape(m, n), {
            'library': str(path), 'library_sha256': file_hash(path),
            'gemm_symbol_binding': binding, 'version_symbol_binding': version_binding,
            'version': version.value, 'shape_m_n_k': [m,n,k],
            'traced_argument_values': list(arguments), 'math_mode': math_mode,
            'reduced_precision': reduced_precision, 'guard_passed': True,
            'workspace_bytes': workspace_bytes, 'workspace_policy': 'explicit' if workspace_bytes else 'library_internal',
        }
    finally:
        call('cublasDestroy_v2', handle)
