# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded diagnostic mirror of Atlas' existing BF16 cuBLASLt wrapper."""
import ctypes as c
import hashlib
from pathlib import Path


def file_hash(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def symbol_binding(function):
    class DlInfo(c.Structure):
        _fields_ = [('filename', c.c_char_p), ('base', c.c_void_p),
                    ('symbol', c.c_char_p), ('address', c.c_void_p)]
    process = c.CDLL(None)
    process.dladdr.argtypes = [c.c_void_p, c.POINTER(DlInfo)]
    process.dladdr.restype = c.c_int
    result = DlInfo()
    if process.dladdr(c.cast(function, c.c_void_p), c.byref(result)) != 1:
        raise RuntimeError('cannot attest actual cuBLASLt symbol binding')
    path = Path(result.filename.decode()).resolve(strict=True)
    return {'path': str(path), 'sha256': file_hash(path),
            'symbol': result.symbol.decode() if result.symbol else None}


def run_bf16(act, weight, library):
    import cupy as cp
    if act.dtype != cp.uint16 or weight.dtype != cp.uint16:
        raise ValueError('BF16 bit-storage matrices required')
    if act.ndim != 2 or weight.ndim != 2 or act.shape[1] != weight.shape[1]:
        raise ValueError('matching [M,K] and [N,K] required')
    if not act.flags.c_contiguous or not weight.flags.c_contiguous:
        raise ValueError('contiguous matrices required')
    path = Path(library).resolve(strict=True)
    lib = c.CDLL(str(path))
    p, i, u, z = c.c_void_p, c.c_int, c.c_uint, c.c_size_t
    signatures = {
        'cublasLtCreate': [c.POINTER(p)], 'cublasLtDestroy': [p],
        'cublasLtMatmulDescCreate': [c.POINTER(p), i, i],
        'cublasLtMatmulDescSetAttribute': [p, u, p, z],
        'cublasLtMatmulDescDestroy': [p],
        'cublasLtMatrixLayoutCreate': [c.POINTER(p), i, c.c_uint64, c.c_uint64, c.c_int64],
        'cublasLtMatrixLayoutDestroy': [p],
        'cublasLtMatmulPreferenceCreate': [c.POINTER(p)],
        'cublasLtMatmulPreferenceSetAttribute': [p, u, p, z],
        'cublasLtMatmulPreferenceDestroy': [p],
        'cublasLtMatmulAlgoGetHeuristic': [p,p,p,p,p,p,p,i,p,c.POINTER(i)],
        'cublasLtMatmul': [p,p,p,p,p,p,p,p,p,p,p,p,p,p,z,p],
    }
    for name, signature in signatures.items():
        function = getattr(lib, name); function.argtypes = signature; function.restype = i
    lib.cublasLtGetVersion.argtypes = []; lib.cublasLtGetVersion.restype = z
    binding = symbol_binding(lib.cublasLtMatmul)
    if binding['sha256'] != file_hash(path):
        raise RuntimeError('actual matmul symbol differs from explicitly selected library')
    version_binding = symbol_binding(lib.cublasLtGetVersion)
    if version_binding['sha256'] != binding['sha256']:
        raise RuntimeError('version and matmul symbols bind different libraries')
    def call(name, *args):
        status = getattr(lib, name)(*args)
        if status:
            raise RuntimeError(f'{name} failed status {status}')
    handles = []
    def create(name, destroy, *args):
        value = p(); call(name, c.byref(value), *args)
        handles.append((destroy, value)); return value
    m, k = act.shape; n = weight.shape[0]
    workspace = cp.empty(64 * 1024 * 1024, cp.uint8)
    guard = 0x42f6
    output = cp.full(m*n + 64, guard, cp.uint16)
    try:
        handle = create('cublasLtCreate', 'cublasLtDestroy')
        desc = create('cublasLtMatmulDescCreate', 'cublasLtMatmulDescDestroy', 68, 0)
        for attr, value in [(3, 1), (4, 0)]:
            number = i(value); call('cublasLtMatmulDescSetAttribute', desc, attr, c.byref(number), 4)
        layouts = [create('cublasLtMatrixLayoutCreate', 'cublasLtMatrixLayoutDestroy', 14, *shape)
                   for shape in [(k,n,k),(k,m,k),(n,m,n)]]
        la, lb, ld = layouts
        pref = create('cublasLtMatmulPreferenceCreate', 'cublasLtMatmulPreferenceDestroy')
        budget = z(workspace.nbytes)
        call('cublasLtMatmulPreferenceSetAttribute', pref, 1, c.byref(budget), c.sizeof(budget))
        # Match Rust's 128-byte result buffer; opaque algorithm occupies its first 64 bytes.
        result = (c.c_uint64 * 16)(); returned = i()
        call('cublasLtMatmulAlgoGetHeuristic', handle,desc,la,lb,ld,ld,pref,1,result,c.byref(returned))
        if returned.value != 1:
            raise RuntimeError('no first cuBLASLt heuristic for exact production shape')
        alpha, beta = c.c_float(1), c.c_float(0)
        call('cublasLtMatmul', handle,desc,c.byref(alpha),p(weight.data.ptr),la,
             p(act.data.ptr),lb,c.byref(beta),p(output.data.ptr),ld,p(output.data.ptr),ld,
             result,p(workspace.data.ptr),workspace.nbytes,p(cp.cuda.get_current_stream().ptr))
        cp.cuda.get_current_stream().synchronize()
        if not bool(cp.all(output[m*n:] == guard)):
            raise RuntimeError('cuBLASLt overwrote output guard')
        bits = cp.asnumpy(output[:m*n]).reshape(m,n)
        maps = [line for line in Path('/proc/self/maps').read_text().splitlines()
                if 'cublas' in line.lower()]
        libraries = sorted({line.split()[-1] for line in maps if line.split()[-1].startswith('/')})
        metadata = {'library': str(path), 'library_sha256': file_hash(path),
                    'matmul_symbol_binding': binding,
                    'version_symbol_binding': version_binding,
                    'version': int(lib.cublasLtGetVersion()), 'shape_m_n_k': [m,n,k],
                    'compute': 68, 'scale_dtype': 0, 'matrix_dtype': 14,
                    'transpose_a_b': [1,0], 'layout_rows_cols_ld': [[k,n,k],[k,m,k],[n,m,n]],
                    'workspace_bytes': workspace.nbytes, 'heuristics_requested': 1,
                    'heuristic_bytes_hex': bytes(result).hex(), 'guard_passed': True,
                    'loaded_library_maps': maps,
                    'loaded_library_sha256': {name: file_hash(name) for name in libraries}}
        return bits, metadata
    finally:
        for destroy, value in reversed(handles):
            call(destroy, value)
