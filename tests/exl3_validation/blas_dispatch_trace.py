# SPDX-License-Identifier: AGPL-3.0-only
"""Capture actual BLAS symbols for a synchronous host dispatch scope."""
import ctypes
from pathlib import Path
from cublaslt_oracle import file_hash


class BlasDispatchTrace:
    def __init__(self, library):
        self.path = Path(library).resolve(strict=True)
        self.lib = ctypes.CDLL(str(self.path))
        self.lib.atlas_blas_trace_begin.argtypes = []
        self.lib.atlas_blas_trace_begin.restype = None
        self.lib.atlas_blas_trace_end.argtypes = []
        self.lib.atlas_blas_trace_end.restype = ctypes.c_uint64
        self.lib.atlas_blas_trace_binding.argtypes = [ctypes.c_uint]
        self.lib.atlas_blas_trace_binding.restype = ctypes.c_char_p
        self.lib.atlas_blas_trace_gemm_argument.argtypes = [ctypes.c_uint]
        self.lib.atlas_blas_trace_gemm_argument.restype = ctypes.c_int
        self.records = {}

    def run(self, name, operation):
        self.lib.atlas_blas_trace_begin()
        try:
            value = operation()
        finally:
            counts = self.lib.atlas_blas_trace_end()
        count = [counts & 0xffffffff, counts >> 32]
        if sum(count) == 0:
            raise RuntimeError('BLAS dispatch interposition did not engage; no dispatch proof')
        bindings = {}
        for index, symbol in enumerate(('cublasGemmEx', 'cublasLtMatmul')):
            if count[index]:
                raw = self.lib.atlas_blas_trace_binding(index)
                if raw is None:
                    raise RuntimeError('called BLAS symbol binding unavailable')
                path = Path(raw.decode()).resolve(strict=True)
                bindings[symbol] = {'calls': count[index], 'path': str(path), 'sha256': file_hash(path)}
        if count[0]:
            names = ('trans_a', 'trans_b', 'm', 'n', 'k', 'a_dtype', 'b_dtype', 'c_dtype', 'lda', 'ldb', 'ldc', 'compute', 'algorithm')
            bindings['cublasGemmEx']['last_call_arguments'] = {key: self.lib.atlas_blas_trace_gemm_argument(index) for index, key in enumerate(names)}
        self.records[name] = bindings
        return value
