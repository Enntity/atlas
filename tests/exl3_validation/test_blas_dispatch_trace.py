# SPDX-License-Identifier: AGPL-3.0-only
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class DispatchInterposition(unittest.TestCase):
    def test_actual_symbol_passthrough_and_binding_without_cuda(self):
        common = Path(__file__).parent.resolve()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root / 'fake.cpp'
            fake.write_text('// SPDX-License-Identifier: AGPL-3.0-only\nextern "C" int cublasGemmEx(...) { return 123; }\nextern "C" int cublasLtMatmul(...) { return 42; }\n')
            backend = root / 'backend.so'; trace = root / 'trace.so'
            for source, output in [(fake, backend), (common/'blas_dispatch_trace.cpp', trace)]:
                subprocess.run(['g++', '-std=c++17', '-shared', '-fPIC', '-Wall', '-Wextra', '-Werror', str(source), '-ldl', '-o', str(output)], check=True, capture_output=True)
            script = '''import ctypes as c,sys
from blas_dispatch_trace import BlasDispatchTrace
backend=c.CDLL(sys.argv[1],mode=c.RTLD_GLOBAL)
trace=BlasDispatchTrace(sys.argv[2])
f=trace.lib.cublasGemmEx; f.argtypes=[c.c_void_p,c.c_int,c.c_int,c.c_int,c.c_int,c.c_int,c.c_void_p,c.c_void_p,c.c_int,c.c_int,c.c_void_p,c.c_int,c.c_int,c.c_void_p,c.c_void_p,c.c_int,c.c_int,c.c_int,c.c_int];f.restype=c.c_int
assert trace.run('classic',lambda:f(*([0]*19)))==123
f=trace.lib.cublasLtMatmul;f.argtypes=[c.c_void_p]*14+[c.c_size_t,c.c_void_p];f.restype=c.c_int
assert trace.run('lt',lambda:f(*([0]*16)))==42
assert trace.records['classic']['cublasGemmEx']['calls']==1
assert trace.records['lt']['cublasLtMatmul']['calls']==1
assert trace.records['classic']['cublasGemmEx']['path']==sys.argv[1]
assert trace.records['lt']['cublasLtMatmul']['path']==sys.argv[1]
try: trace.run('absent',lambda:0)
except RuntimeError as error: assert 'did not engage' in str(error)
else: raise AssertionError('missing interposition falsely passed')
'''
            env = dict(os.environ, LD_PRELOAD=str(trace), PYTHONPATH=str(common))
            subprocess.run([sys.executable, '-c', script, str(backend), str(trace)], env=env,
                           check=True, capture_output=True)


if __name__ == '__main__': unittest.main()
