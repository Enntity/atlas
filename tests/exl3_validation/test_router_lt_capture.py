# SPDX-License-Identifier: AGPL-3.0-only
import unittest
import numpy as np
from router_lt_capture import validate_pair

class RouterCaptureTests(unittest.TestCase):
    def test_actual_layer_request_dtype_shape_and_exact_bf16_transport(self):
        x=np.ones((128,4096),np.float32);y=np.ones((128,288),np.float32)
        m={'layer':3,'rank':0,'invocation':16,'dtype':'bf16-le','stage':'input','chunk_position':{'request_id':'frozen','computed_before':0,'scheduled':128}}
        n=dict(m,dtype='f32-le',stage='router-logits')
        validate_pair(x,y,m,n)
        with self.assertRaises(ValueError):validate_pair(x,y,m,dict(n,rank=1))
        with self.assertRaises(ValueError):validate_pair(x,y,m,dict(n,chunk_position={'request_id':'other'}))
        bad=y.copy();bad[0,0]=np.float32(1.00001)
        with self.assertRaises(ValueError):validate_pair(x,bad,m,n)
        with self.assertRaises(ValueError):validate_pair(x,y[:1],m,n)

    def test_pair_preload_loads_lt_then_classic_before_gpu_import(self):
        from router_lt_capture import preload_pair
        from pathlib import Path
        import ctypes
        calls=[]
        preload_pair([Path('classic'),Path('lt')],lambda p,mode:calls.append((p,mode)))
        self.assertEqual(calls,[('lt',ctypes.RTLD_GLOBAL),('classic',ctypes.RTLD_GLOBAL)])
