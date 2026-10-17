# SPDX-License-Identifier: AGPL-3.0-only
import unittest
from kda_classic_oracle import reference_output, validate_identity, validate_capture_inventory
import numpy as np

class Tests(unittest.TestCase):
    def test_fused_extra_selects_exact_columns_without_padding_logical_rows(self):
        output=np.arange(2*12576,dtype=np.uint16).reshape(2,12576)
        selected=reference_output('extra',output,2,4096,288,32,128)
        np.testing.assert_array_equal(selected,output[:,12288:])
        with self.assertRaises(ValueError):reference_output('extra',output[:1],2,4096,288,32,128)
    def test_actual_identity_requires_matching_phase_positions_and_resident_invocation_zero(self):
        pos={'runner_source_sha256':'source','computed_before':0,'scheduled_tokens':5,'prefix_tokens':5,'phase':'terminal-prefill'}
        x={'layer':0,'rank':1,'invocation':3,'stage':'kda-qkvbfg-input0','chunk_position':pos}
        y=dict(x,stage='kda-qkvbfg');w=dict(x,invocation=0,stage='kda-resident-extra-weight')
        validate_identity('extra',x,y,w)
        with self.assertRaises(ValueError):validate_identity('extra',x,dict(y,chunk_position=dict(pos,prefix_tokens=6)),w)
        with self.assertRaises(ValueError):validate_identity('extra',x,y,dict(w,rank=0))
        with self.assertRaises(ValueError):validate_identity('extra',x,y,dict(w,invocation=2))
    def test_capture_inventory_is_path_independent_and_rejects_changed_bytes(self):
        import tempfile
        from pathlib import Path
        from cublaslt_oracle import file_hash
        with tempfile.TemporaryDirectory() as temp:
            files=[Path(temp)/f'captured-{i}.bin' for i in range(6)]
            for i,p in enumerate(files):p.write_bytes(bytes([i]))
            binding={'files':{p.name:file_hash(p) for p in files}}
            validate_capture_inventory(binding,files)
            files[0].write_bytes(b'changed')
            with self.assertRaises(ValueError):validate_capture_inventory(binding,files)

    def test_full_projection_geometry_is_explicit(self):
        a=np.zeros((1,4096),np.uint16)
        self.assertIs(reference_output('fb',a,1,128,4096,32,128),a)
        with self.assertRaises(ValueError):reference_output('fb',a,1,4096,4096,32,128)

if __name__=='__main__':unittest.main()
