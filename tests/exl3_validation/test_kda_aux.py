# SPDX-License-Identifier: AGPL-3.0-only
import unittest
from kda_aux import validate_geometry, extra_slices, validate_ptx
from pathlib import Path
import tempfile
import hashlib
import json
class Tests(unittest.TestCase):
    def test_explicit_extra_order_and_padding(self):
        self.assertEqual(extra_slices(32,128),[(0,32),(32,160),(160,288)])
        self.assertEqual(validate_geometry('extra',1,4096,288,32,128),3)
        self.assertEqual(validate_geometry('extra',5,4096,288,32,128),5)
        self.assertEqual(validate_geometry('fb',1,128,4096,32,128),1)
    def test_wrong_shape_and_unknown_operation_fail(self):
        for op,m,k,n in [('extra',1,4096,287),('fb',1,4096,128),('guess',1,128,4096),('extra',0,4096,288)]:
            with self.assertRaises(ValueError):validate_geometry(op,m,k,n,32,128)
    def test_arbitrary_ptx_without_final_binary_attestation_fails(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);(root/'binding.json').write_text('{}')
            with self.assertRaises(ValueError):validate_ptx(root/'v.ptx',root/'m.ptx',root/'binding.json',root,'pipelined')
    def test_exact_ptx_contract_and_abi_are_required(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);modules={}
            for module,entry,count in [('dense_gemv_bf16','dense_gemv_bf16',5),('dense_gemm_bf16','dense_gemm_bf16_pipelined',6)]:
                source=root/(module+'.cu');source.write_text('// source fixture')
                path=root/(module+'.ptx');path.write_text('.entry '+entry+'('+','.join('.param .u64 p'+str(i) for i in range(count))+') {}')
                sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
                modules[module]={'file':path.name,'sha256':sha(path),'contained_in_binary':True,
                    'sources':{source.name:sha(source)},'entry_argument_counts':{entry:count}}
            value={'image':'immutable-fixture','embedded_binary_sha256':'a'*64,'build_source_sha256':'b'*64,'modules':modules}
            binding=root/'binding.json';binding.write_text(json.dumps(value))
            v=root/'dense_gemv_bf16.ptx';m=root/'dense_gemm_bf16.ptx'
            self.assertEqual(validate_ptx(v,m,binding,root,'pipelined')['image'],'immutable-fixture')
            value['modules']['dense_gemm_bf16']['entry_argument_counts']['dense_gemm_bf16_pipelined']=5
            binding.write_text(json.dumps(value))
            with self.assertRaises(ValueError):validate_ptx(v,m,binding,root,'pipelined')
if __name__=='__main__':unittest.main()

class ReplayPolicyTests(unittest.TestCase):
    def test_torch_replay_requires_explicit_policy_and_both_flags(self):
        from kda_aux import validate_replay_policy
        self.assertFalse(validate_replay_policy('disabled',None,None))
        self.assertTrue(validate_replay_policy('requested-flags-diagnostic','off','on'))
        for policy,bf16,tf32 in [('disabled','off','on'),('requested-flags-diagnostic',None,'on'),('unknown',None,None)]:
            with self.assertRaises(ValueError):validate_replay_policy(policy,bf16,tf32)
