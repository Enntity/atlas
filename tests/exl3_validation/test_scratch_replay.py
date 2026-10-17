# SPDX-License-Identifier: AGPL-3.0-only
import unittest
from scratch_replay import validate_report
class Tests(unittest.TestCase):
    def test_actual_dispatch_report_rejects_wrong_binary_plan_and_unqualified_guards(self):
        report={'schema':'atlas-native-two-layer-scratch-replay-v1','passed':True,'plan_sha256':'a'*64,
                'executable_sha256':'b'*64,'guards_passed':True,'input_unchanged':True,
                'output_sha256':{'m1-step0':'c'*64},'ptx_sha256':{'exl3_moe_mma':'d'*64}}
        validate_report(report,'a'*64,'b'*64,{'exl3_moe_mma':'d'*64})
        for key,value in [('executable_sha256','c'*64),('plan_sha256','c'*64),('guards_passed',False),('passed',False)]:
            wrong=dict(report,**{key:value})
            with self.assertRaises(ValueError):validate_report(wrong,'a'*64,'b'*64,{'exl3_moe_mma':'d'*64})
    def test_wrong_loaded_ptx_rejected(self):
        report={'schema':'atlas-native-two-layer-scratch-replay-v1','passed':True,'plan_sha256':'a'*64,
                'executable_sha256':'b'*64,'guards_passed':True,'input_unchanged':True,
                'output_sha256':{'m1-step0':'c'*64},'ptx_sha256':{'exl3_moe_mma':'c'*64}}
        with self.assertRaises(ValueError):validate_report(report,'a'*64,'b'*64,{'exl3_moe_mma':'d'*64})

class RealCallerTests(unittest.TestCase):
    def test_real_subprocess_retains_registration_logs_and_checks_raw_outputs(self):
        import tempfile,json,hashlib
        from pathlib import Path
        from scratch_replay import ReplayIORouter,REQUIRED
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);binary=root/'cpu-caller-fixture';plan=root/'plan.json';plan.write_text('{}')
            modules={}
            for name in REQUIRED:
                entry='exl3_moe_mul1_matmul_precision_mma' if name=='exl3_moe_mma' else 'cpu_fixture'
                params=','.join('.param .u32 p'+str(i) for i in range(19)) if name=='exl3_moe_mma' else ''
                data=f'.entry {entry}({params}){{}}'
                path=root/(name+'.ptx');path.write_text(data)
                modules[name]={'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'contained_in_binary':True,'sources':{},'entry_argument_counts':{entry:19}}
            source='''#!/usr/bin/env python3
import json,sys,hashlib
from pathlib import Path
args=sys.argv;plan=Path(args[args.index('--plan')+1]);out=Path(args[args.index('--out')+1]);out.mkdir()
data=b'\\x00\\x00';digest=hashlib.sha256(data).hexdigest()
for mode in ('isolated','shared'):(out/(mode+'-m1-step0.bf16')).write_bytes(data)
report={'schema':'atlas-native-two-layer-scratch-replay-v1','passed':True,'plan_sha256':hashlib.sha256(plan.read_bytes()).hexdigest(),'executable_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'guards_passed':True,'input_unchanged':True,'output_sha256':{'m1-step0':digest},'ptx_sha256':PTX,'loaded_blas_sha256':{'libcublas-cpu-fixture':'e'*64,'libcublasLt-cpu-fixture':'f'*64}}
(out/'report.json').write_text(json.dumps(report))
print('CPU subprocess artifact fixture only; no GPU validation')
'''.replace('PTX',repr({n:v['sha256'] for n,v in modules.items()}))
            source+='\n'+'\n'.join('# '+(root/v['file']).read_text() for v in modules.values())
            binary.write_text(source);binary.chmod(0o700)
            binding=root/'binding.json';binding.write_text(json.dumps({'image':'cpu-fixture','build_source_sha256':'a'*64,'embedded_binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'modules':modules,'runtime_library_sha256':{'libcublas-cpu-fixture':'e'*64,'libcublasLt-cpu-fixture':'f'*64}}))
            receipt=ReplayIORouter(binary,plan,root/'record',root,binding,root).run()
            self.assertTrue(receipt['passed']);self.assertTrue((root/'record/stdout.log').is_file())
            self.assertEqual((root/'record/plan.json').read_bytes(),plan.read_bytes())
