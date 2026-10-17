# SPDX-License-Identifier: AGPL-3.0-only
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from native_ptx import REQUIRED, validate_binding, load_module


class EmbeddedPtxTests(unittest.TestCase):
    def fixture(self, root):
        common = root/'common'; common.mkdir()
        modules = {}
        for module, contract in REQUIRED.items():
            payload = '\n'.join('.visible .entry '+entry+'() {}' for entry in contract['entries'])
            path = root/(module+'.ptx'); path.write_text(payload)
            sources = {}
            for name in contract['sources']:
                source = common/name
                if not source.exists(): source.write_text('// exact frozen '+name)
                sources[name] = hashlib.sha256(source.read_bytes()).hexdigest()
            modules[module] = {'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                               'sources':sources,'contained_in_binary':True}
        manifest = {'image':'immutable-image@sha256:test','build_source_sha256':'a'*64,
                    'embedded_binary_sha256':'b'*64,'modules':modules}
        binding = root/'binding.json'; binding.write_text(json.dumps(manifest))
        return common,binding,manifest

    def test_all_native_modules_load_exact_ptx_path_without_nvrtc(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);common,binding,_=self.fixture(root)
            value=validate_binding(root,binding,common)
            seen=[]
            class Backend:
                def RawModule(self,**kwargs):
                    seen.append(kwargs);return kwargs
            for module in REQUIRED: load_module(Backend(),common,module,root,value)
            self.assertEqual(seen,[{'path':str(root/(module+'.ptx'))} for module in REQUIRED])

    def test_missing_payload_changed_source_hash_export_and_binary_binding_fail_closed(self):
        for mutation in ('missing','source','ptx','export','binary','traversal'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);common,binding,manifest=self.fixture(root)
                item=manifest['modules']['exl3_linear_mma'];path=root/item['file']
                if mutation=='missing':path.unlink()
                elif mutation=='source':(common/'exl3_mma.cuh').write_text('// changed')
                elif mutation=='ptx':path.write_text('// changed')
                elif mutation=='export':
                    path.write_text('.visible .entry wrong() {}');item['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
                elif mutation=='binary':item['contained_in_binary']=False
                else:item['file']='../escape.ptx'
                binding.write_text(json.dumps(manifest))
                with self.assertRaises((ValueError,FileNotFoundError)):validate_binding(root,binding,common)

    def test_existing_source_mode_remains_explicit_nvrtc(self):
        with tempfile.TemporaryDirectory() as tmp:
            common=Path(tmp);(common/'exl3.cu').write_text('// source')
            class Backend:
                def RawModule(self,**kwargs):return kwargs
            value=load_module(Backend(),common,'exl3')
            self.assertEqual(value['code'],'// source');self.assertIn('--fmad=false',value['options'])

if __name__=='__main__':unittest.main()
