# SPDX-License-Identifier: AGPL-3.0-only
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
from norm_replay_artifacts import prepare


class Tests(unittest.TestCase):
    def fixtures(self, root):
        cand, ref, request = [root / name for name in ('cand', 'ref', 'request')]
        for path in (cand, ref, request): path.mkdir()
        stem = 'ffn-L0-rank0-000000-'
        def tensor(where, stage, name, shape, candidate=False):
            data = np.full(shape, 0x3f80, dtype='<u2')
            file = stem + stage + ('-' + name if candidate else '') + '.bin'
            (where / file).write_bytes(data.tobytes())
            item = {'dtype': 'bf16-le', 'shape': list(shape), 'file': file}
            side = {'layer': 0, 'invocation': 0, 'stage': stage}
            side.update({'schema': 'atlas-exl3-ffn-v1', 'tp_rank': 0, 'tp_world': 2}
                        if candidate else {'rank': 0, 'source': 'installed-reference-forward-hook'})
            side.update({'tensors': [dict(item, name=name)]} if candidate else item)
            (where / (stem + stage + '.json')).write_text(json.dumps(side))
            return side
        for stage, name, shape in [('recurrent_output', 'core', (2,32,128)),
                                    ('recurrent_inputs', 'g2', (2,4096)),
                                    ('gated_norm', 'gated', (2,4096)),
                                    ('gated_norm_binding', 'weight', (128,))]:
            side = tensor(cand, stage, name, shape, True)
            if stage == 'gated_norm_binding':
                side['binding'] = {'schema': 'atlas-kda-gated-norm-binding-v1',
                    'eps_f32_bits': struct.unpack('<I', struct.pack('<f', 1e-5))[0],
                    'tokens': 2, 'heads': 32, 'dim': 128, 'kernel_handle': 42,
                    'kernel_module': 'kda', 'kernel_symbol': 'kda_sigmoid_gated_rms_norm',
                    'grid': [64,1,1], 'block': [128,1,1], 'activation': 'sigmoid',
                    'affine': True, 'bias': False, 'residual': False, 'prenorm': False,
                    'input_dtype': 'bf16', 'gate_dtype': 'bf16', 'weight_dtype': 'bf16',
                    'accumulator_dtype': 'f32', 'output_dtype': 'bf16'}
                (cand / (stem + stage + '.json')).write_text(json.dumps(side))
        for stage, shape in [('input-x',(1,2,32,128)), ('input-g',(1,2,32,128)),
                             ('weight',(128,)), ('output',(1,2,32,128))]:
            tensor(ref, 'kda-norm-bound-' + stage, None, shape)
        source_bytes=b'# exact installed source fixture\n'
        source_digest=hashlib.sha256(source_bytes).hexdigest()
        source_root=ref/'norm-installed-sources';source_root.mkdir()
        (source_root/(source_digest+'-fixture.py')).write_bytes(source_bytes)
        identity = {'module': 'fixture', 'qualname': 'forward_cuda', 'source_file': '/fixture.py',
                    'source_sha256': source_digest, 'first_line': 1}
        binding = {'schema': 'reference-norm-runtime-binding-v1', 'eps': 1e-5,
                   'activation': 'sigmoid', 'elementwise_affine': True, 'hidden_size':128,
                   'bias_present': False, 'call': {'prenorm':False, 'residual_present':False,
                                                 'residual_in_fp32':False},
                   'dispatch':identity, 'class':identity, 'source_targets':{'kernel':identity},
                   'runtime_policy':{'compiler':'fixture'}, 'tensors':{
                       name:{'shape':list(shape),'dtype':'torch.bfloat16','device':'cuda:0'}
                       for name,shape in [('x',(1,2,32,128)),('g',(1,2,32,128)),('weight',(128,))]}}
        (ref / (stem + 'kda-norm-runtime-binding.json')).write_text(json.dumps(binding))
        replay = {'schema':'reference-norm-self-replay-v1','self_reproduction_exact':True,
                  'controls':{'input-zero':True,'gate-zero':True,'weight-zero':True},
                  'fresh_operand_clones_per_trial':True,'independent_parameter_map':True}
        (ref / (stem + 'kda-norm-self-replay.json')).write_text(json.dumps(replay))
        launch = {'schema':'reference-norm-compiled-launch-v1','launch_index':0,
                  'launch':{'eps':1e-5,'T':64,'D':128,'BD':128,'BT':32,
                            'ACTIVATION':'sigmoid','IS_RMS_NORM':True,'num_warps':4},
                  'compiled_metadata':{'enable_fp_fusion':True,'launch_pdl':False},
                  'artifacts':{}, 'sources':identity}
        for kind in ('ptx','cubin'):
            payload = ('fixture-' + kind).encode()
            stage = 'kda-norm-launch-000-' + kind
            (ref / (stem + stage + '.bin')).write_bytes(payload)
            launch['artifacts'][kind] = {'stage':stage,'bytes':len(payload),
                                        'sha256':hashlib.sha256(payload).hexdigest()}
        (ref / (stem + 'kda-norm-launch-000.json')).write_text(json.dumps(launch))
        (request / 'request-000000.json').write_text(json.dumps({'index':0,'prefix_tokens':2,'request':{
            'model':'explicit-fixture','prompt':[1,2],'seed':42,'temperature':0,
            'max_tokens':1,'stream':False}}))
        return cand, ref, request

    def run_fixture(self, root):
        cand,ref,request = self.fixtures(root)
        return prepare(cand,ref,request,root/'out',0,0)

    def test_prepares_bound_bf16_arrays_and_hashed_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            manifest=self.run_fixture(root)
            self.assertTrue(manifest['equal_operands']['core'])
            self.assertTrue(manifest['equal_operands']['gate'])
            self.assertTrue(manifest['equal_operands']['weight'])
            self.assertTrue(manifest['epsilon_f32_equal'])
            self.assertEqual(np.load(root/'out'/'core.npy').shape,(2,32,128))
            self.assertEqual(np.load(root/'out'/'gate.npy').shape,(2,4096))
            self.assertEqual(len(manifest['source_files']),23)

    def test_cross_engine_constant_and_input_inequality_remain_reviewable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cand,ref,request=self.fixtures(root)
            weight=ref/'ffn-L0-rank0-000000-kda-norm-bound-weight.bin'
            value=np.frombuffer(weight.read_bytes(),dtype='<u2').copy()
            value[0]=0x4000
            weight.write_bytes(value.tobytes())
            side=ref/'ffn-L0-rank0-000000-kda-norm-runtime-binding.json'
            binding=json.loads(side.read_text());binding['eps']=2e-5
            side.write_text(json.dumps(binding))
            launch_path=ref/'ffn-L0-rank0-000000-kda-norm-launch-000.json'
            launch=json.loads(launch_path.read_text());launch['launch']['eps']=2e-5
            launch_path.write_text(json.dumps(launch))
            manifest=prepare(cand,ref,request,root/'out',0,0)
            self.assertFalse(manifest['equal_operands']['weight'])
            self.assertFalse(manifest['epsilon_f32_equal'])

    def test_invalid_dtype_layout_epsilon_replay_and_controls_reject(self):
        for problem in ('dtype','layout','epsilon','replay','controls','blob','path','request','rank','world','source'):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);cand,ref,request=self.fixtures(root)
                path = cand/'ffn-L0-rank0-000000-recurrent_output.json'
                if problem in ('rank','world'):
                    value=json.loads(path.read_text())
                    value['tp_rank' if problem=='rank' else 'tp_world'] = 1
                elif problem in ('dtype','layout','path'):
                    value=json.loads(path.read_text())
                    value['tensors'][0][{'dtype':'dtype','layout':'shape','path':'file'}[problem]]= {
                        'dtype':'f32-le','layout':[2,128,32],'path':'../outside.bin'}[problem]
                elif problem == 'epsilon':
                    path=cand/'ffn-L0-rank0-000000-gated_norm_binding.json'
                    value=json.loads(path.read_text());value['binding']['eps_f32_bits']=0
                elif problem in ('replay','controls'):
                    path=ref/'ffn-L0-rank0-000000-kda-norm-self-replay.json'
                    value=json.loads(path.read_text())
                    if problem=='replay':value['self_reproduction_exact']=False
                    else:value['controls']['gate-zero']=False
                elif problem=='source':
                    next((ref/'norm-installed-sources').iterdir()).write_bytes(b'tampered source')
                    value=None
                elif problem=='blob':
                    (ref/'ffn-L0-rank0-000000-kda-norm-launch-000-cubin.bin').write_bytes(b'changed')
                    value=None
                else:
                    path=request/'request-000000.json';value=json.loads(path.read_text())
                    value['request']['seed']=43
                if value is not None:path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):prepare(cand,ref,request,root/'out',0,0)
                self.assertFalse((root/'out').exists())

if __name__=='__main__':unittest.main()
