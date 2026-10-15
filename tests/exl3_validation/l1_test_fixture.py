# SPDX-License-Identifier: AGPL-3.0-only
"""CPU fixtures using the actual TP-rank and fused mHC sidecar schemas."""
import hashlib
import json
import struct
import numpy as np
from parity_protocol import canonical, token_sha

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_json(path, value):
    path.write_text(json.dumps(value))

def bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]

def sidecar(tree, engine, row, stage):
    return tree[engine + '_root'] / ('ffn-L%d-rank%d-%06d-%s.json' % (tree['layer'], tree['rank'], row, stage))

def runtime_tree(root, *, layer=1, rank=0):
    source = {'model_sha256': 'a' * 64, 'ffn_hook_sha256': 'b' * 64, 'runner_sha256': 'c' * 64}
    cases = [{'id': 'longer', 'prompt_ids': [11, 12, 13], 'continuation_ids': [14]}]
    tree = {'layer': layer, 'rank': rank, 'rows': [0, 1], 'binding_path': root / 'binding.json'}
    head_bindings = []
    for engine in ('candidate', 'reference'):
        tree[engine + '_root'] = root / engine
        tree[engine + '_root'].mkdir()
        artifact = root / (engine + '-artifact')
        (artifact / 'responses').mkdir(parents=True)
        tree[engine + '_artifact'] = artifact
        captures = []
        if engine == 'reference':
            (artifact / 'heads').mkdir()
        for row in range(2):
            request = engine + '-req-' + str(row)
            response = 'longer-%06d.json' % row
            prefix = cases[0]['prompt_ids'] + cases[0]['continuation_ids'][:row]
            write_json(artifact / 'responses' / response, {'request': {'model': 'frozen-model', 'prompt': prefix, 'max_tokens': 1, 'temperature': 0, 'seed': 42, 'stream': False}, 'response': {'id': 'cmpl-' + request, 'choices': [{'text': 'captured'}]}})
            captures.append({'case': 'longer', 'step': row, 'prefix_sha256': hashlib.sha256(canonical(prefix)).hexdigest(), 'request_id': request, 'response': response})
            if engine == 'reference':
                stem = 'logits-%06d' % row
                raw = artifact / 'heads' / (stem + '.f32')
                np.arange(32, dtype='<f4').tofile(raw)
                metadata = {'dtype': 'f32-le', 'rows': 1, 'vocab': 32, 'file': raw.name}
                side = raw.with_suffix('.json')
                write_json(side, metadata)
                position = artifact / 'heads' / ('head-position-%06d.json' % row)
                write_json(position, {'head_capture': stem, 'rows': 1, 'context': {
                    'request_id': 'cmpl-' + request + '-0-%08x' % row,
                    'computed_before': 0, 'scheduled_tokens': len(prefix),
                    'logical_after': len(prefix), 'prefix_tokens': len(prefix),
                    'prefill_tokens': len(prefix), 'phase': 'terminal-prefill',
                    'discard': False, 'sampleable_terminal_prefix': True,
                    'runner_source_sha256': source['runner_sha256']}})
                captures[-1].update({'capture': '/remote/heads/' + side.name,
                                     'capture_metadata': metadata, 'raw_sha256': sha(raw),
                                     'row_sha256': sha(raw)})
                head_bindings.append({'head_position_file': str(position), 'head_position_sha256': sha(position),
                                      'raw_sidecar_file': str(side), 'raw_sidecar_sha256': sha(side),
                                      'response_sha256': sha(artifact / 'responses' / response)})
        prov = {'engine': 'atlas' if engine == 'candidate' else 'vllm', 'checkpoint_sha256': {'checkpoint': 'd' * 64}, 'checkpoint_revision': 'frozen', 'tokenizer_sha256': {'tokenizer': 'e' * 64}, 'config_sha256': {'config': 'f' * 64}, 'execution': {'tp': 2, 'ep': 1, 'activation_dtype': 'bf16', 'eager': True, 'speculative': False, 'prefix_caching': False}}
        write_json(artifact / 'manifest.json', {'schema': 'atlas-exl3-teacher-forced-v1', 'cases': cases, 'model': 'frozen-model', 'vocab': 32, 'tokens_sha256': token_sha(cases), 'captures': captures, 'request_ids': [c['request_id'] for c in captures], 'provenance': prov, 'request_policy': {'temperature': 0, 'seed': 42, 'max_tokens': 1}})
    write_json(tree['reference_root'] / 'hook-source-vllm.models.glm5next.nvidia.model-42.json', {'module': 'vllm.models.glm5next.nvidia.model', 'sha256': source['model_sha256'], 'hook_sha256': source['ffn_hook_sha256'], 'layers': [0, 1]})
    for row in range(2):
        m = 3 + row
        highway = np.full((m, 4, 2), 16256 + row, dtype='<u2')
        norm = np.full((m, 2), 16256 + row, dtype='<u2')
        post = np.full((m, 4), 0.25 + row, dtype='<f4')
        comb = np.full((m, 4, 4), 0.125 + row, dtype='<f4')
        core = np.full((m, 4096), 16256 + row, dtype='<u2')
        cand = {'incoming_highway': {'highway': highway}, 'attn_mhc_norm': {'normed': norm, 'highway': highway, 'post': post, 'comb': comb}, 'recurrent_output': {'core': core}, 'gated_norm': {'gated': core}, 'attention_reduced_output': {'output': norm}, 'ffn_mhc_norm': {'normed': norm, 'highway': highway, 'post': post, 'comb': comb}}
        binding = {'schema': 'atlas-mhc-boundary-binding-v1', 'rows': m, 'hc_mult': 4, 'sinkhorn_iters': 20, 'hc_eps_bits': bits(1e-06), 'rms_eps_bits': bits(1e-05), 'residual_precision': 'UnroundedPost', 'sequence_parallel': False, 'request_id': 'candidate-req-' + str(row)}
        for stage, values in cand.items():
            p = sidecar(tree, 'candidate', row, stage)
            records = []
            for name, array in values.items():
                filename = p.stem + '-' + name + '.bin'
                array.tofile(p.parent / filename)
                records.append({'name': name, 'dtype': 'f32-le' if array.dtype == np.dtype('<f4') else 'bf16-le', 'shape': list(array.shape), 'bytes': array.nbytes, 'file': filename})
            meta = {'schema': 'atlas-exl3-ffn-v1', 'layer': layer, 'tp_rank': rank, 'tp_world': 2, 'invocation': row, 'stage': stage, 'tensors': records}
            if stage in ('attn_mhc_norm', 'ffn_mhc_norm'):
                meta['binding'] = binding
            write_json(p, meta)
        context = {'request_id': 'cmpl-reference-req-' + str(row) + '-0-%08x' % row, 'computed_before': 0, 'scheduled_tokens': m, 'logical_after': m, 'prefix_tokens': m, 'prefill_tokens': m, 'phase': 'terminal-prefill', 'discard': False, 'sampleable_terminal_prefix': True, 'runner_source_sha256': source['runner_sha256']}
        attn = 2 * row if layer else row
        ffn = 2 * row + 1 if layer else row
        kind = 'fused' if layer else 'pre'
        refs = [(attn, 'mhc-' + kind + '-post', post), (attn, 'mhc-' + kind + '-comb', comb), (attn, 'mhc-' + kind + '-layer-input', norm), (attn, 'mhc-' + kind + ('-residual-output' if layer else '-input-x'), highway), (attn, 'mhc-' + kind + '-input-residual', highway ^ np.uint16(1)), (row, 'kda-core-output', core.reshape(1, m, 32, 128)), (row, 'kda-gated-norm', core.reshape(1, m, 32, 128)), (row, 'kda-o-proj', norm), (row, 'kda-o-proj-input0', core), (ffn, 'mhc-fused-layer-input', norm)]
        for invoke, stage, array in refs:
            p = sidecar(tree, 'reference', invoke, stage)
            array.tofile(p.with_suffix('.bin'))
            write_json(p, {'rank': rank, 'layer': layer, 'invocation': invoke, 'stage': stage, 'chunk_position': context, 'dtype': 'f32-le' if array.dtype == np.dtype('<f4') else 'bf16-le', 'shape': list(array.shape), 'file': p.with_suffix('.bin').name})
        for invoke, phase in [(attn, kind), (ffn, 'fused')]:
            policy = {'rms_eps': 1e-05, 'hc_pre_eps': 1e-06, 'hc_sinkhorn_eps': 1e-06, 'norm_eps': 1e-05, 'sinkhorn_repeat': 20, 'hc_post_mult_value': 1.0, 'dispatch': {'mhc_pre_op': {'value': 'bound.pre'}, 'mhc_fused_post_pre_op': {'value': 'bound.fused'}}, 'split_geometry': {'n_splits': 'unknown', 'n_splits_reason': 'internal argument', 'tile_n': 'unknown', 'tile_n_reason': 'internal argument'}}
            write_json(sidecar(tree, 'reference', invoke, 'mhc-' + phase + '-policy'), {'rank': rank, 'layer': layer, 'invocation': invoke, 'policy': policy, 'chunk_position': context})
    write_json(tree['binding_path'], {'schema': 'atlas-l1-bisect-inputs-v1', 'reference_head_provenance': head_bindings, 'reference_source': source, 'candidate_manifest_sha256': sha(tree['candidate_artifact'] / 'manifest.json'), 'reference_manifest_sha256': sha(tree['reference_artifact'] / 'manifest.json')})
    return tree
