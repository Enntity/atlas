# SPDX-License-Identifier: AGPL-3.0-only
"""Prepare strict BF16 replay fixtures from runtime-bound layer-zero dumps."""
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path, hashes):
    _require(path.is_file() and not path.is_symlink(), f'Regular local artifact required: {path}')
    data = path.read_bytes()
    hashes[str(path)] = hashlib.sha256(data).hexdigest()
    return data


def _json(path, hashes):
    value = json.loads(_read(path, hashes))
    _require(isinstance(value, dict), f'Object sidecar required: {path}')
    return value


def _tensor(root, stem, stage, name, rank, invocation, hashes):
    side = _json(root / (stem + stage + '.json'), hashes)
    # Native FFN router names its TP owner tp_rank; installed reference
    # hooks name it rank. These are separate explicit producer contracts.
    owner = 'rank' if name is None else 'tp_rank'
    _require(all(side.get(key) == value and type(side.get(key)) is type(value) for key, value in
                 (('layer', 0), (owner, rank), ('invocation', invocation), ('stage', stage))),
             'Tensor sidecar invocation/stage binding differs')
    if name is None:
        _require(side.get('source') == 'installed-reference-forward-hook',
                 'Reference tensor producer binding differs')
        item = side
    else:
        _require(side.get('schema') == 'atlas-exl3-ffn-v1' and
                 type(side.get('tp_world')) is int and side['tp_world'] == 2,
                 'Native tensor schema/TP world binding differs')
        items = side['tensors']
        selected = [item for item in items if item['name'] == name]
        _require(len(selected) == 1, f'Unique candidate tensor {name} required')
        item = selected[0]
    _require(item['dtype'] == 'bf16-le', 'Only explicit little-endian BF16 capture is accepted')
    shape = item['shape']
    _require(isinstance(shape, list) and shape and
             all(type(dim) is int and dim > 0 for dim in shape), 'Invalid tensor geometry')
    filename = item['file']
    _require(isinstance(filename, str) and filename not in ('', '.', '..') and
             Path(filename).name == filename, 'Tensor filename must be local basename')
    raw = _read(root / filename, hashes)
    _require(len(raw) == math.prod(shape) * 2, 'Capture byte count does not match shape')
    value = np.frombuffer(raw, dtype='<u2').reshape(shape).copy()
    _require(not np.any((value & 0x7f80) == 0x7f80), 'Finite BF16 capture required')
    return value, side


def _identity(value, source_root, hashes):
    _require(isinstance(value, dict), 'Source identity object required')
    for key in ('module', 'qualname', 'source_file'):
        _require(isinstance(value.get(key), str) and value[key], 'Installed source identity incomplete')
    digest = value.get('source_sha256')
    _require(isinstance(digest, str) and len(digest) == 64 and
             all(char in '0123456789abcdef' for char in digest), 'Source hash missing or invalid')
    basename = Path(value['source_file']).name
    _require(basename not in ('', '.', '..'), 'Installed source basename invalid')
    path = source_root / (digest + '-' + basename)
    raw = _read(path, hashes)
    _require(hashlib.sha256(raw).hexdigest() == digest,
             'Archived installed source bytes differ from runtime identity')
    for wrapper in value.get('wrappers', []):
        _require(wrapper['via'] in ('fn', '__wrapped__'), 'Unknown source wrapper binding')
        _identity(wrapper['identity'], source_root, hashes)


def _f32bits(epsilon):
    _require(type(epsilon) in (float, int) and math.isfinite(epsilon) and epsilon > 0,
             'Finite positive bound epsilon required')
    try:
        raw = struct.pack('<f', epsilon)
    except OverflowError as error:
        raise ValueError('Bound epsilon overflows f32') from error
    _require(struct.unpack('<f', raw)[0] > 0, 'Bound epsilon underflows f32')
    return struct.unpack('<I', raw)[0]


def _reference_layout(value, tokens):
    permitted = ((1, tokens, 32, 128), (tokens, 32, 128), (tokens * 32, 128))
    _require(value.shape in permitted, 'Unregistered reference norm layout')
    return value.reshape(tokens, 32, 128).copy()


def prepare(candidate_root, reference_root, request_root, out, rank, invocation):
    """Validate captures before creating immutable .npy outputs and fixture.json.

    Array values remain uint16 BF16 bit patterns. Only singleton-batch removal
    and contiguous head flattening are permitted; dimensions never transpose.
    Input/constant differences are retained as evidence rather than rejected.
    """
    _require(type(rank) is int and rank in (0, 1) and type(invocation) is int and invocation >= 0,
             'Explicit rank and nonnegative invocation required')
    cand, ref, request_root, out = map(Path, (candidate_root, reference_root, request_root, out))
    _require(not out.exists(), 'Replay output must be a fresh directory')
    hashes = {}
    stem = f'ffn-L0-rank{rank}-{invocation:06d}-'
    request = _json(request_root / f'request-{invocation:06d}.json', hashes)
    payload = request['request']
    _require(isinstance(payload['model'], str) and payload['model'], 'Explicit model alias required')
    prompt = payload['prompt']
    _require(isinstance(prompt, list) and prompt and all(type(v) is int and v >= 0 for v in prompt),
             'Explicit frozen token IDs required')
    tokens = len(prompt)
    _require(type(request['index']) is int and type(request['prefix_tokens']) is int and
             request['index'] == invocation and request['prefix_tokens'] == tokens,
             'Request prefix/invocation binding differs')
    _require(type(payload['seed']) is int and payload['seed'] == 42 and
             type(payload['temperature']) in (int, float) and payload['temperature'] == 0 and
             type(payload['max_tokens']) is int and
             payload['max_tokens'] == 1 and payload['stream'] is False, 'Registered request policy differs')
    arrays = {}
    for stage, name, key, shape in (
        ('recurrent_output', 'core', 'core', (tokens,32,128)),
        ('recurrent_inputs', 'g2', 'gate', (tokens,4096)),
        ('gated_norm', 'gated', 'expected', (tokens,4096)),
        ('gated_norm_binding', 'weight', 'weight', (128,)),
    ):
        value, side = _tensor(cand, stem, stage, name, rank, invocation, hashes)
        _require(value.shape == shape, f'Candidate {key} geometry differs')
        arrays[key] = value.reshape(tokens,32,128) if key == 'expected' else value
        if key == 'weight':
            candidate_binding = side['binding']
    binding = candidate_binding
    _require(binding['schema'] == 'atlas-kda-gated-norm-binding-v1', 'Candidate binding schema differs')
    for key, expected in {'tokens':tokens,'heads':32,'dim':128,'kernel_module':'kda',
                           'kernel_symbol':'kda_sigmoid_gated_rms_norm','grid':[tokens*32,1,1],
                           'block':[128,1,1],'activation':'sigmoid','affine':True,'bias':False,
                           'residual':False,'prenorm':False,'input_dtype':'bf16','gate_dtype':'bf16',
                           'weight_dtype':'bf16','accumulator_dtype':'f32','output_dtype':'bf16'}.items():
        _require(binding[key] == expected and type(binding[key]) is type(expected),
                 f'Candidate dispatch binding differs: {key}')
    _require(type(binding['kernel_handle']) is int and binding['kernel_handle'] > 0,
             'Candidate kernel handle missing')
    cbits = binding['eps_f32_bits']
    _require(type(cbits) is int and 0 <= cbits <= 0xffffffff, 'Invalid epsilon f32 bits')
    cepsilon = struct.unpack('<f', struct.pack('<I', cbits))[0]
    _require(_f32bits(cepsilon) == cbits, 'Candidate epsilon f32 binding differs')
    raw_shapes = {}
    for stage, key in (('input-x','core'),('input-g','gate'),('weight','weight'),('output','expected')):
        value, _ = _tensor(ref, stem, 'kda-norm-bound-' + stage, None, rank, invocation, hashes)
        raw_shapes[key] = list(value.shape)
        if key == 'weight':
            _require(value.shape == (128,), 'Reference resident norm weight geometry differs')
            arrays['reference-weight'] = value
        else:
            mapped = _reference_layout(value, tokens)
            arrays['reference-' + key] = mapped.reshape(tokens,4096) if key == 'gate' else mapped
    reference_binding = _json(ref / (stem + 'kda-norm-runtime-binding.json'), hashes)
    rb = reference_binding
    _require(rb['schema'] == 'reference-norm-runtime-binding-v1' and
             rb['activation'] == 'sigmoid' and rb['elementwise_affine'] is True and
             rb['hidden_size'] == 128 and rb['bias_present'] is False,
             'Reference norm arithmetic binding differs')
    _require(rb['call'] == {'prenorm':False,'residual_in_fp32':False,'residual_present':False} and
             all(type(v) is bool for v in rb['call'].values()), 'Unsupported reference call flags')
    rbits = _f32bits(rb['eps'])
    for key, operand in (('x','core'),('g','gate'),('weight','weight')):
        info = rb['tensors'][key]
        _require(info['shape'] == raw_shapes[operand] and info['dtype'] == 'torch.bfloat16',
                 'Reference runtime tensor metadata differs from serialized capture')
    source_root = ref / 'norm-installed-sources'
    _require(source_root.is_dir() and not source_root.is_symlink(), 'Installed source capsule required')
    for identity in (rb['class'],rb['dispatch'],*rb['source_targets'].values()):
        _identity(identity, source_root, hashes)
    if rb.get('fallback_dispatch') is not None:
        _identity(rb['fallback_dispatch'], source_root, hashes)
    _require(rb['source_targets'] and rb['runtime_policy'], 'Reference runtime/source binding missing')
    replay = _json(ref / (stem + 'kda-norm-self-replay.json'), hashes)
    _require(replay['schema'] == 'reference-norm-self-replay-v1' and
             replay['self_reproduction_exact'] is True and
             replay['fresh_operand_clones_per_trial'] is True and
             replay['independent_parameter_map'] is True, 'Reference self reproduction failed')
    _require(replay['controls'] == {'input-zero':True,'gate-zero':True,'weight-zero':True} and
             all(type(v) is bool for v in replay['controls'].values()), 'Reference engagement controls failed')
    launch = _json(ref / (stem + 'kda-norm-launch-000.json'), hashes)
    _require(launch['schema'] == 'reference-norm-compiled-launch-v1' and launch['launch_index'] == 0,
             'Original reference launch binding required')
    args = launch['launch']
    _require(_f32bits(args['eps']) == rbits and args['T'] == tokens*32 and args['D'] == 128 and
             args['BD'] == 128 and args['ACTIVATION'] == 'sigmoid' and args['IS_RMS_NORM'] is True and
             type(args['BT']) is int and args['BT'] > 0 and type(args['num_warps']) is int and
             args['num_warps'] > 0, 'Reference launch operands/config binding differs')
    _require(type(launch['compiled_metadata']['enable_fp_fusion']) is bool and
             type(launch['compiled_metadata']['launch_pdl']) is bool, 'Effective compile/PDL binding missing')
    _identity(launch['sources'], source_root, hashes)
    for kind in ('ptx','cubin'):
        artifact = launch['artifacts'][kind]
        _require(artifact['stage'] == f'kda-norm-launch-000-{kind}', 'Compiled artifact stage differs')
        data = _read(ref / (stem + artifact['stage'] + '.bin'), hashes)
        _require(len(data) == artifact['bytes'] and hashlib.sha256(data).hexdigest() == artifact['sha256'],
                 'Compiled artifact bytes/hash binding differs')
    manifest = {'schema':'norm-replay-fixtures-v1','rank':rank,'invocation':invocation,'tokens':tokens,
                'request':request,'candidate_binding':binding,'reference_binding':rb,
                'reference_launch':launch,'reference_self_replay':replay,
                'candidate_epsilon_f32':cepsilon,'candidate_epsilon_f32_bits':cbits,
                'reference_epsilon':rb['eps'],'reference_epsilon_f32_bits':rbits,
                'epsilon_f32_equal':cbits == rbits,'reference_source_shapes':raw_shapes,
                'equal_operands':{key:bool(np.array_equal(arrays[key],arrays['reference-'+key]))
                                  for key in ('core','gate','weight')},'source_files':hashes,'fixtures':{}}
    out.mkdir(parents=True, exist_ok=False)
    for name,value in arrays.items():
        path = out / (name + '.npy')
        np.save(path,value,allow_pickle=False)
        manifest['fixtures'][name] = {'file':path.name,'shape':list(value.shape),'dtype':'bf16-le',
                                     'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (out/'fixture.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest
