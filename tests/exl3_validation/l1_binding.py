# SPDX-License-Identifier: AGPL-3.0-only
"""Strict collection/request and mHC-policy bindings for the L1 diagnostic."""
import hashlib
import json
import math
from pathlib import Path
import re
import struct

from parity_protocol import canonical, read_raw, token_sha, validate_cases, validate_run_pair


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'regular JSON file required: ' + str(path))
    return json.loads(path.read_text())


def source_binding(reference_root, source):
    require(set(source) == {'model_sha256', 'ffn_hook_sha256', 'runner_sha256'}, 'reference source keys invalid')
    require(all(isinstance(v, str) and re.fullmatch('[0-9a-f]{64}', v) for v in source.values()),
            'reference source SHA256 required')
    records = list(Path(reference_root).glob('hook-source-vllm.models.glm5next.nvidia.model-*.json'))
    require(bool(records), 'reference source receipt missing')
    for path in records:
        value = read_json(path)
        require(value.get('module') == 'vllm.models.glm5next.nvidia.model'
                and value.get('sha256') == source['model_sha256']
                and value.get('hook_sha256') == source['ffn_hook_sha256']
                and value.get('layers') == [0, 1], 'reference source receipt differs')
    return {str(p): sha(p) for p in records}


def head_request(capture, provenance, request, vocab, source):
    """Bind the collector's exact head bytes/context to the HTTP request.

    Pinned completion serving appends prompt index 0; InputProcessor then
    appends eight random hex characters. The observed ID is read from a
    hash-pinned head position, never inferred by stripping a prefix/suffix.
    """
    for field in ('head_position_file', 'raw_sidecar_file'):
        require(isinstance(provenance[field], str) and Path(provenance[field]).is_absolute(),
                'explicit absolute head provenance path required')
    position_path = Path(provenance['head_position_file'])
    raw_path = Path(provenance['raw_sidecar_file'])
    require(sha(position_path) == provenance['head_position_sha256'], 'head position fingerprint differs')
    require(sha(raw_path) == provenance['raw_sidecar_sha256'], 'raw head sidecar fingerprint differs')
    position = read_json(position_path)
    metadata = read_json(raw_path)
    stem = Path(capture['capture']).stem
    require(re.fullmatch(r'logits-\d{6}', stem) and raw_path.stem == stem
            and position_path.name == 'head-position-' + stem.removeprefix('logits-') + '.json'
            and position.get('head_capture') == stem and type(position.get('rows')) is int
            and position['rows'] == metadata['rows'] == 1, 'head capture/index identity differs')
    require(metadata == capture['capture_metadata'] and metadata['vocab'] == vocab,
            'raw head metadata differs from collection')
    require(isinstance(metadata.get('file'), str) and Path(metadata['file']).name == metadata['file'],
            'raw head payload filename must be local')
    payload = raw_path.parent / metadata['file']
    require(payload.is_file() and not payload.is_symlink(), 'regular raw head payload required')
    require(sha(payload) == capture['raw_sha256'], 'raw head payload fingerprint differs')
    row, _ = read_raw(raw_path)
    require(hashlib.sha256(row.tobytes()).hexdigest() == capture['row_sha256'],
            'raw head row fingerprint differs')
    context = position['context']
    gpu_id = context['request_id']
    require(isinstance(gpu_id, str) and re.fullmatch(re.escape(request['response_id']) + r'-0-[0-9a-f]{8}', gpu_id),
            'GPU request ID differs from pinned single-prompt response/index/random-suffix contract')
    bound = dict(request, gpu_request_id=gpu_id)
    reference_context({'chunk_position': context}, bound, source)
    return bound, {str(p): sha(p) for p in (position_path, raw_path, payload)}


def registered_requests(candidate_artifact, reference_artifact, binding_path, reference_root):
    binding = read_json(binding_path)
    require(binding.get('schema') == 'atlas-l1-bisect-inputs-v1', 'L1 input binding schema invalid')
    manifests, requests, files = {}, {}, {}
    source = binding['reference_source']
    heads = binding['reference_head_provenance']
    require(isinstance(heads, list), 'explicit reference head provenance list required')
    for engine, root in [('candidate', candidate_artifact), ('reference', reference_artifact)]:
        root = Path(root)
        path = root / 'manifest.json'
        require(sha(path) == binding[engine + '_manifest_sha256'], 'manifest fingerprint differs: ' + engine)
        manifest = read_json(path)
        require(manifest['schema'] == 'atlas-exl3-teacher-forced-v1', 'collection schema invalid')
        require(manifest['provenance']['engine'] == ('atlas' if engine == 'candidate' else 'vllm'),
                'collection engine differs')
        execution = manifest['provenance']['execution']
        require(execution['tp'] == 2 and execution['ep'] == 1 and execution['eager'] is True
                and execution['speculative'] is False and execution['prefix_caching'] is False,
                'L1 requires pure TP2/EP1 eager non-speculative cold-prefix collection')
        require(manifest['request_policy'] == {'temperature': 0, 'seed': 42, 'max_tokens': 1},
                'L1 request policy differs from the frozen teacher-forced profile')
        cases = validate_cases(manifest['cases'])
        require(token_sha(cases) == manifest['tokens_sha256'], 'collection token hash differs')
        rows = [(case['id'], step, case['prompt_ids'] + case['continuation_ids'][:step])
                for case in cases for step in range(len(case['continuation_ids']) + 1)]
        captures = manifest['captures']
        require(len(captures) == len(rows) == len(manifest['request_ids']), 'collection request count differs')
        require(len(set(manifest['request_ids'])) == len(rows), 'request IDs must be unique')
        if engine == 'reference':
            require(len(heads) == len(rows), 'reference head provenance count differs')
        requests[engine] = []
        for index, ((case, step, prefix), capture) in enumerate(zip(rows, captures)):
            prefix_sha = hashlib.sha256(canonical(prefix)).hexdigest()
            require(capture['case'] == case and capture['step'] == step
                    and capture['prefix_sha256'] == prefix_sha, 'collection prefix/order differs')
            request_id = capture['request_id']
            require(isinstance(request_id, str) and request_id and request_id == manifest['request_ids'][index],
                    'collection request identity differs')
            response_name = capture['response']
            require(response_name == '%s-%06d.json' % (case, step),
                    'response filename differs from registered case/step')
            response_path = root / 'responses' / response_name
            if engine == 'reference':
                require(sha(response_path) == heads[index]['response_sha256'], 'reference HTTP response fingerprint differs')
            envelope = read_json(response_path)
            require(envelope['request'] == {
                'model': manifest['model'], 'prompt': prefix, 'max_tokens': 1,
                'temperature': 0, 'seed': 42, 'stream': False},
                'recorded HTTP request differs from prescribed prefix/model/policy')
            response = envelope['response']
            require('error' not in response and bool(response.get('choices')), 'HTTP completion failed')
            response_id = response['id']
            require(isinstance(response_id, str) and response_id, 'response request ID missing')
            request = {'case': case, 'step': step, 'prefix_sha256': prefix_sha,
                       'prefix_tokens': len(prefix), 'request_id': request_id, 'response_id': response_id}
            if engine == 'reference':
                require(max(prefix) < manifest['vocab'], 'prefix token outside bound head vocabulary')
                request, evidence = head_request(capture, heads[index], request, manifest['vocab'], source)
                files.update(evidence)
            requests[engine].append(request)
            files[str(response_path)] = sha(response_path)
        manifests[engine] = manifest
        files[str(path)] = sha(path)
    validate_run_pair(manifests['reference'], manifests['candidate'],
                      ['cases', 'tokens_sha256', 'request_policy', 'vocab'])
    require(len({request['gpu_request_id'] for request in requests['reference']}) == len(requests['reference']),
            'observed GPU request IDs must be unique')
    files.update(source_binding(reference_root, source))
    return requests, source, {'binding_sha256': sha(binding_path), 'file_sha256': files,
                              'reference_source': source}


def reference_context(metadata, request, source):
    context = metadata.get('chunk_position')
    require(isinstance(context, dict), 'reference chunk binding missing')
    expected = {'request_id': request['gpu_request_id'], 'computed_before': 0,
                'scheduled_tokens': request['prefix_tokens'], 'logical_after': request['prefix_tokens'],
                'prefix_tokens': request['prefix_tokens'], 'prefill_tokens': request['prefix_tokens'],
                'phase': 'terminal-prefill', 'discard': False, 'sampleable_terminal_prefix': True,
                'runner_source_sha256': source['runner_sha256']}
    require(all(type(context.get(k)) is type(v) and context[k] == v for k, v in expected.items()),
            'reference request/prefix/runner context differs')


def f32_bits(value):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0, 'positive finite policy value required')
    try:
        return struct.unpack('<I', struct.pack('<f', value))[0]
    except OverflowError as exc:
        raise ValueError('policy value does not fit F32') from exc


def policy_record(root, kind, invocation, layer, rank, request, source):
    path = Path(root) / ('ffn-L%d-rank%d-%06d-mhc-%s-policy.json' % (layer, rank, invocation, kind))
    record = read_json(path)
    require(all(type(record.get(k)) is int and record[k] == v
                for k, v in {'layer': layer, 'rank': rank, 'invocation': invocation}.items()),
            'reference policy identity differs')
    reference_context(record, request, source)
    policy = record['policy']
    for field in ('rms_eps', 'hc_pre_eps', 'hc_sinkhorn_eps', 'norm_eps', 'hc_post_mult_value'):
        f32_bits(policy[field])
    require(type(policy['sinkhorn_repeat']) is int and policy['sinkhorn_repeat'] > 0,
            'reference policy sinkhorn iterations invalid')
    for field in ('mhc_pre_op', 'mhc_fused_post_pre_op'):
        label = policy['dispatch'][field]
        require(isinstance(label.get('value'), str) and bool(label['value']), 'mHC dispatch identity missing')
        if label['value'] == 'unknown':
            require(isinstance(label.get('reason'), str) and bool(label['reason']),
                    'unknown dispatch requires reason')
    for field in ('n_splits', 'tile_n'):
        geometry = policy['split_geometry']
        value = geometry[field]
        reason = geometry.get(field + '_reason')
        require(type(value) is int and value > 0
                or value == 'unknown' and isinstance(reason, str) and bool(reason),
                'split geometry requires a positive integer or unknown with reason')
    return record


def candidate_boundary(metadata, shape, request, policy):
    binding = metadata.get('binding')
    require(isinstance(binding, dict) and binding.get('schema') == 'atlas-mhc-boundary-binding-v1',
            'candidate boundary binding missing')
    require(binding['request_id'] == request['request_id'] and type(binding['rows']) is int
            and binding['rows'] == request['prefix_tokens'] == shape[0], 'candidate request/prefix binding differs')
    require(binding['sequence_parallel'] is False and binding['residual_precision'] in ('Rounded', 'UnroundedPost'),
            'candidate precision/sequence-parallel policy invalid')
    require(type(binding['hc_mult']) is int and binding['hc_mult'] > 0, 'candidate stream count invalid')
    require(type(binding['sinkhorn_iters']) is int and binding['sinkhorn_iters'] == policy['sinkhorn_repeat'],
            'candidate/reference sinkhorn policy differs')
    require(type(binding['hc_eps_bits']) is int and type(binding['rms_eps_bits']) is int
            and binding['hc_eps_bits'] == f32_bits(policy['hc_pre_eps']) == f32_bits(policy['hc_sinkhorn_eps'])
            and binding['rms_eps_bits'] == f32_bits(policy['norm_eps']) == f32_bits(policy['rms_eps']),
            'candidate/reference epsilon policy differs')
    return binding
