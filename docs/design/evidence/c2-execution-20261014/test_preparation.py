# SPDX-License-Identifier: AGPL-3.0-only
"""Source binding, numerical neutrality and phase-ownership controls."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
import prepare
import run_neutrality


def runtime_binding():
    config = {'image_id': 'sha256:' + 'a' * 64, 'runtime_source': 'b' * 40,
              'source_snapshot_sha256': 'c' * 64, 'image_tag': 'atlas:src-' + 'c' * 12}
    build = {'schema': 'atlas-exl3-build-manifest-v1', 'git_commit': 'b' * 40,
             'source_snapshot_sha256': 'c' * 64, 'binaries': {'spark': {'sha256': 'd' * 64}}}
    return config, build


def test_explicit_source_binary_image_binding_rejects_placeholder_or_foreign_manifest():
    cfg, build = runtime_binding()
    assert prepare.bind_runtime(cfg, build)['binary_sha256'] == 'd' * 64
    with pytest.raises(ValueError, match='source identity'):
        prepare.bind_runtime(cfg, dict(build, git_commit='e' * 40))
    with pytest.raises(ValueError, match='image ID'):
        prepare.bind_runtime(dict(cfg, image_id='pending-build'), build)
    with pytest.raises(ValueError, match='tag snapshot'):
        prepare.bind_runtime(dict(cfg, image_tag='atlas:latest'), build)


def suite():
    cases = [{'id': f'case{i}', 'prompt_ids': [1], 'continuation_ids': [2, 3, 4]} for i in range(4)]
    prov = {'serve_flags': ['serve', 'model'], 'execution': {'kv_dtype': 'bf16'},
            'native_numerical_policy': {'source_identity': 'c' * 64, 'kv_cache_effective': 'fp8_g128'},
            'checkpoint_revision': 'checkpoint', 'checkpoint_sha256': {}, 'tokenizer_sha256': {}, 'config_sha256': {},
            'engine_build': {'source_sha256': 'c' * 64}}
    manifest = {'cases': cases, 'tokens_sha256': 'frozen', 'vocab': 8, 'request_policy': {}, 'thresholds': {},
                'provenance': prov, 'generated_text': {case['id']: ['x'] * 4 for case in cases}}
    arrays = {case['id']: np.arange(32, dtype=np.float32).reshape(4, 8) for case in cases}
    return manifest, arrays


def test_full_suite_compares_every_row_and_only_source_policy_identity_may_change():
    old = suite()
    new = copy.deepcopy(old)
    new[0]['provenance']['engine_build']['source_sha256'] = 'e' * 64
    new[0]['provenance']['native_numerical_policy']['source_identity'] = 'e' * 64
    assert run_neutrality.compare_suite(old, new)['passed']
    new[1]['case3'][3, 7] += 0.5
    result = run_neutrality.compare_suite(old, new)
    assert not result['passed'] and result['rows_byte_equal'] == 15
    new = copy.deepcopy(old)
    new[0]['provenance']['native_numerical_policy']['kv_cache_effective'] = 'bf16'
    with pytest.raises(ValueError, match='beyond source identity'):
        run_neutrality.compare_suite(old, new)


def test_descriptor_rewrite_preserves_numerical_flags_and_never_adds_diagnostic_markers():
    original = ['docker', 'create', '--env', 'ATLAS_EXL3_LOGITS_DIR=/tmp/capture', '--env', 'ATLAS_KDA_FLA_REFERENCE=1',
                '--volume', '/old:/tmp/capture', 'atlas:src-old', 'serve', 'model', '--kv-cache-dtype', 'fp8_g128']
    rewritten = prepare.rewrite_descriptor(original, 'atlas:src-new', Path('/fresh/logits'))
    assert rewritten[rewritten.index('serve'):] == original[original.index('serve'):]
    assert 'ATLAS_KDA_FLA_REFERENCE=1' in rewritten
    assert 'ATLAS_EXL3_LOGITS_DIR=/fresh/logits' in rewritten
    assert not any('DUMP_' in arg or 'SUBSTITUTE' in arg for arg in rewritten)
    assert original[original.index('serve') - 1] == 'atlas:src-old'


def test_strict_proof_consumption_rejects_boolean_pass_without_source_artifacts(tmp_path):
    with pytest.raises((ValueError, KeyError, FileNotFoundError)):
        run_neutrality.verify_proof({'passed': True}, {}, tmp_path)
