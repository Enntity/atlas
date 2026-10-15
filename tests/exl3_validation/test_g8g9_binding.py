# SPDX-License-Identifier: AGPL-3.0-only
"""CPU tests for the G8/G9 provenance binding receipt (no ssh, no GPU)."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import g8g9_binding
from image_cycle import sha


class FakeRunner:
    def __init__(self, lines, rc=0):
        self.lines = lines
        self.rc = rc
        self.calls = []

    def run(self, argv, *, env=None, timeout):
        self.calls.append(list(argv))
        return {'rc': self.rc, 'stdout': '\n'.join(self.lines).encode(),
                'stderr': b''}


class CollectTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='atlas-g8g9-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = {'schema': 'atlas-exl3-g8g9-collect-v1',
                       'host': 'colum@10.50.0.10',
                       'jump': 'colum@192.168.1.50',
                       'image': 'glm53-flash-mikeroz:dt-carry20',
                       'entries': [
                           {'label': 'exl3-ext', 'path': '/opt/plugin/exllamav3_ext.so'},
                           {'label': 'vllm-c', 'path': '/usr/lib/python3/dist-packages/vllm/_C.so'}]}
        self.config_path = self.root / 'collect.json'
        self.write()

    def write(self):
        self.config_path.write_text(json.dumps(self.config))

    def collect(self, lines, rc=0):
        out = self.root / 'inventory.json'
        runner = FakeRunner(lines, rc)
        return runner, g8g9_binding.collect(self.config_path, out, runner)

    def test_collect_hashes_every_explicit_path(self):
        lines = ['a' * 64 + '  /opt/plugin/exllamav3_ext.so',
                 'b' * 64 + '  /usr/lib/python3/dist-packages/vllm/_C.so']
        runner, inventory = self.collect(lines)
        self.assertEqual(inventory['schema'], 'atlas-exl3-g8g9-inventory-v1')
        self.assertEqual([entry['label'] for entry in inventory['entries']],
                         ['exl3-ext', 'vllm-c'])
        self.assertEqual(inventory['entries'][0]['sha256'], 'a' * 64)
        command = runner.calls[0][-1]
        self.assertIn('docker run --rm --entrypoint /bin/sh', command)
        self.assertIn('sha256sum "$f"', command)

    def test_missing_binary_fails_the_collection(self):
        lines = ['a' * 64 + '  /opt/plugin/exllamav3_ext.so']
        with self.assertRaisesRegex(ValueError, 'missing inside image'):
            self.collect(lines)

    def test_nonzero_exit_fails_with_stderr(self):
        with self.assertRaisesRegex(ValueError, 'sha256sum failed'):
            self.collect([], rc=1)

    def test_unsafe_path_rejected_before_ssh(self):
        self.config['entries'][0]['path'] = "/opt/evil; rm -rf /"
        self.write()
        with self.assertRaisesRegex(ValueError, 'shell-safe'):
            self.collect([])

    def test_duplicate_label_rejected(self):
        self.config['entries'].append(dict(self.config['entries'][0]))
        self.write()
        with self.assertRaisesRegex(ValueError, 'duplicate entry label'):
            self.collect([])


class BindFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='atlas-g8g9-bind-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        checkpoint = self.root / 'config.json'
        checkpoint.write_text(json.dumps({'index_topk': 2048,
                                          'layer_types': ['kda', 'mla'],
                                          'router_dtype': 'float32'}))
        self.checkpoint_sha = sha(checkpoint)
        self.manifests = []
        for index in range(2):
            path = self.root / f'ref-manifest-{index}.json'
            path.write_text(json.dumps(
                {'provenance': {'config_sha256': {'config.json': self.checkpoint_sha}}}))
            self.manifests.append(path)
        self.descriptors = {}
        for engine, entry in (('candidate', 'serve'), ('reference', '/start.sh')):
            self.descriptors[engine] = {}
            for rank in ('0', '1'):
                path = self.root / f'{engine}-rank{rank}-create.json'
                path.write_text(json.dumps(
                    ['docker', 'create', '--env', 'VLLM_USE_FUSED_MOE_GROUPED_TOPK=1',
                     '--env', 'ATLAS_EXL3_TILE=fast', 'glm53:img', entry, '--tp', '2']))
                self.descriptors[engine][rank] = path
        self.pins = self.root / 'cublas131.sha256'
        self.pins.write_text(
            '4e346a7b60a01552bd9bf4bbc6c862e11eb851559c3da3572293345de6aaed75'
            '  /opt/atlas/cublas-13.1.1.3/libcublas.so.13\n'
            '930418840b747413584867cb7167dd5911dcf4184135589e0334527e9abbac9f'
            '  /opt/atlas/cublas-13.1.1.3/libcublasLt.so.13\n')
        self.report = self.root / 'provider-report.json'
        self.report.write_text(json.dumps(
            {'ok': True, 'issues': [], 'pid': 42,
             'libraries': [
                 {'name': 'libcublas.so.13', 'ok': True,
                  'sha256': '4e346a7b60a01552bd9bf4bbc6c862e11eb851559c3da3572293345de6aaed75'},
                 {'name': 'libcublasLt.so.13', 'ok': True,
                  'sha256': '930418840b747413584867cb7167dd5911dcf4184135589e0334527e9abbac9f'}]}))
        self.inventory = self.root / 'inventory.json'
        self.inventory.write_text(json.dumps(
            {'schema': 'atlas-exl3-g8g9-inventory-v1', 'image': 'ref:img',
             'entries': [{'label': 'exl3-ext', 'path': '/opt/plugin/x.so',
                          'sha256': 'c' * 64}]}))
        self.build_manifest = self.root / 'build-manifest.json'
        self.build_manifest.write_text(json.dumps(
            {'schema': 'atlas-exl3-build-manifest-v1', 'git_commit': 'a' * 40,
             'source_snapshot_sha256': 'd' * 32,
             'binaries': {'spark': {'sha256': 'ee' * 32}},
             'ptx_modules': ['one.ptx']}))
        self.anchor = self.root / 'coop_autotune.cu'
        self.anchor.write_text('// anchor source\n')
        self.plan_doc = self.root / 'plan.md'
        self.plan_doc.write_text(
            '## Source and binary contract\n\n'
            '| Local source anchor | SHA256 |\n'
            '|---|---|\n'
            f'| `AUDIT/installed-extra-source/quant/coop_autotune.cu` '
            f'| `{sha(self.anchor)}` |\n')
        self.config = {
            'schema': 'atlas-exl3-g8g9-bind-v1',
            'checkpoint_config': {'path': str(self.root / 'config.json'),
                                  'sha256': self.checkpoint_sha},
            'g8_fields': ['index_topk', 'layer_types', 'router_dtype'],
            'reference_manifests': [{'path': str(path), 'sha256': sha(path)}
                                    for path in self.manifests],
            'descriptors': {engine: {rank: {'path': str(path), 'sha256': sha(path)}
                                     for rank, path in ranks.items()}
                            for engine, ranks in self.descriptors.items()},
            'pins_file': {'path': str(self.pins), 'sha256': sha(self.pins)},
            'provider_reports': [{'path': str(self.report), 'sha256': sha(self.report)}],
            'inventories': [{'path': str(self.inventory), 'sha256': sha(self.inventory)}],
            'build_manifest': {'path': str(self.build_manifest),
                               'sha256': sha(self.build_manifest)},
            'source_anchors': {'AUDIT/installed-extra-source/quant/coop_autotune.cu':
                               str(self.anchor)},
            'plan_doc': {'path': str(self.plan_doc), 'sha256': sha(self.plan_doc)},
        }
        self.config_path = self.root / 'bind.json'
        self.write()

    def write(self):
        self.config_path.write_text(json.dumps(self.config))

    def bind(self):
        out = self.root / 'binding.json'
        return g8g9_binding.bind(self.config_path, out)


class BindTests(BindFixture):
    def test_full_binding_assembles_g8_and_g9(self):
        binding = self.bind()
        self.assertEqual(binding['schema'], 'atlas-exl3-g8g9-binding-v1')
        g8 = binding['g8']
        self.assertEqual(g8['checkpoint']['fields']['index_topk'], 2048)
        self.assertEqual(g8['manifest_crosscheck'], {'manifests': 2,
                            'config_sha256': self.checkpoint_sha,
                            'all_equal': True})
        self.assertEqual(g8['effective']['candidate']['0']['env'],
                         {'VLLM_USE_FUSED_MOE_GROUPED_TOPK': '1',
                          'ATLAS_EXL3_TILE': 'fast'})
        self.assertEqual(g8['effective']['reference']['1']['command'],
                         ['/start.sh', '--tp', '2'])
        self.assertEqual(g8['effective']['candidate']['0']['command'],
                         ['serve', '--tp', '2'])
        self.assertIn('libcublas.so.13', g8['providers']['pins'])
        self.assertEqual(len(g8['providers']['reports']), 1)
        g9 = binding['g9']
        self.assertEqual(g9['candidate_build']['git_commit'], 'a' * 40)
        self.assertEqual(g9['candidate_build']['ptx_modules'], 1)
        self.assertEqual(g9['inventories'][0]['entries'][0]['label'], 'exl3-ext')
        self.assertEqual(binding['inputs']['checkpoint_config']['sha256'],
                         self.checkpoint_sha)

    def test_manifest_config_sha_mismatch_fails(self):
        other = self.root / 'other-config.json'
        other.write_text('{"index_topk": 1, "layer_types": [], "router_dtype": "x"}')
        self.config['reference_manifests'][1]['sha256'] = sha(
            self.root / 'ref-manifest-1.json')  # keep file binding valid
        manifest = json.loads((self.root / 'ref-manifest-1.json').read_text())
        manifest['provenance']['config_sha256']['config.json'] = sha(other)
        (self.root / 'ref-manifest-1.json').write_text(json.dumps(manifest))
        self.config['reference_manifests'][1]['sha256'] = sha(
            self.root / 'ref-manifest-1.json')
        self.write()
        with self.assertRaisesRegex(ValueError, 'config_sha256 differs'):
            self.bind()

    def test_missing_g8_field_fails(self):
        self.config['g8_fields'].append('swiglu_limit')
        self.write()
        with self.assertRaisesRegex(ValueError, 'missing from checkpoint config'):
            self.bind()

    def test_tampered_anchor_fails_against_plan_table(self):
        self.anchor.write_text('// tampered\n')
        with self.assertRaisesRegex(ValueError, 'plan source contract'):
            self.bind()

    def test_tampered_bound_input_fails_early(self):
        self.inventory.write_text('{}')
        with self.assertRaisesRegex(ValueError, r'hash differs: inventories\[0\]'):
            self.bind()

    def test_no_inventory_fails_g9(self):
        self.config['inventories'] = []
        self.write()
        with self.assertRaisesRegex(ValueError, 'inventory required'):
            self.bind()

    def test_provider_report_disagreeing_with_pins_fails(self):
        report = json.loads(self.report.read_text())
        report['libraries'][0]['sha256'] = 'f' * 64
        self.report.write_text(json.dumps(report))
        self.config['provider_reports'][0]['sha256'] = sha(self.report)
        self.write()
        with self.assertRaisesRegex(ValueError, 'differs from pins'):
            self.bind()

    def test_unpinned_report_library_rejected_cleanly(self):
        report = json.loads(self.report.read_text())
        report['libraries'].append({'name': 'libcudart.so.13', 'ok': True,
                                    'sha256': 'd' * 64})
        self.report.write_text(json.dumps(report))
        self.config['provider_reports'][0]['sha256'] = sha(self.report)
        self.write()
        with self.assertRaisesRegex(ValueError, 'not pinned'):
            self.bind()

    def test_g8_symmetry_requires(self):
        for key, value, message in (
                ('provider_reports', [], 'provider report'),
                ('reference_manifests', [], 'reference manifest')):
            with self.subTest(key=key):
                config = dict(self.config)
                config[key] = value
                self.config_path.write_text(json.dumps(config))
                with self.assertRaisesRegex(ValueError, message):
                    self.bind()
        config = dict(self.config)
        config['descriptors'] = {'candidate': self.config['descriptors']['candidate']}
        self.config_path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, 'both engines'):
            self.bind()

    def test_descriptor_env_conflict_fails(self):
        path = self.root / 'candidate-rank0-create.json'
        path.write_text(json.dumps(
            ['docker', 'create', '--env', 'A=1', '--env', 'A=2',
             'glm53:img', 'serve']))
        self.config['descriptors']['candidate']['0'] = {'path': str(path),
                                                        'sha256': sha(path)}
        self.write()
        with self.assertRaisesRegex(ValueError, 'unique'):
            self.bind()


if __name__ == '__main__':
    unittest.main()
