# SPDX-License-Identifier: AGPL-3.0-only
"""CPU tests for the probe runner's command and provenance-draft derivation."""
import json
import unittest

import probe_runner as pr

SRC0 = '/home/colum/.cache/vllm-glm53-flash'


def campaign_command(src=SRC0):
    return ['docker', 'create', '--name', 'gate-ref-0', '--network', 'host',
            '--entrypoint', 'bash',
            '--env', 'EXLLAMAV3_TUNE_CACHE=/tmp/reference-autotune/coop_autotune_v1.bin',
            '--env', 'PYTHONPATH=/opt/atlas-logits-patch',
            '--volume', src + '/nv:/root/.nv/ComputeCache',
            '--volume', src + ':/root/.cache/vllm',
            '--volume', src + '/triton:/root/.triton/cache',
            '--volume', src + '/tilelang:/root/.tilelang/cache',
            '--volume', '/tmp/ctx/reference-autotune:/tmp/reference-autotune',
            '--volume', '/tmp/ctx-patch:/opt/atlas-logits-patch:ro',
            '--volume', '/tmp/ctx/reference-logits:/tmp/reference-logits',
            '--volume', '/tmp/ctx/ffn-capture:/tmp/ffn-capture',
            pr.IMAGE, '/start.sh']


def volumes(command):
    return {command[i + 1].split(':')[1]: command[i + 1]
            for i, a in enumerate(command[:-1]) if a == '--volume'}


def envs(command):
    return {command[i + 1].partition('=')[0]: command[i + 1].partition('=')[2]
            for i, a in enumerate(command[:-1]) if a == '--env'}


class ProbeCommandTests(unittest.TestCase):
    def build(self, attempt, rank=0):
        return pr.probe_command(campaign_command(), attempt=attempt, rank=rank,
                                config_json='{"k":1}')

    def test_arms_differ_only_by_the_telemetry_env(self):
        off, on = self.build('off01'), self.build('on01')
        norm = lambda c, a: [x.replace('/attempts/' + a + '/', '/attempts/X/')
                             .replace('-' + a + '-', '-X-') for x in c]
        on_n = norm(on, 'on01')
        for var in ('ATLAS_PROBE_TELEMETRY_DIR=' + pr.TELEMETRY,
                    'ATLAS_PROBE_CONFIG={"k":1}'):
            i = on_n.index(var)
            self.assertEqual(on_n[i - 1], '--env')
            del on_n[i - 1:i + 1]
        self.assertEqual(norm(off, 'off01'), on_n)
        for command in (off, on):
            self.assertEqual(command[command.index('--entrypoint') + 1], 'bash')
            self.assertEqual(command[-2:], [pr.IMAGE, '/start.sh'])
            self.assertNotIn(pr.NSYS_DIR + ':' + pr.NSYS_DIR + ':ro', command)

    def test_fixed_mount_sources_exclude_attempt_scoped_paths(self):
        sources = pr.fixed_mount_sources(self.build('off01'), 'off01')
        self.assertIn(pr.REMOTE_OVERLAY, sources)
        self.assertIn('/tmp/ctx/ffn-capture', sources)
        self.assertFalse([s for s in sources if '/attempts/off01/' in s])

    def test_off_arm_matches_campaign_except_isolation_mounts(self):
        off = self.build('off02', rank=1)
        campaign = campaign_command()
        self.assertEqual(envs(off), envs(campaign))
        self.assertEqual(off[off.index('--name') + 1], 'probe-rm-off02-ref-1')
        vols = volumes(off)
        ra = pr.REMOTE + '/attempts/off02'
        self.assertEqual(vols['/root/.cache/vllm'], ra + '/cache/vllm:/root/.cache/vllm')
        self.assertEqual(vols['/root/.nv/ComputeCache'],
                         ra + '/cache/vllm/nv:/root/.nv/ComputeCache')
        self.assertEqual(vols['/tmp/reference-autotune'],
                         ra + '/cache/tune:/tmp/reference-autotune')
        self.assertEqual(vols['/tmp/reference-logits'],
                         ra + '/capture:/tmp/reference-logits')
        self.assertEqual(vols['/opt/atlas-logits-patch'],
                         pr.REMOTE_OVERLAY + ':/opt/atlas-logits-patch:ro')
        self.assertEqual(vols['/tmp/ffn-capture'], volumes(campaign)['/tmp/ffn-capture'])
        self.assertEqual(vols[pr.TELEMETRY], ra + '/telemetry:' + pr.TELEMETRY)

    def test_unregistered_attempt_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unregistered'):
            self.build('off04')

    def test_cache_subdirectories_must_share_the_vllm_parent(self):
        command = campaign_command()
        index = command.index(SRC0 + '/triton:/root/.triton/cache')
        command[index] = '/elsewhere/triton:/root/.triton/cache'
        with self.assertRaisesRegex(ValueError, 'triton'):
            pr.probe_command(command, attempt='off01', rank=0, config_json='{}')

    def test_cache_sources(self):
        self.assertEqual(pr.cache_sources(campaign_command()),
                         {'vllm': SRC0, 'tune': '/tmp/ctx/reference-autotune'})


class ProbeDraftTests(unittest.TestCase):
    def draft(self, attempt):
        campaign = {'environment': {'A': '1'},
                    'capture_hooks_sha256': {'sitecustomize.py': 'old',
                                             'reference_chunk_trace.py': 'old2'},
                    'hook_package_sha256': 'oldpkg'}
        files = {'sitecustomize.py': 'new', 'reference_chunk_trace.py': 'new2',
                 'probe_wiring.py': 'x'}
        return pr.probe_draft(campaign, attempt=attempt, config_json='{"k":1}',
                              package_files=files, package_sha='pkg')

    def test_on_arm_draft_carries_probe_env(self):
        draft = self.draft('on02')
        self.assertEqual(draft['environment'],
                         {'A': '1', 'ATLAS_PROBE_TELEMETRY_DIR': pr.TELEMETRY,
                          'ATLAS_PROBE_CONFIG': '{"k":1}'})
        self.assertEqual(draft['capture_hooks_sha256'],
                         {'sitecustomize.py': 'new', 'reference_chunk_trace.py': 'new2'})
        self.assertEqual(draft['hook_package_sha256'], 'pkg')

    def test_off_arm_draft_keeps_campaign_env(self):
        self.assertEqual(self.draft('off03')['environment'], {'A': '1'})


class ConfigValueTests(unittest.TestCase):
    def test_compact_sorted(self):
        self.assertEqual(pr.config_json_value({'b': 1, 'a': [1, 2]}),
                         '{"a":[1,2],"b":1}')


if __name__ == '__main__':
    unittest.main()
