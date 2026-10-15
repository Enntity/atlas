# SPDX-License-Identifier: AGPL-3.0-only
"""CPU tests for the generic reference-experiment runner."""
import unittest

import probe_runner as pr
import ref_experiment_runner as rx
from test_probe_runner import campaign_command, envs, volumes

DET = rx.EXPERIMENTS['det']


def with_fused_env(value='1'):
    command = campaign_command()
    at = command.index(pr.IMAGE)
    return command[:at] + ['--env', 'EXL3_FUSED_MOE=' + value] + command[at:]


class DetCommandTests(unittest.TestCase):
    def test_registered_value_replaces_the_campaign_value(self):
        campaign = with_fused_env('1')
        command = rx.experiment_command(DET, campaign, attempt='det02', rank=1)
        self.assertEqual(sum(1 for x in command if x.startswith('EXL3_FUSED_MOE=')), 1)
        delta = {k: v for k, v in envs(command).items() if envs(campaign).get(k) != v}
        self.assertEqual(delta, {'EXL3_FUSED_MOE': '0'})

    def test_isolation_mounts_and_campaign_observer(self):
        campaign = with_fused_env()
        command = rx.experiment_command(DET, campaign, attempt='det01', rank=0)
        vols = volumes(command)
        ra = DET.remote + '/attempts/det01'
        self.assertEqual(vols['/tmp/reference-logits'], ra + '/capture:/tmp/reference-logits')
        self.assertEqual(vols['/tmp/reference-autotune'], ra + '/cache/tune:/tmp/reference-autotune')
        self.assertEqual(vols['/opt/atlas-logits-patch'],
                         volumes(campaign)['/opt/atlas-logits-patch'])
        self.assertEqual(command[command.index('--name') + 1], 'det-det01-ref-0')
        self.assertEqual(command[-2:], [pr.IMAGE, '/start.sh'])

    def test_absent_variable_is_added(self):
        command = rx.experiment_command(DET, campaign_command(), attempt='det01', rank=0)
        self.assertEqual(envs(command)['EXL3_FUSED_MOE'], '0')

    def test_unregistered_attempt_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unregistered'):
            rx.experiment_command(DET, with_fused_env(), attempt='det04', rank=0)

    def test_draft_carries_the_delta(self):
        draft = rx.experiment_draft(DET, {'environment': {'EXL3_FUSED_MOE': '1', 'A': 'x'}})
        self.assertEqual(draft['environment'], {'EXL3_FUSED_MOE': '0', 'A': 'x'})

    def test_campaign_commands_set_fused_moe_on_both_ranks(self):
        for rank in (0, 1):
            self.assertEqual(envs(pr.campaign_command(rank)).get('EXL3_FUSED_MOE'), '1')


IDX = rx.EXPERIMENTS['idx']


class IdxCommandTests(unittest.TestCase):
    def test_observer_override_and_env_delta(self):
        campaign = with_fused_env('1')
        command = rx.experiment_command(IDX, campaign, attempt='idx01', rank=0)
        self.assertEqual(volumes(command)['/opt/atlas-logits-patch'],
                         rx.remote_observer(IDX) + ':/opt/atlas-logits-patch:ro')
        delta = {k: v for k, v in envs(command).items() if envs(campaign).get(k) != v}
        mla = ','.join(str(i) for i in range(3, 45, 4))
        self.assertEqual(delta, {'EXL3_FUSED_MOE': '0',
                                 'ATLAS_EXL3_FFN_DUMP_DIR': '/tmp/ffn-capture',
                                 'ATLAS_EXL3_FFN_DUMP_LAYERS': mla,
                                 'ATLAS_EXL3_ATTN_DUMP_LAYERS': mla})
        self.assertEqual(mla.split(',')[:3], ['3', '7', '11'])
        self.assertEqual(len(mla.split(',')), 11)

    def test_draft_hooks_follow_the_observer(self):
        draft = rx.experiment_draft(IDX, {'environment': {},
                                          'capture_hooks_sha256': {'sitecustomize.py': 'old',
                                                                   'reference_chunk_trace.py': 'old'}})
        hashes = rx.observer_hashes(IDX)
        self.assertEqual(draft['capture_hooks_sha256'],
                         {k: hashes[k] for k in ('sitecustomize.py', 'reference_chunk_trace.py')})
        self.assertIn('reference_attn_trace.py', hashes)


if __name__ == '__main__':
    unittest.main()
