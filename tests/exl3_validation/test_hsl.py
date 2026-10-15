# SPDX-License-Identifier: AGPL-3.0-only
"""CPU tests for the hidden-state localization runner and analyzer."""
import json
from pathlib import Path
import tempfile
import unittest

import hsl_localize as hl
import hsl_runner as hr
import probe_runner as pr
from test_probe_runner import campaign_command, envs, volumes

ROWS = {'short16/a-000000': ['cmpl-aaa', 5]}


def write_tensor(root, layer, rank, inv, stage, payload, rid='cmpl-aaa-0-x', chunk=0):
    stem = 'ffn-L%d-rank%d-%06d-%s' % (layer, rank, inv, stage)
    (root / (stem + '.bin')).write_bytes(payload)
    (root / (stem + '.json')).write_text(json.dumps(
        {'file': stem + '.bin',
         'chunk_position': {'request_id': rid, 'computed_before': chunk}}))


def make_run(root, overrides=()):
    """Layer 0 (pre + one fused, dense FFN) and layer 3 (two fused, MoE)."""
    over = dict(overrides)
    def put(layer, inv, stage, base):
        payload = over.get((layer, inv, stage), over.get((layer, stage), base))
        for rank in (0, 1):
            write_tensor(root, layer, rank, inv, stage, payload)
    put(0, 0, 'mhc-pre-input-x', b'e')
    put(0, 0, 'mhc-pre-layer-input', b'a0')
    put(0, 0, 'mhc-fused-input-x', b'o0')
    put(0, 0, 'mhc-fused-layer-input', b'f0')
    put(0, 0, 'input', b'f0')
    put(0, 0, 'post-tp-output', b'm0')
    put(3, 4, 'mhc-fused-input-x', b'm2')
    put(3, 4, 'mhc-fused-layer-input', b'a3')
    put(3, 5, 'mhc-fused-input-x', b'o3')
    put(3, 5, 'mhc-fused-layer-input', b'f3')
    put(3, 0, 'input', b'f3')
    put(3, 0, 'router-logits', b'g3')
    put(3, 0, 'routed-output', b'r3')
    put(3, 0, 'post-tp-output', b'p3')
    (root / 'ffn-L3-rank0-000000-runtime-semantics.json').write_text('{}')


class LocalizeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.a = Path(self.tmp.name) / 'a'
        self.b = Path(self.tmp.name) / 'b'
        self.a.mkdir()
        self.b.mkdir()

    def run_pair(self, overrides):
        make_run(self.a)
        make_run(self.b, overrides)
        a, b = hl.index(self.a, ROWS), hl.index(self.b, ROWS)
        hl.self_check(a, 'A')
        hl.self_check(b, 'B')
        return hl.localize(a, b)

    def test_identical_runs(self):
        results = self.run_pair(())
        self.assertEqual({v['first_stage'] for v in results.values()}, {None})

    def test_first_difference_is_the_attention_output_at_layer3(self):
        # Attention output (FFN-side mHC input x) differs; everything later too.
        results = self.run_pair({(3, 5, 'mhc-fused-input-x'): b'O3',
                                 (3, 5, 'mhc-fused-layer-input'): b'F3',
                                 (3, 'input'): b'F3', (3, 'router-logits'): b'G3'})
        value = results['short16/a-000000|rank0']
        self.assertEqual((value['first_layer'], value['first_stage']), (3, 'mhc-ffn-input-x'))
        self.assertEqual(value['first_ffn_input_layer'], 3)
        self.assertEqual(value['first_router_logits_layer'], 3)

    def test_attention_side_call_precedes_ffn_side(self):
        # Both fused calls write 'layer-input'; the lower invocation (4) is the
        # attention side. Alter only it and the attention output after it.
        results = self.run_pair({(3, 4, 'mhc-fused-layer-input'): b'A3',
                                 (3, 5, 'mhc-fused-input-x'): b'O3'})
        value = results['short16/a-000000|rank0']
        self.assertEqual(value['first_stage'], 'mhc-attn-layer-input')
        self.assertEqual(value['first_five'][1], [0, 3, 'mhc-ffn-input-x'])
        self.assertIsNone(value['first_ffn_input_layer'])

    def test_rank_local_first_difference_is_flagged(self):
        make_run(self.a)
        make_run(self.b, {(3, 'routed-output'): b'R3', (3, 'post-tp-output'): b'P3'})
        value = hl.localize(hl.index(self.a, ROWS), hl.index(self.b, ROWS))['short16/a-000000|rank1']
        self.assertEqual(value['first_stage'], 'routed-output')
        self.assertTrue(value['first_is_rank_local'])

    def test_join_self_check_catches_misaligned_ffn_input(self):
        make_run(self.a, {(3, 'input'): b'X'})
        with self.assertRaisesRegex(ValueError, 'join is wrong'):
            hl.self_check(hl.index(self.a, ROWS), 'A')

    def test_router_logits_before_input_is_rejected(self):
        make_run(self.a)
        make_run(self.b, {(3, 'router-logits'): b'G3x'})
        a, b = hl.index(self.a, ROWS), hl.index(self.b, ROWS)
        with self.assertRaisesRegex(ValueError, 'before their input'):
            hl.localize(a, b)

    def test_chunks_order_before_layers_and_attention_stages_in_dataflow(self):
        # Two prefill chunks; chunk 128's indexer top-k differs first, before
        # its sparse-attention output and before any later stage.
        for root, top in ((self.a, b't'), (self.b, b'T')):
            for chunk in (0, 128):
                inv = chunk // 128
                for stage, payload in (('attn-input-hidden_states', b'h'),
                                       ('attn-indexer_wq_b.0', b'q'),
                                       ('attn-indexer_indexer_op',
                                        top if chunk == 128 else b't'),
                                       ('attn-mla_attn_mla_attn', b'M' if (chunk == 128 and root is self.b) else b'm'),
                                       ('attn-output', b'o'), ('input', b'f')):
                    write_tensor(root, 3, 0, inv, stage, payload, chunk=chunk)
        value = hl.localize(hl.index(self.a, ROWS), hl.index(self.b, ROWS))['short16/a-000000|rank0']
        self.assertEqual((value['first_chunk'], value['first_layer'], value['first_stage']),
                         (128, 3, 'attn-indexer_indexer_op'))
        self.assertEqual(value['first_five'][1], [128, 3, 'attn-mla_attn_mla_attn'])
        self.assertTrue(value['first_is_rank_local'])

    def test_unknown_request_is_strict_by_default(self):
        make_run(self.a)
        write_tensor(self.a, 3, 0, 9, 'input', b'z', rid='cmpl-zzz-0-y')
        with self.assertRaisesRegex(ValueError, 'unknown request'):
            hl.index(self.a, ROWS)
        self.assertIn(('short16/a-000000', 0), hl.index(self.a, ROWS, ignore_unknown=True))


class RunnerTests(unittest.TestCase):
    def test_command_adds_only_dump_env_and_isolation_mounts(self):
        campaign = campaign_command()
        command = hr.hsl_command(campaign, attempt='hsl02', rank=1)
        delta = {k: v for k, v in envs(command).items() if envs(campaign).get(k) != v}
        self.assertEqual(delta, hr.DUMP_ENV)
        vols = volumes(command)
        ra = hr.REMOTE + '/attempts/hsl02'
        self.assertEqual(vols['/tmp/ffn-capture'], ra + '/dump:/tmp/ffn-capture')
        self.assertEqual(vols['/tmp/reference-logits'], ra + '/capture:/tmp/reference-logits')
        self.assertEqual(vols['/root/.cache/vllm'], ra + '/cache/vllm:/root/.cache/vllm')
        # The campaign observer itself, unchanged (not the probe overlay).
        self.assertEqual(vols['/opt/atlas-logits-patch'],
                         volumes(campaign)['/opt/atlas-logits-patch'])
        self.assertEqual(command[command.index('--name') + 1], 'hsl-hsl02-ref-1')
        self.assertEqual(command[-2:], [pr.IMAGE, '/start.sh'])

    def test_dump_layers_cover_every_decoder_layer(self):
        self.assertEqual(hr.DUMP_ENV['ATLAS_EXL3_FFN_DUMP_LAYERS'].split(','),
                         [str(i) for i in range(45)])

    def test_unregistered_attempt_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unregistered'):
            hr.hsl_command(campaign_command(), attempt='hsl04', rank=0)

    def test_draft_adds_dump_env_only(self):
        draft = hr.hsl_draft({'environment': {'A': '1'}, 'capture_hooks_sha256': {'x': 'y'}})
        self.assertEqual(draft['environment'], dict({'A': '1'}, **hr.DUMP_ENV))
        self.assertEqual(draft['capture_hooks_sha256'], {'x': 'y'})


if __name__ == '__main__':
    unittest.main()
