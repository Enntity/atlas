# SPDX-License-Identifier: AGPL-3.0-only
"""CPU contract controls for reference router capture instrumentation."""
import copy
from pathlib import Path
import unittest

from reference_router_capture import install_router_capture


class Tensor:
    """Minimal mutable tensor boundary; arithmetic is deliberately not simulated."""
    def __init__(self, values):
        self.values = list(values)
        self.shape = (len(values),)
        self.dtype = 'bf16-fixture'
        self.device = 'cpu'

    def detach(self):
        return self

    def clone(self):
        return Tensor(self.values)


class FakeEplb:
    """Stand-in EPLBLayerState so recorded status carries its concrete class."""


class FakeRouter:
    """Mirrors BaseRouter._select_experts: compute -> capture_fn(ids) -> eplb -> dtype convert.

    The fixture mirrors the installed reference sequence so the real capture
    callback fires through it. ``mutate_logits`` makes the original
    ``_compute_routing`` mutate router_logits in place, proving the binding's
    snapshot-before-call discipline.
    """
    def __init__(self, mutate_logits=False, eplb_state=...):
        if eplb_state is not ...:
            self.eplb_state = eplb_state
        self.mutate_logits = mutate_logits
        self.capture_fn = None
        self.routing_result = (Tensor([0.25, 0.75]), Tensor([3, 5]))
        self.compute_calls = []
        self.eplb_mapped = []

    def set_capture_fn(self, capture_fn):
        self.capture_fn = capture_fn

    def _compute_routing(self, hidden_states, router_logits, indices_type, *, input_ids=None):
        self.compute_calls.append((hidden_states, router_logits, indices_type, input_ids))
        if self.mutate_logits:
            router_logits.values[:] = [-99.0, -99.0, -99.0]  # in-place mutation
        return self.routing_result

    def _select_experts(self, hidden_states, router_logits, indices_type, *, input_ids=None):
        topk_weights, topk_ids = self._compute_routing(
            hidden_states, router_logits, indices_type, input_ids=input_ids)
        if self.capture_fn is not None:
            self.capture_fn(topk_ids)  # supported hook: logical ids before EPLB
        if getattr(self, 'eplb_state', None) is not None:
            topk_ids = Tensor([value + 100 for value in topk_ids.values])
            self.eplb_mapped.append(topk_ids)
        return topk_weights, topk_ids  # dtype conversion is identity in the fixture


def source_dependency():
    """Stand-in installed router infrastructure whose source is hashed."""


class Recorder:
    def __init__(self):
        self.tensors = []
        self.metadata = []
        self.paths = []

    def write_tensor(self, stage, tensor, context):
        self.tensors.append((stage, list(tensor.values), context))

    def write_metadata(self, stage, value, context):
        self.metadata.append((stage, copy.deepcopy(value), context))

    def read_source(self, path):
        self.paths.append(path)
        return Path(path).read_bytes()


class Tests(unittest.TestCase):
    def install(self, router, recorder, context, read_source=None):
        return install_router_capture(
            router, context=context, write_tensor=recorder.write_tensor,
            write_metadata=recorder.write_metadata,
            read_source=read_source or recorder.read_source,
            source_targets={'fixture': source_dependency})

    def test_real_capture_callback_fires_through_mirrored_sequence_with_binding(self):
        router, recorder = FakeRouter(), Recorder()
        binding = self.install(router, recorder, lambda: (0, 1, 7, 'req-42'))
        hidden, logits = Tensor([9.0]), Tensor([1.0, 2.0])
        input_ids = object()
        weights, ids = router._select_experts(hidden, logits, 'fp16', input_ids=input_ids)
        values = {stage: data for stage, data, _ in recorder.tensors}
        contexts = {stage: ctx for stage, _, ctx in recorder.tensors}
        self.assertEqual(list(values),
                         ['router-gate-logits-000', 'router-topk-weights-000',
                          'router-topk-ids-000'])
        self.assertEqual(values['router-gate-logits-000'], [1.0, 2.0])
        self.assertEqual(values['router-topk-weights-000'], [0.25, 0.75])
        self.assertEqual(values['router-topk-ids-000'], [3, 5])  # pre-EPLB logical ids
        self.assertTrue(all(ctx == (0, 1, 7, 'req-42') for ctx in contexts.values()))
        self.assertIs(weights, router.routing_result[0])
        self.assertIs(ids, router.routing_result[1])
        stage, meta, ctx = recorder.metadata[0]
        self.assertEqual((stage, ctx), ('router-binding', None))
        self.assertEqual(meta['class'],
                         {'module': __name__, 'qualname': 'FakeRouter'})
        self.assertEqual(meta['eplb_status'], 'absent')
        self.assertTrue(meta['logical_ids_are_physical'])
        self.assertTrue(meta['sources']['compute_routing']['qualname']
                        .endswith('_compute_routing'))
        self.assertTrue(meta['sources']['select_experts']['qualname']
                        .endswith('_select_experts'))
        self.assertTrue(meta['sources']['set_capture_fn']['qualname']
                        .endswith('set_capture_fn'))
        self.assertEqual(meta['source_targets']['fixture']['qualname'],
                         'source_dependency')
        self.assertEqual(binding.stats(),
                         {'compute_routing_calls': 1, 'capture_callback_calls': 1,
                          'gate_logits_written': 1, 'weights_written': 1,
                          'ids_written': 1})

    def test_eplb_state_statuses_are_recorded_and_ids_stay_logical(self):
        for eplb, status, physical in ((None, 'disabled', True),
                                       (FakeEplb(), 'FakeEplb', False)):
            with self.subTest(eplb=status):
                router, recorder = FakeRouter(eplb_state=eplb), Recorder()
                self.install(router, recorder, lambda: (0, 0, 1, 'req-eplb'))
                router._select_experts(Tensor([1.0]), Tensor([1.0, 2.0]), 'fp16')
                meta = recorder.metadata[0][1]
                self.assertEqual(meta['eplb_status'], status)
                self.assertEqual(meta['logical_ids_are_physical'], physical)
                ids_writes = [data for stage, data, _ in recorder.tensors
                              if stage.startswith('router-topk-ids')]
                self.assertEqual(ids_writes, [[3, 5]])  # captured before EPLB mapping

    def test_unmarked_context_zeroes_writes_but_counters_still_engage(self):
        router, recorder = FakeRouter(), Recorder()
        binding = self.install(router, recorder, lambda: None)
        router._select_experts(Tensor([1.0]), Tensor([1.0, 2.0]), 'fp16')
        self.assertEqual(recorder.tensors, [])
        self.assertEqual([item[0] for item in recorder.metadata], ['router-binding'])
        self.assertEqual(binding.stats(),
                         {'compute_routing_calls': 1, 'capture_callback_calls': 1,
                          'gate_logits_written': 0, 'weights_written': 0,
                          'ids_written': 0})
        stats = binding.stats()
        stats['compute_routing_calls'] = 999
        self.assertEqual(binding.stats()['compute_routing_calls'], 1)

    def test_exact_argument_forwarding_and_original_result_identity(self):
        router, recorder = FakeRouter(), Recorder()
        self.install(router, recorder, lambda: (0, 2, 0, 'req-forward'))
        hidden, logits, input_ids = object(), Tensor([1.0, 2.0]), object()
        original_result = router.routing_result
        result = router._compute_routing(hidden, logits, 'fp32', input_ids=input_ids)
        self.assertIs(result, original_result)
        seen = router.compute_calls[0]
        self.assertEqual((seen[0] is hidden, seen[1] is logits, seen[3] is input_ids),
                         (True, True, True))
        # kwargs router_logits path (positional index 1 absent)
        router._compute_routing(hidden_states=hidden, indices_type='fp32',
                                router_logits=logits)
        self.assertIs(router.compute_calls[1][1], logits)

    def test_snapshot_before_call_survives_in_place_mutation_of_input(self):
        router, recorder = FakeRouter(mutate_logits=True), Recorder()
        self.install(router, recorder, lambda: (0, 0, 0, 'req-snap'))
        logits = Tensor([1.0, 2.0])
        router._compute_routing(Tensor([0.0]), logits, 'fp16')
        written = {stage: data for stage, data, _ in recorder.tensors}
        self.assertEqual(written['router-gate-logits-000'], [1.0, 2.0])
        self.assertEqual(logits.values, [-99.0, -99.0, -99.0])  # original mutated

    def test_unknown_source_and_install_guards_reject(self):
        for problem in ('raises', 'empty', 'text'):
            with self.subTest(problem=problem):
                router, recorder = FakeRouter(), Recorder()
                if problem == 'raises':
                    def read_source(path): raise OSError('source unavailable')
                elif problem == 'empty':
                    read_source = lambda path: b''
                else:
                    read_source = lambda path: 'not bytes'
                with self.assertRaises(ValueError):
                    self.install(router, recorder, lambda: None, read_source=read_source)

    def test_double_install_preexisting_callback_and_missing_router_rejected(self):
        router, recorder = FakeRouter(), Recorder()
        self.install(router, recorder, lambda: None)
        with self.assertRaises(ValueError):
            self.install(router, recorder, lambda: None)
        fresh, fresh_recorder = FakeRouter(), Recorder()
        fresh.capture_fn = lambda ids: None
        with self.assertRaises(ValueError):
            self.install(fresh, fresh_recorder, lambda: None)

        class Partial:
            def set_capture_fn(self, capture_fn):
                self.capture_fn = capture_fn

        for bad_router in (object(), Partial()):
            with self.subTest(router=type(bad_router).__name__), self.assertRaises(ValueError):
                self.install(bad_router, Recorder(), lambda: None)
        with self.assertRaises(ValueError):
            install_router_capture(FakeRouter(), context=lambda: None,
                                   write_tensor=Recorder().write_tensor,
                                   write_metadata=Recorder().write_metadata,
                                   read_source=Recorder().read_source,
                                   source_targets={})

    def test_bad_context_values_fail_at_call_time(self):
        for ctx in ((0, 1, 7), (-1, 0, 0, 'req'), (0, 0, 0, ''), ('0', 0, 0, 'req')):
            with self.subTest(ctx=ctx):
                router, recorder = FakeRouter(), Recorder()
                self.install(router, recorder, lambda: ctx)
                with self.assertRaises(ValueError):
                    router._compute_routing(Tensor([0.0]), Tensor([1.0]), 'fp16')
                self.assertEqual(recorder.tensors, [])

    def test_capture_callback_context_is_validated_when_it_fires(self):
        router, recorder = FakeRouter(), Recorder()
        contexts = iter([(0, 0, 0, 'req-good'), (0, 0, 0, '')])
        self.install(router, recorder, lambda: next(contexts))
        with self.assertRaises(ValueError):
            router._select_experts(Tensor([0.0]), Tensor([1.0]), 'fp16')
        self.assertFalse([s for s, _, _ in recorder.tensors
                          if s.startswith('router-topk-ids')])

    def test_non_pair_routing_result_fails_closed(self):
        for bad_result in ((Tensor([1.0]), Tensor([2.0]), Tensor([3.0])), 5):
            with self.subTest(result=bad_result):
                router, recorder = FakeRouter(), Recorder()
                router.routing_result = bad_result
                self.install(router, recorder, lambda: (0, 0, 0, 'req-bad'))
                with self.assertRaises(ValueError):
                    router._compute_routing(Tensor([0.0]), Tensor([1.0]), 'fp16')
                self.assertFalse([s for s, _, _ in recorder.tensors
                                  if s.startswith('router-topk-weights')])

    def test_restore_unwraps_and_reinstall_works(self):
        router, recorder = FakeRouter(), Recorder()
        original = router._compute_routing
        binding = self.install(router, recorder, lambda: (0, 0, 0, 'req'))
        binding.restore()
        self.assertIs(router._compute_routing.__func__, original.__func__)
        self.assertNotIn('_compute_routing', vars(router))
        self.assertIsNone(router.capture_fn)
        self.assertFalse(hasattr(router, '_atlas_router_capture'))
        binding.restore()  # second restore is a safe no-op
        binding2 = self.install(router, recorder, lambda: (0, 0, 0, 'req'))
        router._select_experts(Tensor([1.0]), Tensor([1.0, 2.0]), 'fp16')
        self.assertEqual(binding2.stats()['capture_callback_calls'], 1)
        binding2.restore()
        self.assertIs(router._compute_routing.__func__, original.__func__)

    def test_foreign_replacement_before_restore_fails_closed(self):
        for foreign in ('compute', 'capture'):
            with self.subTest(foreign=foreign):
                router, recorder = FakeRouter(), Recorder()
                binding = self.install(router, recorder, lambda: None)
                if foreign == 'compute':
                    router._compute_routing = lambda *a, **k: None
                else:
                    router.capture_fn = lambda ids: None
                with self.assertRaises(ValueError):
                    binding.restore()

    def test_nonretaining_capture_fn_fails_closed_without_residue(self):
        class NonRetainingRouter(FakeRouter):
            def set_capture_fn(self, capture_fn):  # drops the registration
                pass

        router, recorder = NonRetainingRouter(), Recorder()
        with self.assertRaises(ValueError):
            self.install(router, recorder, lambda: None)
        self.assertIsNone(router.capture_fn)
        self.assertFalse(hasattr(router, '_atlas_router_capture'))
        self.assertNotIn('_compute_routing', vars(router))


if __name__ == '__main__':
    unittest.main()
