# SPDX-License-Identifier: AGPL-3.0-only
"""CPU contract tests for the probe observer glue (no vLLM, no torch, no GPU).

The fixtures mirror the installed reference interfaces the wiring binds:
Glm5NextMoE construction/forward hooks with a ``layers.<n>.mlp`` prefix, the
serial v2 ``prepare_inputs`` result, and the CudaCommunicator dispatch
surface. Frozen harness behaviour itself is covered by its own tests; these
tests prove the glue's obligations: side-effect-free contexts, buffered
writes with the registered flush boundary, engagement counters, fail-closed
discovery/coverage, and installation receipts.
"""
import json
import os
from pathlib import Path
import tempfile
import types
import unittest

import numpy as np

import probe_wiring


def make_tensor(values, dtype='float32'):
    numpy_dtype = {'float32': '<f4', 'float16': '<f2', 'int64': '<i8'}[dtype]
    array = np.array(values, dtype=numpy_dtype)

    class FakeTensor:
        def __init__(self, data):
            self._data = data
            self.dtype = 'torch.' + dtype
            self.shape = data.shape
            self.device = 'cpu'

        def detach(self):
            return self

        def contiguous(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._data

        def clone(self):
            return FakeTensor(self._data.copy())

    return FakeTensor(array)


class FakeRouter:
    def __init__(self):
        self.capture_fn = None
        self.routing_result = (make_tensor([0.25, 0.75]),
                               make_tensor([3, 5], 'int64'))

    def set_capture_fn(self, capture_fn):
        self.capture_fn = capture_fn

    def _compute_routing(self, hidden_states, router_logits, indices_type, **kwargs):
        return self.routing_result

    def _select_experts(self, hidden_states, router_logits, indices_type, **kwargs):
        topk_weights, topk_ids = self._compute_routing(
            hidden_states, router_logits, indices_type, **kwargs)
        if self.capture_fn is not None:
            self.capture_fn(topk_ids)
        return topk_weights, topk_ids


class FakeRunner:
    """Mirrors the pinned MoERunner: a module holding the router as a plain
    attribute (FusedMoERouter is not an nn.Module)."""
    def __init__(self, router):
        self.router = router


class FakeMoE:
    def __init__(self, prefix='', tp_rank=0):
        self.prefix = prefix
        self.tp_rank = tp_rank
        self._pre = []
        self._post = []
        self.router_inst = FakeRouter()
        self.experts = FakeRunner(self.router_inst)

    def register_forward_pre_hook(self, fn, with_kwargs=False):
        self._pre.append(fn)

    def register_forward_hook(self, fn, with_kwargs=False, always_call=False):
        self._post.append((fn, always_call))

    def named_modules(self):
        # Modules only, as nn.Module.named_modules yields: the router is not one.
        return [('', self), ('experts', self.experts)]

    def forward(self, hidden, logits):
        output = None
        for hook in self._pre:
            hook(self, (hidden, logits), {})
        try:
            output = self.router_inst._select_experts(hidden, logits, 'fp16')
        finally:
            for hook, always_call in self._post:
                if output is not None or always_call:
                    hook(self, (hidden, logits), {}, output)
        return output


class FakePynccl:
    def __init__(self):
        self.calls = 0

    def all_reduce(self, value):
        self.calls += 1
        return value

    def all_gather(self, value):
        self.calls += 1
        return value


class CudaCommunicator:
    def __init__(self, unique_name='tp:0', rank=0, world_size=2,
                 pynccl_comm=None, ca_comm=None):
        self.unique_name = unique_name
        self.rank = rank
        self.world_size = world_size
        self.pynccl_comm = pynccl_comm
        self.ca_comm = ca_comm

    def all_reduce(self, value):
        return self.pynccl_comm.all_reduce(value)

    def all_gather(self, value):
        return self.pynccl_comm.all_gather(value)


class GPUModelRunner:
    """Mirrors the pinned v2 execute_model signature."""
    def __init__(self, state=None):
        self.state = state
        self.prepare_wrapper = None

    def execute_model(self, scheduler_output, intermediate_tensors=None,
                      dummy_run=False, skip_attn_for_dummy_run=False,
                      is_profile=False, context_len=0):
        if self.state is not None and not dummy_run and \
                scheduler_output.num_scheduled_tokens:
            # A routed forward inside the step writes marked telemetry.
            self.state.begin_forward(3, 0)
            self.state.write_metadata('m', {}, self.state.collective_context()())
            self.state.end_forward()
        return 'out'


def sched(*ids):
    return types.SimpleNamespace(num_scheduled_tokens={i: 1 for i in ids},
                                 total_num_scheduled_tokens=len(ids))


BACKENDS = [{'name': 'PYNCL_AR', 'attr': 'pynccl_comm',
             'method': 'all_reduce', 'required': True},
            {'name': 'PYNCL_AG', 'attr': 'pynccl_comm',
             'method': 'all_gather', 'required': True}]


def module_with_file(name, tmp):
    path = Path(tmp) / (name + '.py')
    path.write_text('# fixture module ' + name + '\nVALUE = 1\n')
    module = types.ModuleType(name)
    module.__file__ = str(path)
    return module, path


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='atlas-probe-wiring-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'telemetry'
        self.root.mkdir()
        self.state = probe_wiring.ProbeState(self.root, register_atexit=False)
        # Flushes land under the flushing process's own pid directory.
        self.out = self.root / ('pid-%d' % os.getpid())

    def flush_dirs(self):
        if not self.out.exists():
            return []
        return sorted(path.name for path in self.out.iterdir())


class StateTests(Fixture):
    def test_processes_sharing_a_dir_flush_under_their_own_pid(self):
        # API server, engine core and worker share one telemetry dir.
        self.state.record_install('sitecustomize', {})
        self.state.write_metadata('m', {}, None)
        self.state.close()
        self.assertEqual(self.flush_dirs(), ['flush-000001'])
        real_getpid = os.getpid
        try:
            os.getpid = lambda: real_getpid() + 1
            forked = probe_wiring.ProbeState(self.root, register_atexit=False)
            forked.write_metadata('m', {}, None)
            forked.close()
        finally:
            os.getpid = real_getpid
        self.assertEqual(sorted(p.name for p in self.root.iterdir()),
                         sorted(['pid-%d' % real_getpid(),
                                 'pid-%d' % (real_getpid() + 1)]))

    def test_context_is_side_effect_free_and_marked_only(self):
        context = self.state.router_context(3, 0)
        self.assertIsNone(context())
        self.state.mark_request('req-A')
        self.state.begin_forward(3, 0)
        first = context()
        self.assertEqual(first, (3, 0, 0, 'req-A'))
        self.assertEqual(context(), first)
        self.assertEqual(context(), first)
        self.state.end_forward()
        self.assertIsNone(context())

    def test_invocation_advances_per_forward_not_per_context_call(self):
        context = self.state.router_context(3, 0)
        self.state.mark_request('req-A')
        for expected in (0, 1, 2):
            self.state.begin_forward(3, 0)
            self.assertEqual(context()[2], expected)
            self.state.end_forward()

    def test_context_layer_mismatch_fails_closed(self):
        context = self.state.router_context(3, 0)
        self.state.mark_request('req-A')
        self.state.begin_forward(5, 0)
        with self.assertRaisesRegex(ValueError, 'does not match'):
            context()

    def test_unmarked_forward_yields_none_context(self):
        context = self.state.router_context(3, 0)
        self.state.begin_forward(3, 0)
        self.assertIsNone(context())
        self.assertIsNone(self.state.collective_context()())

    def test_every_routed_layer_in_one_window_gets_its_own_file(self):
        # on01 regression: stage counters are per (context, stage), so all
        # routed layers in one step write 'router-gate-logits-000'.
        self.state.mark_request('req-A')
        for layer in range(3, 45):
            self.state.begin_forward(layer, 0)
            self.state.write_tensor('router-gate-logits-000',
                                    make_tensor([float(layer)]),
                                    (layer, 0, 0, 'req-A'))
            self.state.end_forward()
        self.state.flush()
        payload = json.loads((self.out / 'flush-000001' / 'records.json').read_text())
        self.assertEqual(len(payload['records']), 42)
        for record in payload['records']:
            raw = (self.out / 'flush-000001' / record['file']).read_bytes()
            self.assertEqual(raw, np.array([float(record['ctx'][0])],
                                           dtype='<f4').tobytes())

    def test_request_transition_flushes_and_counters_are_per_request(self):
        self.state.mark_request('req-A')
        self.state.begin_forward(3, 0)
        ctx = (3, 0, 0, 'req-A')
        self.state.write_tensor('router-gate-logits-000',
                                make_tensor([1.0, 2.0]), ctx)
        self.state.write_tensor('router-topk-ids-000',
                                make_tensor([7], 'int64'), ctx)
        self.state.end_forward()
        self.assertEqual(self.flush_dirs(), [])
        self.state.mark_request('req-B')
        self.assertEqual(self.flush_dirs(), ['flush-000001'])
        payload = json.loads((self.out / 'flush-000001'
                              / 'records.json').read_text())
        self.assertEqual(payload['counters']['req-A'],
                         {'gate_logits_written': 1, 'weights_written': 0,
                          'ids_written': 1})
        self.assertEqual(payload['counters'].get('req-B'), None)
        bins = sorted(p.name for p in (self.out / 'flush-000001').iterdir()
                      if p.suffix == '.bin')
        self.assertEqual(bins, ['000000-router-gate-logits-000.bin',
                                '000001-router-topk-ids-000.bin'])
        raw = (self.out / 'flush-000001'
               / '000000-router-gate-logits-000.bin').read_bytes()
        self.assertEqual(raw, np.array([1.0, 2.0], dtype='<f4').tobytes())
        self.state.write_tensor('router-topk-ids-000',
                                make_tensor([8], 'int64'),
                                (3, 0, 0, 'req-B'))
        self.state.unmark_request()
        self.assertEqual(self.flush_dirs(),
                         ['flush-000001', 'flush-000002'])
        payload = json.loads((self.out / 'flush-000002'
                              / 'records.json').read_text())
        self.assertEqual(payload['counters']['req-B']['ids_written'], 1)
        self.state.close()

    def test_coverage_violation_forces_an_immediate_flush(self):
        self.state.mark_request('req-A')
        self.state.begin_forward(3, 0)
        ctx = (3, 0, 0, 'req-A')
        self.state.write_metadata('collective-coverage-violation',
                                  {'op': 'all_gather'}, ctx)
        self.assertEqual(self.flush_dirs(), ['flush-000001'])
        payload = json.loads((self.out / 'flush-000001'
                              / 'records.json').read_text())
        names = [record['name'] for record in payload['records']]
        self.assertIn('collective-coverage-violation', names)

    def test_empty_flush_is_a_noop_and_install_receipts_land_once(self):
        self.state.flush()
        self.assertEqual(self.flush_dirs(), [])
        self.state.record_install('router', {'layer': 3})
        self.state.write_metadata('collective-all_reduce-000', {'op': 1}, None)
        self.state.flush()
        first = json.loads((self.out / 'flush-000001'
                            / 'records.json').read_text())
        self.assertIn('install_receipts', first)
        self.state.write_metadata('collective-all_reduce-001', {'op': 2}, None)
        self.state.flush()
        second = json.loads((self.out / 'flush-000002'
                             / 'records.json').read_text())
        self.assertNotIn('install_receipts', second)

    def test_unsupported_tensor_dtype_rejected(self):
        tensor = make_tensor([1.0])
        tensor.dtype = 'torch.float64'
        with self.assertRaisesRegex(ValueError, 'dtype'):
            probe_wiring.tensor_bytes(tensor)


class PatchMoeTests(Fixture):
    def patch(self):
        module, path = module_with_file('fixture_model', self.tmp.name)
        module.Glm5NextMoE = type('Glm5NextMoE', (FakeMoE,), {})
        probe_wiring.patch_moe(module, self.state, {'fixture': 'json:loads'},
                               probe_wiring.sha_file(path))
        return module

    def test_router_install_capture_and_flush_chain(self):
        module = self.patch()
        moe = module.Glm5NextMoE(prefix='layers.3.mlp', tp_rank=1)
        self.state.mark_request('req-A')
        moe.forward(make_tensor([9.0]), make_tensor([1.0, 2.0]))
        self.state.mark_request('req-B')  # transition flushes request A
        payload = json.loads((self.out / 'flush-000001'
                              / 'records.json').read_text())
        self.assertEqual(payload['counters']['req-A'],
                         {'gate_logits_written': 1, 'weights_written': 1,
                          'ids_written': 1})
        tensor_records = [r for r in payload['records'] if r['kind'] == 'tensor']
        self.assertEqual([r['name'] for r in tensor_records],
                         ['router-gate-logits-000', 'router-topk-weights-000',
                          'router-topk-ids-000'])
        self.assertTrue(all(r['ctx'] == [3, 1, 0, 'req-A']
                            for r in tensor_records))
        install = payload['install_receipts']
        router_install = [r for r in install if r['kind'] == 'router']
        self.assertEqual(len(router_install), 1)
        self.assertEqual(router_install[0]['layer'], 3)
        self.assertEqual(router_install[0]['router_class'], 'FakeRouter')
        stats = payload['cumulative_stats'][router_install[0]['stats_key']]
        self.assertEqual(stats['capture_callback_calls'], 1)
        self.assertEqual(stats['ids_written'], 1)
        # Second forward: invocation advances, no reinstall.
        moe.forward(make_tensor([8.0]), make_tensor([3.0, 4.0]))
        self.state.close()
        second = json.loads((self.out / 'flush-000002'
                             / 'records.json').read_text())
        self.assertEqual([r['ctx'][2] for r in second['records']
                          if r['kind'] == 'tensor'], [1, 1, 1])
        router_installs = [r for r in second.get('install_receipts', [])
                           if r['kind'] == 'router']
        self.assertEqual(router_installs, [])

    def test_unmatched_prefix_instance_stays_uninstrumented(self):
        module = self.patch()
        moe = module.Glm5NextMoE(prefix='aux.head', tp_rank=0)
        self.state.mark_request('req-A')
        moe.forward(make_tensor([9.0]), make_tensor([1.0, 2.0]))
        self.state.flush()
        self.assertEqual(self.flush_dirs(), [])

    def test_ambiguous_router_discovery_fails_the_forward(self):
        module = self.patch()
        moe = module.Glm5NextMoE(prefix='layers.3.mlp', tp_rank=0)

        def duplicated():
            return [('', moe), ('a', FakeRunner(FakeRouter())),
                    ('b', FakeRunner(FakeRouter()))]
        moe.named_modules = duplicated
        self.state.mark_request('req-A')
        with self.assertRaisesRegex(ValueError, 'exactly one resident router'):
            moe.forward(make_tensor([9.0]), make_tensor([1.0, 2.0]))

    def test_router_module_child_is_still_discovered_once(self):
        module = self.patch()
        moe = module.Glm5NextMoE(prefix='layers.3.mlp', tp_rank=0)
        # Same router reachable as a module child and as a runner attribute.
        moe.named_modules = lambda: [('', moe), ('experts', moe.experts),
                                     ('experts.router', moe.router_inst)]
        self.state.mark_request('req-A')
        moe.forward(make_tensor([9.0]), make_tensor([1.0, 2.0]))
        self.state.close()
        payload = json.loads((self.out / 'flush-000001'
                              / 'records.json').read_text())
        routers = [r for r in payload['install_receipts'] if r['kind'] == 'router']
        self.assertEqual(len(routers), 1)

    def test_missing_router_fails_the_forward(self):
        module = self.patch()
        moe = module.Glm5NextMoE(prefix='layers.3.mlp', tp_rank=0)
        moe.named_modules = lambda: [('', moe)]
        self.state.mark_request('req-A')
        with self.assertRaisesRegex(ValueError, 'found 0'):
            moe.forward(make_tensor([9.0]), make_tensor([1.0, 2.0]))

    def test_preflight_rejects_tampered_model_source(self):
        module, path = module_with_file('fixture_model2', self.tmp.name)
        module.Glm5NextMoE = type('Glm5NextMoE', (FakeMoE,), {})
        with self.assertRaisesRegex(ValueError, 'installed source differs'):
            probe_wiring.patch_moe(module, self.state, {'fixture': 'json:loads'},
                                   '0' * 64)


class PatchRunnerTests(Fixture):
    def patch(self):
        module, path = module_with_file('fixture_runner', self.tmp.name)
        module.GPUModelRunner = type('GPUModelRunner', (GPUModelRunner,), {})
        probe_wiring.patch_runner(module, self.state,
                                  probe_wiring.sha_file(path))
        return module

    def test_every_step_flushes_under_the_scheduled_request(self):
        module = self.patch()
        runner = module.GPUModelRunner(self.state)
        self.assertEqual(runner.execute_model(sched('req-1')), 'out')
        self.assertEqual(self.flush_dirs(), ['flush-000001'])
        runner.execute_model(sched('req-1'))
        runner.execute_model(sched('req-2'))
        self.assertEqual(self.flush_dirs(),
                         ['flush-000001', 'flush-000002', 'flush-000003'])
        third = json.loads((self.out / 'flush-000003' / 'records.json').read_text())
        self.assertEqual([r['request'] for r in third['records']], ['req-2'])
        # The last request is on disk without any atexit/SIGTERM flush.
        self.assertIsNone(self.state.collective_context()())

    def test_idle_and_dummy_steps_unmark(self):
        module = self.patch()
        runner = module.GPUModelRunner(self.state)
        runner.execute_model(sched('req-1'))
        runner.execute_model(sched())
        self.state.begin_forward(3, 0)
        self.assertIsNone(self.state.collective_context()())
        self.state.end_forward()
        runner.execute_model(sched('req-2'), None, True)
        self.state.begin_forward(3, 0)
        self.assertIsNone(self.state.collective_context()())
        self.state.end_forward()

    def test_prepare_inputs_is_left_to_the_lineage_observer(self):
        module, path = module_with_file('fixture_runner3', self.tmp.name)
        def prepare(self, scheduler_output, batch_req_state, batch_desc):
            return None
        module.GPUModelRunner = type('GPUModelRunner', (GPUModelRunner,),
                                     {'prepare_inputs': prepare})
        probe_wiring.patch_runner(module, self.state, probe_wiring.sha_file(path))
        self.assertIs(module.GPUModelRunner.prepare_inputs, prepare)

    def test_concurrent_step_fails_closed(self):
        module = self.patch()
        runner = module.GPUModelRunner(self.state)
        with self.assertRaisesRegex(ValueError, 'serial'):
            runner.execute_model(sched('a', 'b'))

    def test_preflight_rejects_tampered_runner_source(self):
        module, path = module_with_file('fixture_runner2', self.tmp.name)
        module.GPUModelRunner = type('GPUModelRunner', (GPUModelRunner,), {})
        with self.assertRaisesRegex(ValueError, 'installed source differs'):
            probe_wiring.patch_runner(module, self.state, '0' * 64)


class PatchCommunicatorTests(Fixture):
    def patch(self, backends=None):
        module, path = module_with_file('fixture_comm', self.tmp.name)
        module.CudaCommunicator = type('CudaCommunicator',
                                       (CudaCommunicator,), {})
        probe_wiring.patch_communicator(
            module, self.state, backends if backends is not None else BACKENDS,
            {'fixture': 'json:loads'}, probe_wiring.sha_file(path))
        return module

    def test_dispatch_records_carry_forward_context_and_backends(self):
        module = self.patch()
        comm = module.CudaCommunicator(unique_name='tp:0',
                                       pynccl_comm=FakePynccl())
        self.state.mark_request('req-A')
        self.state.begin_forward(3, 0)
        comm.all_reduce('value')
        self.state.end_forward()
        self.state.close()
        payload = json.loads((self.out / 'flush-000001'
                              / 'records.json').read_text())
        records = [r for r in payload['records']
                   if r['name'].startswith('collective-all_reduce')]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['ctx'], [3, 0, 0, 'req-A'])
        self.assertEqual(records[0]['record']['backends'],
                         [{'backend': 'PYNCL_AR', 'op': 'all_reduce',
                           'status': 'returned'}])
        install = [r for r in payload['install_receipts']
                   if r['kind'] == 'dispatch'][0]
        self.assertEqual(install['group']['unique_name'], 'tp:0')
        stats = payload['cumulative_stats'][install['stats_key']]
        self.assertEqual(stats['all_reduce_calls'], 1)
        self.assertEqual(stats['backend_calls'], 1)

    def test_unregistered_backend_branch_fails_closed_with_evidence(self):
        module = self.patch(BACKENDS[:1])  # all_gather backend not registered
        comm = module.CudaCommunicator(unique_name='tp:0',
                                       pynccl_comm=FakePynccl())
        self.state.mark_request('req-A')
        with self.assertRaisesRegex(ValueError, 'Unknown collective branch'):
            comm.all_gather('value')
        self.assertEqual(self.flush_dirs(), ['flush-000001'])
        payload = json.loads((self.out / 'flush-000001'
                              / 'records.json').read_text())
        names = [r['name'] for r in payload['records']]
        self.assertIn('collective-coverage-violation', names)

    def test_optional_absent_backend_is_recorded_not_fatal(self):
        specs = BACKENDS + [{'name': 'CUSTOM', 'attr': 'ca_comm',
                             'method': 'custom_all_reduce', 'required': False}]
        module = self.patch(specs)
        module.CudaCommunicator(pynccl_comm=FakePynccl())
        self.assertEqual(len(self.state.absent_backends), 1)
        self.assertEqual(self.state.absent_backends[0]['name'], 'CUSTOM')
        # absent backends are emitted once, like install receipts
        self.state.write_metadata('collective-all_reduce-000', {'op': 1}, None)
        self.state.flush()
        self.state.write_metadata('collective-all_reduce-001', {'op': 2}, None)
        self.state.flush()
        first = json.loads((self.out / 'flush-000001'
                            / 'records.json').read_text())
        second = json.loads((self.out / 'flush-000002'
                             / 'records.json').read_text())
        self.assertEqual(first['absent_backends'][0]['name'], 'CUSTOM')
        self.assertNotIn('absent_backends', second)

    def test_required_absent_backend_fails_the_attempt(self):
        specs = [{'name': 'CUSTOM', 'attr': 'ca_comm',
                  'method': 'custom_all_reduce', 'required': True}]
        module = self.patch(specs)
        with self.assertRaisesRegex(ValueError, 'required collective backend'):
            module.CudaCommunicator()

    def test_malformed_backend_spec_rejected(self):
        with self.assertRaisesRegex(ValueError, 'backend specs'):
            self.patch([{'name': 'X', 'attr': 'pynccl_comm'}])


class ResolveTargetsTests(unittest.TestCase):
    def test_explicit_map_resolves_attributes(self):
        resolved = probe_wiring.resolve_targets({'loads': 'json:loads'})
        self.assertIs(resolved['loads'], json.loads)

    def test_bare_module_reference_rejected(self):
        with self.assertRaisesRegex(ValueError, 'module:attribute'):
            probe_wiring.resolve_targets({'x': 'json'})

    def test_empty_map_rejected(self):
        with self.assertRaisesRegex(ValueError, 'source-target'):
            probe_wiring.resolve_targets({})


if __name__ == '__main__':
    unittest.main()
