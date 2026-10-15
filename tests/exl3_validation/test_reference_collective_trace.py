# SPDX-License-Identifier: AGPL-3.0-only
"""CPU contract controls for reference collective dispatch instrumentation."""
import copy
from pathlib import Path
import unittest

from reference_collective_trace import install_collective_trace


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


class Backend:
    """Device-communicator backend gate: reject (TypeError), decline (None), succeed."""
    def __init__(self, mode):
        self.mode = mode
        self.calls = []
        self.result = None

    def custom_all_reduce(self, input_):
        return self._run(input_)

    def all_reduce(self, input_):
        return self._run(input_)

    def _run(self, input_):
        self.calls.append(input_)
        if self.mode == 'reject':
            raise TypeError('size/dtype gate rejects input')
        if self.mode == 'decline':
            return None
        return self.result if self.result is not None else input_.clone()


class GatherBackend:
    def __init__(self):
        self.calls = []
        self.result = None

    def all_gather(self, input_):
        self.calls.append(input_)
        return self.result if self.result is not None else input_.clone()


class TorchFallback:
    """torch.distributed-like fallback reached by the dispatch's unknown branch."""
    def __init__(self):
        self.calls = []

    def all_reduce(self, input_):
        self.calls.append(input_)
        return input_.clone()


class Dispatch:
    """Mirrors CudaCommunicator.all_reduce: try backends in order, then fallback."""
    unique_name = 'fixture-tp-group'
    rank = 0
    world_size = 2

    def __init__(self, chain, fallback, gatherer=None, crash=None):
        self.chain = chain  # ordered (target, method_name) backend attempts
        self.fallback = fallback
        self.gatherer = gatherer
        self.crash = crash
        self.calls = []
        self.gather_calls = []

    def all_reduce(self, input_):
        self.calls.append(input_)
        if self.crash is not None:
            raise self.crash
        for target, method in self.chain:
            try:
                out = getattr(target, method)(input_)
            except TypeError:
                continue
            if out is not None:
                return out
        return self.fallback.all_reduce(input_)

    def all_gather(self, input_, dim=-1):
        self.gather_calls.append((input_, dim))
        return self.gatherer.all_gather(input_)


class MinimalDispatch:
    """Dispatch without group attributes; group fields must bind as None."""
    def __init__(self, backend):
        self.backend = backend

    def all_reduce(self, input_):
        return self.backend.all_reduce(input_)

    def all_gather(self, input_, dim=-1):
        return input_


class Passthrough:
    """Backend used to prove the no-tensor-argument record path."""
    def all_reduce(self, input_):
        return input_


def source_dependency():
    """Stand-in installed communicator infrastructure whose source is hashed."""


class Recorder:
    def __init__(self):
        self.metadata = []
        self.paths = []

    def write_metadata(self, stage, value, context):
        self.metadata.append((stage, copy.deepcopy(value), context))

    def read_source(self, path):
        self.paths.append(path)
        return Path(path).read_bytes()


def chain_dispatch(ca_mode='reject', nxt_mode='success', crash=None, with_gatherer=False):
    ca, nxt, fallback = Backend(ca_mode), Backend(nxt_mode), TorchFallback()
    gatherer = GatherBackend() if with_gatherer else None
    dispatch = Dispatch([(ca, 'custom_all_reduce'), (nxt, 'all_reduce')],
                        fallback, gatherer, crash)
    return dispatch, ca, nxt, fallback, gatherer


def standard_backends(ca, nxt, gather=None):
    backends = [('CUSTOM', ca, 'custom_all_reduce'), ('PYNCCL', nxt, 'all_reduce')]
    if gather is not None:
        backends.append(('GATHER', gather, 'all_gather'))
    return backends


def install(dispatch, recorder, context, backends, stream_fn=None,
            read_source=None, source_targets=None):
    return install_collective_trace(
        dispatch, context=context, write_metadata=recorder.write_metadata,
        read_source=read_source or recorder.read_source,
        source_targets=({'fixture': source_dependency}
                        if source_targets is None else source_targets),
        backends=backends, stream_fn=stream_fn)


class Tests(unittest.TestCase):
    def test_fallback_chain_records_raised_then_returned_under_one_dispatch(self):
        dispatch, ca, nxt, fallback, _ = chain_dispatch()
        recorder = Recorder()
        nxt.result = Tensor([9.0])
        install(dispatch, recorder, lambda: (0, 0, 1, 'req-a'),
                standard_backends(ca, nxt), stream_fn=lambda: 17)
        input_ = Tensor([1.0, 2.0])
        out = dispatch.all_reduce(input_)
        self.assertIs(out, nxt.result)  # original result object identity
        self.assertIs(dispatch.calls[0], input_)
        self.assertIs(ca.calls[0], input_)  # exact argument forwarding
        self.assertIs(nxt.calls[0], input_)
        self.assertEqual(fallback.calls, [])  # success short-circuits the fallback
        self.assertEqual([s for s, _, _ in recorder.metadata],
                         ['collective-all_reduce-000'])
        stage, record, ctx = recorder.metadata[0]
        self.assertEqual(ctx, (0, 0, 1, 'req-a'))
        self.assertEqual(record['op'], 'all_reduce')
        self.assertEqual(record['seq'], 0)
        self.assertEqual(record['request'], 'req-a')
        self.assertEqual(record['dispatch_status'], 'returned')
        self.assertEqual(record['backends'],
                         [{'backend': 'CUSTOM', 'op': 'all_reduce', 'status': 'raised'},
                          {'backend': 'PYNCCL', 'op': 'all_reduce', 'status': 'returned'}])
        self.assertEqual(record['tensor'],
                         {'shape': [2], 'dtype': 'bf16-fixture',
                          'device': 'cpu', 'numel': 2})
        self.assertEqual(record['group'],
                         {'unique_name': 'fixture-tp-group', 'rank': 0, 'world_size': 2})
        self.assertIsNone(record['group_reason'])
        self.assertEqual(record['stream'], 17)
        self.assertIsNone(record['stream_reason'])

    def test_unknown_branch_without_registered_backend_fails_coverage(self):
        dispatch, _, _, fallback, _ = chain_dispatch(ca_mode='decline', nxt_mode='decline')
        recorder = Recorder()
        install(dispatch, recorder, lambda: (0, 0, 0, 'req-b'), backends=[])
        with self.assertRaises(ValueError) as caught:
            dispatch.all_reduce(Tensor([1.0]))
        self.assertIn('all_reduce', str(caught.exception))
        self.assertIn('sequence 0', str(caught.exception))
        self.assertEqual([s for s, _, _ in recorder.metadata],
                         ['collective-coverage-violation'])
        stage, record, _ = recorder.metadata[0]
        self.assertEqual(record['dispatch_status'], 'returned')
        self.assertEqual(record['backends'], [])  # unknown branch: zero backend events
        self.assertEqual(len(fallback.calls), 1)  # dispatch reached its own fallback

    def test_registered_torch_fallback_backend_makes_same_path_pass(self):
        dispatch, ca, nxt, fallback, _ = chain_dispatch(ca_mode='decline', nxt_mode='decline')
        recorder = Recorder()
        install(dispatch, recorder, lambda: (0, 0, 0, 'req-c'),
                [('TORCH_FALLBACK', fallback, 'all_reduce')])
        out = dispatch.all_reduce(Tensor([1.0]))
        self.assertEqual([s for s, _, _ in recorder.metadata],
                         ['collective-all_reduce-000'])
        stage, record, _ = recorder.metadata[0]
        self.assertEqual(record['backends'],
                         [{'backend': 'TORCH_FALLBACK', 'op': 'all_reduce',
                           'status': 'returned'}])
        self.assertEqual(record['dispatch_status'], 'returned')
        self.assertIsNotNone(out)

    def test_unmarked_dispatch_still_records_with_ctx_none(self):
        # Deliberate deviation: collectives are process-global, so every
        # dispatch records; ctx None identifies the unmarked call.
        dispatch, ca, nxt, _, _ = chain_dispatch()
        recorder = Recorder()
        install(dispatch, recorder, lambda: None, standard_backends(ca, nxt))
        dispatch.all_reduce(Tensor([1.0]))
        stage, record, ctx = recorder.metadata[0]
        self.assertEqual(stage, 'collective-all_reduce-000')
        self.assertIsNone(ctx)
        self.assertIsNone(record['ctx'])
        self.assertIsNone(record['request'])
        self.assertEqual(record['dispatch_status'], 'returned')

    def test_absent_group_fields_and_unsupplied_stream_bind_none_with_reason(self):
        backend = Backend('success')
        recorder = Recorder()
        dispatch = MinimalDispatch(backend)
        install(dispatch, recorder, lambda: None,
                [('PLAIN', backend, 'all_reduce')])
        dispatch.all_reduce(Tensor([1.0]))
        dispatch_record = [r for s, r, _ in recorder.metadata
                           if s == 'collective-all_reduce-000'][0]
        self.assertEqual(dispatch_record['group'],
                         {'unique_name': None, 'rank': None, 'world_size': None})
        self.assertEqual(dispatch_record['group_reason'],
                         'absent on dispatch: unique_name, rank, world_size')
        self.assertIsNone(dispatch_record['stream'])
        self.assertEqual(dispatch_record['stream_reason'], 'stream_fn not supplied')

    def test_no_tensor_argument_records_none_with_reason(self):
        recorder = Recorder()
        dispatch = MinimalDispatch(Passthrough())
        install(dispatch, recorder, lambda: None, [('PASS', dispatch.backend, 'all_reduce')])
        dispatch.all_reduce(object())  # no .shape/.dtype anywhere in the call
        record = [r for s, r, _ in recorder.metadata
                  if s == 'collective-all_reduce-000'][0]
        self.assertIsNone(record['tensor'])
        self.assertEqual(record['tensor_reason'], 'no tensor argument exposed')

    def test_standalone_backend_calls_record_with_standalone_true(self):
        dispatch, ca, nxt, _, _ = chain_dispatch(ca_mode='success')
        recorder = Recorder()
        install(dispatch, recorder, lambda: None, standard_backends(ca, nxt))
        ca.custom_all_reduce(Tensor([1.0]))
        ca.custom_all_reduce(Tensor([2.0]))
        self.assertEqual([s for s, _, _ in recorder.metadata],
                         ['collective-backend-all_reduce-000',
                          'collective-backend-all_reduce-001'])
        for (stage, record, ctx) in recorder.metadata:
            self.assertIsNone(ctx)
            self.assertEqual(record['standalone'], True)
            self.assertEqual(record['backend'], 'CUSTOM')
            self.assertEqual(record['op'], 'all_reduce')
            self.assertEqual(record['status'], 'returned')
        rejecting = Backend('reject')
        rejecting_recorder = Recorder()
        install(MinimalDispatch(rejecting), rejecting_recorder, lambda: None,
                [('REJECT', rejecting, 'all_reduce')])
        with self.assertRaises(TypeError):  # standalone raises stay unmasked
            rejecting.all_reduce(Tensor([5.0]))
        stage, record, ctx = rejecting_recorder.metadata[0]
        self.assertEqual((stage, record['status'], record['standalone']),
                         ('collective-backend-all_reduce-000', 'raised', True))

    def test_kwargs_forwarding_and_gather_result_identity(self):
        dispatch, ca, nxt, _, gatherer = chain_dispatch(with_gatherer=True)
        recorder = Recorder()
        gatherer.result = Tensor([7.0])
        install(dispatch, recorder, lambda: (0, 0, 2, 'req-g'),
                standard_backends(ca, nxt, gatherer))
        input_ = Tensor([1.0, 2.0])
        out = dispatch.all_gather(input_, dim=0)
        self.assertIs(out, gatherer.result)
        self.assertEqual(dispatch.gather_calls[0][1], 0)  # exact kwargs forwarding
        self.assertIs(gatherer.calls[0], input_)
        record = [r for s, r, _ in recorder.metadata
                  if s == 'collective-all_gather-000'][0]
        self.assertEqual(record['op'], 'all_gather')
        self.assertEqual(record['seq'], 0)
        self.assertEqual(record['tensor']['shape'], [2])

    def test_unknown_source_rejection_for_dispatch_backend_and_targets(self):
        for problem in ('raises', 'empty', 'text'):
            with self.subTest(problem=problem):
                dispatch, ca, nxt, _, _ = chain_dispatch()
                recorder = Recorder()
                if problem == 'raises':
                    def read_source(path): raise OSError('source unavailable')
                elif problem == 'empty':
                    read_source = lambda path: b''
                else:
                    read_source = lambda path: 'not bytes'
                with self.assertRaises(ValueError):
                    install(dispatch, recorder, lambda: None,
                            standard_backends(ca, nxt), read_source=read_source)
        # A backend target whose method has no readable installed source.
        namespace = {}
        exec('def custom_all_reduce(self, x):\n    return x', namespace)
        opaque = type('Opaque', (), {})()
        opaque.custom_all_reduce = namespace['custom_all_reduce']
        dispatch, ca, nxt, _, _ = chain_dispatch()
        with self.assertRaises(ValueError):
            install(dispatch, Recorder(), lambda: None,
                    standard_backends(ca, nxt) + [('OPAQUE', opaque, 'custom_all_reduce')])
        # A source target whose installed source cannot be bound.
        with self.assertRaises(ValueError):
            install(dispatch, Recorder(), lambda: None, standard_backends(ca, nxt),
                    source_targets={'opaque': opaque.custom_all_reduce})

    def test_backend_op_suffix_duplicates_and_self_target_rejected(self):
        dispatch, ca, nxt, _, gatherer = chain_dispatch(with_gatherer=True)
        recorder = Recorder()
        for backends in ([('GATHER', gatherer, 'gather_v')],
                         [('RS', gatherer, 'custom_reduce_scatter')],
                         [('CUSTOM', ca, 'custom_all_reduce'),
                          ('CUSTOM', nxt, 'all_reduce')],
                         [('SELF', dispatch, 'all_reduce')],
                         [('MISSING', object(), 'all_reduce')],
                         [('BAD', ca, 'custom_all_reduce', 'extra')]):
            with self.subTest(backends=backends), self.assertRaises(ValueError):
                install(dispatch, recorder, lambda: None, backends)

    def test_install_guards_reject_double_install_and_bad_dispatch(self):
        dispatch, ca, nxt, _, _ = chain_dispatch()
        recorder = Recorder()
        install(dispatch, recorder, lambda: None, standard_backends(ca, nxt))
        with self.assertRaises(ValueError):
            install(dispatch, recorder, lambda: None, standard_backends(ca, nxt))

        class NoGather:
            def all_reduce(self, input_):
                return input_

        with self.assertRaises(ValueError):
            install(NoGather(), Recorder(), lambda: None, [])
        with self.assertRaises(ValueError):
            install(dispatch, Recorder(), lambda: None, [], stream_fn='not callable')

    def test_original_dispatch_exception_reraised_unmasked_with_raised_record(self):
        dispatch, ca, nxt, _, _ = chain_dispatch(crash=RuntimeError('dispatch boom'))
        recorder = Recorder()
        install(dispatch, recorder, lambda: (0, 0, 0, 'req-x'), standard_backends(ca, nxt))
        with self.assertRaises(RuntimeError) as caught:
            dispatch.all_reduce(Tensor([1.0]))
        self.assertEqual(str(caught.exception), 'dispatch boom')
        self.assertEqual([s for s, _, _ in recorder.metadata],
                         ['collective-all_reduce-000'])  # no coverage violation
        stage, record, _ = recorder.metadata[0]
        self.assertEqual(record['dispatch_status'], 'raised')
        self.assertEqual(record['backends'], [])

    def test_restore_unwraps_everything_and_foreign_replacement_fails(self):
        dispatch, ca, nxt, _, gatherer = chain_dispatch(with_gatherer=True)
        recorder = Recorder()
        binding = install(dispatch, recorder, lambda: None,
                          standard_backends(ca, nxt, gatherer))
        binding.restore()
        self.assertIs(dispatch.all_reduce.__func__, Dispatch.all_reduce)
        self.assertNotIn('all_reduce', vars(dispatch))
        self.assertIs(ca.custom_all_reduce.__func__, Backend.custom_all_reduce)
        self.assertNotIn('custom_all_reduce', vars(ca))
        self.assertFalse(hasattr(dispatch, '_atlas_collective_trace'))
        binding.restore()  # second restore is a safe no-op
        binding2 = install(dispatch, recorder, lambda: None,
                           standard_backends(ca, nxt, gatherer))
        dispatch.all_reduce(Tensor([1.0]))
        self.assertEqual(binding2.stats()['backend_calls'], 2)
        binding2.restore()
        self.assertIs(dispatch.all_reduce.__func__, Dispatch.all_reduce)

    def test_foreign_replacement_before_restore_fails_closed(self):
        for foreign in ('dispatch', 'backend'):
            with self.subTest(foreign=foreign):
                dispatch, ca, nxt, _, _ = chain_dispatch()
                recorder = Recorder()
                binding = install(dispatch, recorder, lambda: None,
                                  standard_backends(ca, nxt))
                if foreign == 'dispatch':
                    dispatch.all_gather = lambda *a, **k: None
                else:
                    ca.custom_all_reduce = lambda *a, **k: None
                with self.assertRaises(ValueError):
                    binding.restore()

    def test_stats_counts_match_all_dispatch_and_backend_calls(self):
        dispatch, ca, nxt, _, gatherer = chain_dispatch(with_gatherer=True)
        recorder = Recorder()
        binding = install(dispatch, recorder, lambda: None,
                          standard_backends(ca, nxt, gatherer))
        dispatch.all_reduce(Tensor([1.0]))
        dispatch.all_reduce(Tensor([2.0]))
        ca.mode = 'success'  # gate accepts this input on the standalone call
        ca.custom_all_reduce(Tensor([3.0]))  # standalone
        dispatch.all_gather(Tensor([4.0]))
        self.assertEqual(binding.stats(),
                         {'all_reduce_calls': 2, 'all_gather_calls': 1,
                          'backend_calls': 5, 'standalone_backend_calls': 1})
        self.assertEqual([s for s, _, _ in recorder.metadata],
                         ['collective-all_reduce-000', 'collective-all_reduce-001',
                          'collective-backend-all_reduce-000',
                          'collective-all_gather-000'])
        seqs = [r['seq'] for _, r, _ in recorder.metadata if 'seq' in r]
        self.assertEqual(seqs, [0, 1, 2])

    def test_unwritable_backend_target_rolls_back_all_wrapping(self):
        class Frozen:
            """Backend object whose method slot rejects instance assignment."""
            def __init__(self):
                self.calls = []

            def all_reduce(self, input_):
                return input_.clone()

            def __setattr__(self, name, value):
                if name == 'all_reduce':
                    raise AttributeError('method slot is frozen')
                object.__setattr__(self, name, value)

        dispatch, ca, nxt, _, _ = chain_dispatch()
        frozen = Frozen()
        with self.assertRaises(ValueError):
            install(dispatch, Recorder(), lambda: None,
                    [('CUSTOM', ca, 'custom_all_reduce'),
                     ('FROZEN', frozen, 'all_reduce')])
        self.assertIs(dispatch.all_reduce.__func__, Dispatch.all_reduce)
        self.assertNotIn('all_reduce', vars(dispatch))
        self.assertNotIn('_atlas_collective_trace', vars(dispatch))
        self.assertIs(ca.custom_all_reduce.__func__, Backend.custom_all_reduce)
        self.assertNotIn('all_reduce', vars(frozen))


if __name__ == '__main__':
    unittest.main()
