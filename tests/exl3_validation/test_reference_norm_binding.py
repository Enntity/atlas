# SPDX-License-Identifier: AGPL-3.0-only
"""CPU contract controls for resident gated normalization instrumentation."""
import copy
from pathlib import Path
import unittest

from reference_norm_binding import install_norm_binding


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


class ResidentNorm:
    def __init__(self):
        self.weight = Tensor([2, 3])
        self.bias = None
        self.eps = 0.125
        self.activation = 'sigmoid'
        self.elementwise_affine = True
        self.hidden_size = 2
        self._forward_method = self.forward_cuda
        self.calls = []

    def forward_native(self, x, g, residual=None, prenorm=False, residual_in_fp32=False):
        return self.forward_cuda(x, g, residual, prenorm, residual_in_fp32)

    def forward_cuda(self, x, g, residual=None, prenorm=False, residual_in_fp32=False):
        self.calls.append((x, g, residual, prenorm, residual_in_fp32))
        x.values[:] = [71, 89]  # CUDA may alias output to core input.
        return (x, residual) if prenorm else x


class Router:
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


def source_dependency():
    """Stand-in compiled-kernel source target whose real source is hashed."""


class JitWrapper:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class Tests(unittest.TestCase):
    def install(self, norm, router, context):
        return install_norm_binding(
            norm, context=context, write_tensor=router.write_tensor,
            write_metadata=router.write_metadata, read_source=router.read_source,
            runtime_policy=lambda: {'tf32': False, 'compiler': 'cpu-control'},
            source_targets={'kernel': source_dependency}, replay=None,
        )

    def test_aliasing_captures_original_operands_and_returns_same_object(self):
        norm, router = ResidentNorm(), Router()
        self.install(norm, router, lambda: (0, 1, 7))
        x, g = Tensor([1, 2]), Tensor([3, 4])
        result = norm._forward_method(x, g)
        self.assertIs(result, x)
        self.assertIs(norm.calls[0][0], x)
        values = {stage: data for stage, data, _ in router.tensors}
        self.assertEqual(values['kda-norm-bound-input-x'], [1, 2])
        self.assertEqual(values['kda-norm-bound-input-g'], [3, 4])
        self.assertEqual(values['kda-norm-bound-weight'], [2, 3])
        self.assertEqual(values['kda-norm-bound-output'], [71, 89])
        meta = router.metadata[0][1]
        self.assertEqual(meta['eps'], 0.125)
        self.assertEqual(meta['activation'], 'sigmoid')
        self.assertEqual(meta['call'], {'prenorm': False, 'residual_in_fp32': False,
                                        'residual_present': False})
        self.assertTrue(meta['dispatch']['qualname'].endswith('forward_cuda'))
        self.assertEqual(meta['source_targets']['kernel']['source_sha256'],
                         meta['dispatch']['source_sha256'])
        self.assertEqual(router.metadata[0][2], (0, 1, 7))

    def test_unmarked_invocation_does_no_capture_or_source_io(self):
        norm, router = ResidentNorm(), Router()
        self.install(norm, router, lambda: None)
        x = Tensor([1, 2])
        self.assertIs(norm._forward_method(x, Tensor([2, 3])), x)
        self.assertFalse(router.tensors)
        self.assertFalse(router.metadata)
        self.assertFalse(router.paths)

    def test_residual_flags_and_tuple_result_are_preserved(self):
        norm, router = ResidentNorm(), Router()
        self.install(norm, router, lambda: (0, 0, 0))
        x, g, residual = Tensor([1, 2]), Tensor([3, 4]), Tensor([5, 6])
        out = norm._forward_method(x=x, g=g, residual=residual,
                                   prenorm=True, residual_in_fp32=True)
        self.assertIs(out[0], x)
        self.assertIs(out[1], residual)
        meta = router.metadata[0][1]
        self.assertEqual(meta['call'], {'prenorm': True, 'residual_in_fp32': True,
                                        'residual_present': True})
        stages = {stage for stage, _, _ in router.tensors}
        self.assertIn('kda-norm-bound-residual', stages)
        self.assertIn('kda-norm-bound-residual-output', stages)

    def test_resident_constants_are_read_again_on_every_marked_invocation(self):
        norm, router = ResidentNorm(), Router()
        self.install(norm, router, lambda: (0, 0, 0))
        norm._forward_method(Tensor([1, 2]), Tensor([3, 4]))
        norm.eps = 0.5
        norm.weight.values[0] = 9
        norm.bias = Tensor([0, 1])
        norm._forward_method(Tensor([1, 2]), Tensor([3, 4]))
        self.assertEqual(router.metadata[-1][1]['eps'], 0.5)
        self.assertTrue(router.metadata[-1][1]['bias_present'])
        self.assertEqual([v for s, v, _ in router.tensors if s == 'kda-norm-bound-weight'],
                         [[2, 3], [9, 3]])

    def test_missing_runtime_config_fails_before_original_execution(self):
        for field in ('eps', 'activation', 'elementwise_affine', 'weight', 'bias'):
            with self.subTest(field=field):
                norm, router = ResidentNorm(), Router()
                delattr(norm, field)
                self.install(norm, router, lambda: (0, 0, 0))
                with self.assertRaises(ValueError):
                    norm._forward_method(Tensor([1, 2]), Tensor([3, 4]))
                self.assertFalse(norm.calls)

    def test_double_install_fails_and_restoration_works(self):
        norm, router = ResidentNorm(), Router()
        original = norm._forward_method
        restore = self.install(norm, router, lambda: (0, 0, 0))
        with self.assertRaises(ValueError):
            self.install(norm, router, lambda: (0, 0, 0))
        restore()
        self.assertIs(norm._forward_method, original)

    def test_native_fallback_binds_actual_cuda_source(self):
        norm, router = ResidentNorm(), Router()
        norm._forward_method = norm.forward_native
        self.install(norm, router, lambda: (0, 0, 0))
        norm._forward_method(Tensor([1, 2]), Tensor([3, 4]), prenorm=True,
                             residual=Tensor([5, 6]))
        meta = router.metadata[0][1]
        self.assertTrue(meta['native_residual_fallback_requested'])
        self.assertTrue(meta['fallback_dispatch']['qualname'].endswith('forward_cuda'))

    def test_empty_runtime_policy_and_missing_source_fail_before_call(self):
        for problem in ('policy', 'source'):
            with self.subTest(problem=problem):
                norm, router = ResidentNorm(), Router()
                install_norm_binding(
                    norm, context=lambda: (0, 0, 0), write_tensor=router.write_tensor,
                    write_metadata=router.write_metadata,
                    read_source=(lambda path: b'') if problem == 'source' else router.read_source,
                    runtime_policy=(lambda: {}) if problem == 'policy' else lambda: {'explicit': True},
                    source_targets={'kernel': source_dependency}, replay=None,
                )
                with self.assertRaises(ValueError):
                    norm._forward_method(Tensor([1, 2]), Tensor([3, 4]))
                self.assertFalse(norm.calls)

    def test_invalid_epsilon_or_flags_fail_before_call(self):
        for epsilon in (0, -1, float('nan'), float('inf')):
            with self.subTest(epsilon=epsilon):
                norm, router = ResidentNorm(), Router()
                norm.eps = epsilon
                self.install(norm, router, lambda: (0, 0, 0))
                with self.assertRaises(ValueError):
                    norm._forward_method(Tensor([1, 2]), Tensor([3, 4]))
                self.assertFalse(norm.calls)
        norm, router = ResidentNorm(), Router()
        self.install(norm, router, lambda: (0, 0, 0))
        with self.assertRaises(ValueError):
            norm._forward_method(Tensor([1, 2]), Tensor([3, 4]), prenorm='False')
        self.assertFalse(norm.calls)

    def test_explicit_replay_callback_receives_stable_pre_call_and_output_snapshots(self):
        norm, router = ResidentNorm(), Router()
        calls = []
        def replay(resident, method, snapshots, output, binding, **kwargs):
            calls.append((resident, method, snapshots, output, binding, kwargs))
            snapshots['input-x'].values[0] = 555
            output.values[0] = 999
        install_norm_binding(
            norm, context=lambda: (0, 0, 3), write_tensor=router.write_tensor,
            write_metadata=router.write_metadata, read_source=router.read_source,
            runtime_policy=lambda: {'compiler': 'fixture'},
            source_targets={'kernel': source_dependency}, replay=replay,
        )
        x = Tensor([1, 2])
        output = norm._forward_method(x, Tensor([3, 4]))
        self.assertIs(output, x)
        self.assertEqual(output.values, [71, 89])
        self.assertIs(calls[0][0], norm)
        self.assertEqual(calls[0][4]['eps'], 0.125)
        self.assertEqual(calls[0][5]['context'], (0, 0, 3))

    def test_nested_jit_wrappers_bind_each_class_and_terminal_function(self):
        from reference_norm_binding import _identity
        router = Router()
        identity = _identity(JitWrapper(JitWrapper(source_dependency)), router.read_source)
        self.assertEqual(identity['qualname'], source_dependency.__qualname__)
        self.assertEqual(len(identity['wrappers']), 2)
        self.assertEqual([item['via'] for item in identity['wrappers']], ['fn', 'fn'])
        self.assertTrue(all(item['identity']['qualname'] == 'JitWrapper'
                            for item in identity['wrappers']))
        self.assertTrue(all(len(item['identity']['source_sha256']) == 64
                            for item in identity['wrappers']))

    def test_decorated_function_binds_wrapper_source_and_terminal_function(self):
        from reference_norm_binding import _identity
        def decorated():
            return source_dependency()
        decorated.__wrapped__ = source_dependency
        identity = _identity(decorated, Router().read_source)
        self.assertEqual(identity['qualname'], source_dependency.__qualname__)
        self.assertEqual(identity['wrappers'][0]['via'], '__wrapped__')
        self.assertTrue(identity['wrappers'][0]['identity']['qualname'].endswith('decorated'))

    def test_wrapper_cycles_and_unbound_chains_fail(self):
        from reference_norm_binding import _identity
        cycle = JitWrapper(None)
        cycle.fn = cycle
        for target in (cycle, JitWrapper(None), JitWrapper(object())):
            with self.subTest(target=type(target).__name__), self.assertRaises(ValueError):
                _identity(target, Router().read_source)

    def test_source_binding_and_policy_cannot_be_empty(self):
        norm, router = ResidentNorm(), Router()
        with self.assertRaises(ValueError):
            install_norm_binding(norm, context=lambda: None,
                                 write_tensor=router.write_tensor,
                                 write_metadata=router.write_metadata,
                                 read_source=router.read_source,
                                 runtime_policy=lambda: {}, source_targets={}, replay=None)


if __name__ == '__main__':
    unittest.main()
