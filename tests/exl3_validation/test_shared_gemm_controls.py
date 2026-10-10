# SPDX-License-Identifier: AGPL-3.0-only
import unittest
from shared_gemm_controls import independent_gemm_controls, reduction_controls


class Controls(unittest.TestCase):
    def test_independent_down_uses_exact_reference_activation_for_both_libraries(self):
        calls = []
        def gemm(act, weight, library):
            calls.append((act, weight, library))
            return (act, weight, library), {'library': library}
        arrays, metadata = independent_gemm_controls('input', 'gate-up', 'reference-act', 'down',
            {'candidate': '/candidate', 'reference': '/reference'}, gemm)
        self.assertEqual(calls, [('input', 'gate-up', '/candidate'),
                                ('reference-act', 'down', '/candidate'),
                                ('input', 'gate-up', '/reference'),
                                ('reference-act', 'down', '/reference')])
        self.assertEqual(set(arrays), {'candidate_gate_up', 'candidate_independent_down',
                                     'reference_gate_up', 'reference_independent_down'})
        self.assertEqual(set(metadata), set(arrays))

    def test_both_library_bindings_are_explicit(self):
        with self.assertRaises(ValueError):
            independent_gemm_controls(1, 2, 3, 4, {'candidate': 'x'}, lambda *a: None)

    def test_reduction_flag_is_restored_when_diagnostic_fails(self):
        value = [True]
        def set_flag(flag): value[0] = flag
        def run(flag): raise RuntimeError('operator failed')
        with self.assertRaisesRegex(RuntimeError, 'operator failed'):
            reduction_controls(lambda: value[0], set_flag, run)
        self.assertTrue(value[0])

    def test_reduction_controls_exercise_both_explicit_flags(self):
        value = [True]
        def set_flag(flag): value[0] = flag
        result = reduction_controls(lambda: value[0], set_flag, lambda flag: (flag, value[0]))
        self.assertEqual(result, {False: (False, False), True: (True, True)})
        self.assertTrue(value[0])


if __name__ == '__main__': unittest.main()
