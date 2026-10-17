# SPDX-License-Identifier: AGPL-3.0-only
import unittest
import numpy as np
from differential_expert import metrics, validate_input


class DifferentialInputTests(unittest.TestCase):
    def test_captured_input_preserves_bf16_bytes_and_rejects_precision_drift(self):
        original = np.array([[1, -2, 0.5]], dtype=np.float32)
        np.testing.assert_array_equal(validate_input(original), original)
        for bad in (original.astype(np.float16), np.array([[1.1]], np.float32),
                    np.full((1, 1), np.nan, np.float32), np.zeros((0, 1), np.float32),
                    np.zeros((129, 1), np.float32)):
            with self.assertRaises(ValueError):
                validate_input(bad)

    def test_stage_control_detects_changed_arithmetic_and_nonfinite_outputs(self):
        reference = np.array([[1, -2]], np.float32)
        self.assertEqual(metrics(reference, reference.copy(), 0, 0)['violations'], 0)
        changed = np.array([[1.125, -2]], np.float32)
        self.assertEqual(metrics(reference, changed, 0.01, 0.01)['violations'], 1)
        self.assertEqual(metrics(reference, changed, 0.2, 0)['violations'], 0)
        for bad in (np.full_like(reference, np.inf), np.zeros(1, np.float32)):
            with self.assertRaises(ValueError):
                metrics(reference, bad, 0.01, 0.01)


if __name__ == '__main__':
    unittest.main()
