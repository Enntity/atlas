# SPDX-License-Identifier: AGPL-3.0-only
import unittest
import numpy as np
from index_quant_oracle import finite_e4m3_values


class E4M3RepresentationTests(unittest.TestCase):
    def test_finite_fp8_power_two_dequant_is_exact_bf16(self):
        values = finite_e4m3_values()
        self.assertEqual(len(values), 127)
        self.assertEqual(values[-1], 448)
        self.assertEqual(values[1], 2 ** -9)
        for exponent in (-22, -10, 0, 10, 100):
            scaled = np.concatenate([values, -values]) * np.float32(2. ** exponent)
            self.assertTrue(np.isfinite(scaled).all())
            np.testing.assert_array_equal(scaled.view(np.uint32) & 65535, 0)


class ScaleBoundaryTests(unittest.TestCase):
    def test_single_bf16_impulse_brackets_post_hadamard_power_two_maxima(self):
        from index_quant_oracle import scale_boundary_impulses
        x, metadata = scale_boundary_impulses((-22, -1, 0, 1, 10, 100))
        self.assertEqual(x.shape, (36, 128))
        np.testing.assert_array_equal(x.view(np.uint32) & 65535, 0)
        self.assertTrue(np.isfinite(x).all())

        for item in metadata:
            lo, hi = item['positive_rows']
            self.assertLess(item['hadamard_maxima'][0], item['boundary'])
            self.assertGreater(item['hadamard_maxima'][1], item['boundary'])
            self.assertEqual(x[lo, 0], -x[item['negative_rows'][0], 0])
            self.assertLess(x[lo, 0], x[hi, 0])
            eq = item['equality_rows'][0]
            self.assertEqual(item['equality_hadamard_maximum'], item['boundary'])
            self.assertEqual(np.count_nonzero(x[eq]), 2)


if __name__ == '__main__':
    unittest.main()
