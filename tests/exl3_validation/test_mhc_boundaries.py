# SPDX-License-Identifier: AGPL-3.0-only
import unittest
import numpy as np
from mhc_boundaries import normalize_collapse, round_bf16


class MhcBoundaryTests(unittest.TestCase):
    def test_unrounded_square_sum_is_distinct_from_rounded_square_sum(self):
        collapse = np.array([[1.003, 2.007, -.501]], dtype=np.float32)
        weight = np.array([1., .5, 2.], dtype=np.float32)
        actual = normalize_collapse(collapse, weight, 1e-6, square_sum='unrounded')
        inverse = np.float32(1.) / np.sqrt(np.mean(collapse * collapse, axis=-1,
                                                 keepdims=True) + np.float32(1e-6))
        expected = round_bf16(round_bf16(collapse) * inverse * round_bf16(weight))
        np.testing.assert_array_equal(actual, expected)
        wrong = normalize_collapse(collapse, weight, 1e-6, square_sum='rounded')
        self.assertFalse(np.array_equal(actual, wrong))

    def test_rounding_ties_and_signed_zero(self):
        bits = np.array([0x3f808000, 0x3f818000, 0x80000000, 0x00000000], dtype=np.uint32)
        rounded = round_bf16(bits.view(np.float32)).view(np.uint32)
        np.testing.assert_array_equal(rounded, [0x3f800000, 0x3f820000, 0x80000000, 0])

    def test_invalid_contract_fails_closed(self):
        values, weights = np.ones((2, 3), np.float32), np.ones(3, np.float32)
        for epsilon, policy in ((0., 'unrounded'), (float('nan'), 'unrounded'),
                                (1e-6, 'guess')):
            with self.assertRaises(ValueError):
                normalize_collapse(values, weights, epsilon, square_sum=policy)
        for x, w in ((np.ones((2, 0), np.float32), weights),
                     (values, np.ones(2, np.float32)),
                     (values * np.nan, weights)):
            with self.assertRaises(ValueError):
                normalize_collapse(x, w, 1e-6, square_sum='unrounded')


if __name__ == '__main__':
    unittest.main()
