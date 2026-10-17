# SPDX-License-Identifier: AGPL-3.0-only
import unittest
import numpy as np
from selector_tail_oracle import oracle


class SelectorTailTests(unittest.TestCase):
    def test_native_budget_reserves_pool_and_raw_tail(self):
        scores = np.array([1, 3, 2], np.float32)
        native, dense = oracle(scores, 9, 8, 4, 11, True)
        np.testing.assert_array_equal(native, [4, 5, 6, 7, 8, -1, -1, -1, -1, -1, -1])
        self.assertEqual(dense, 0)
        legacy, _ = oracle(scores, 9, 8, 4, 11, False)
        np.testing.assert_array_equal(legacy, [0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1])

    def test_positive_zero_precedes_negative_zero(self):
        output, _ = oracle(np.array([-0., 0., -1.], np.float32), 8, 8, 4, 11, True)
        np.testing.assert_array_equal(output[:4], [4, 5, 6, 7])

    def test_dense_boundary_ties_and_invalid_length(self):
        scores = np.array([0, 0, np.nan], np.float32)
        np.testing.assert_array_equal(oracle(scores, 8, 8, 4, 11, True)[0][:4], [0, 1, 2, 3])
        self.assertEqual(oracle(scores, 7, 8, 4, 11, True)[1], 7)
        self.assertEqual(oracle(scores, 8, 8, 4, 11, False)[1], 8)
        for length in [0, 13]:
            output, dense = oracle(scores, length, 8, 4, 11, True)
            np.testing.assert_array_equal(output, np.full(11, -1))
            self.assertEqual(dense, 0)


if __name__ == '__main__':
    unittest.main()
