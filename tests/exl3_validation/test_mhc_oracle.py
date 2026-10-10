# SPDX-License-Identifier: AGPL-3.0-only
import unittest
from types import SimpleNamespace

from mhc_oracle import validate_fused_geometry


class MhcOracleBoundaryTests(unittest.TestCase):
    def tensors(self):
        return [SimpleNamespace(shape=shape) for shape in (
            (10, 4096), (10, 4, 4096), (10, 4, 1), (10, 4, 4),
            (24, 16384), (3,), (24,), (4096,))]

    def test_registered_shapes_accept_without_cuda(self):
        validate_fused_geometry(10, 4096, *self.tensors())

    def test_no_larger_batch_can_reuse_smaller_output_arenas(self):
        values = self.tensors()
        values[0].shape, values[1].shape = (11, 4096), (11, 4, 4096)
        with self.assertRaises(ValueError):
            validate_fused_geometry(10, 4096, *values)

    def test_every_tensor_axis_is_checked_before_a_gpu_launch(self):
        for index in range(8):
            with self.subTest(index=index):
                values = self.tensors()
                values[index].shape = (*values[index].shape, 1)
                with self.assertRaises(ValueError):
                    validate_fused_geometry(10, 4096, *values)
        for rows, hidden in ((0, 4096), (10, 0), (10, 4095)):
            with self.assertRaises(ValueError):
                validate_fused_geometry(rows, hidden, *self.tensors())


if __name__ == '__main__':
    unittest.main()
