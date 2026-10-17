# SPDX-License-Identifier: AGPL-3.0-only
import unittest
import numpy as np
from shared_reconstruction import byte_metrics, shard_weight


class SharedReconstructionBoundaries(unittest.TestCase):
    def test_column_and_row_shards_preserve_original_weight_axes(self):
        value = np.arange(32, dtype=np.uint16).reshape(8,4)
        np.testing.assert_array_equal(shard_weight(value,'gate',1),value[4:])
        np.testing.assert_array_equal(shard_weight(value,'up',0),value[:4])
        np.testing.assert_array_equal(shard_weight(value,'down',1),value[:,2:])
        with self.assertRaises(ValueError): shard_weight(value[:7],'gate',0)
        with self.assertRaises(ValueError): shard_weight(value[:,:3],'down',0)
        with self.assertRaises(ValueError): shard_weight(value,'unknown',0)

    def test_byte_gate_retains_signed_zero_and_rejects_nonfinite(self):
        positive = np.array([[0,0x3f80]],np.uint16)
        negative = np.array([[0x8000,0x3f80]],np.uint16)
        comparison = byte_metrics(positive,negative)
        self.assertFalse(comparison['exact'])
        self.assertEqual(comparison['unequal_bytes_elements'],1)
        self.assertEqual(comparison['max_abs'],0)
        with self.assertRaises(ValueError): byte_metrics(positive,negative.astype(np.float32))
        with self.assertRaises(ValueError): byte_metrics(positive,np.array([[0,0x7f80]],np.uint16))

    def test_fp16_before_bf16_round_is_observable(self):
        def bf16(value):
            bits = np.asarray(value,np.float32).view(np.uint32)
            return ((bits+32767+((bits>>16)&1))>>16).astype(np.uint16)
        value = np.array([1.00390625+0.0001],np.float32)
        self.assertEqual(int(bf16(value)[0]),0x3f81)
        self.assertEqual(int(bf16(value.astype(np.float16).astype(np.float32))[0]),0x3f80)


if __name__ == '__main__':
    unittest.main()
