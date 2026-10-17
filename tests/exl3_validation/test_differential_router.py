# SPDX-License-Identifier: AGPL-3.0-only
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from differential_router import compare, digest, load_capture, validate_weights


class RouterBoundaryTests(unittest.TestCase):
    def test_typed_capture_preserves_exact_bf16_and_rejects_wrong_size(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = root / 'input.bin'
            raw.write_bytes(np.array([0x3f80, 0xc000], '<u2').tobytes())
            meta = root / 'input.json'
            meta.write_text(json.dumps({'dtype': 'bf16-le', 'shape': [1, 2], 'file': raw.name}))
            values, _, _ = load_capture(meta)
            np.testing.assert_array_equal(values, [[1, -2]])
            raw.write_bytes(b'\0')
            with self.assertRaises(ValueError): load_capture(meta)
            meta.write_text(json.dumps({'dtype': 'bf16-le', 'shape': [1, 2], 'file': '../input.bin'}))
            with self.assertRaises(ValueError): load_capture(meta)

    def test_checkpoint_binding_and_representability_are_explicit(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'weights.npy'
            weights = np.array([[1, -2]], np.float16)
            np.save(path, weights)
            metadata = {'shape': [1, 2], 'file_sha256': digest(path),
                        'selected_tensor_sha256': hashlib.sha256(weights.tobytes()).hexdigest()}
            np.testing.assert_array_equal(validate_weights(weights, metadata, path), weights)
            metadata['file_sha256'] = '0' * 64
            with self.assertRaises(ValueError): validate_weights(weights, metadata, path)
            weights = np.array([[1.0009765625, -2]], np.float16)
            np.save(path, weights)
            metadata.update(file_sha256=digest(path),
                            selected_tensor_sha256=hashlib.sha256(weights.tobytes()).hexdigest())
            with self.assertRaises(ValueError): validate_weights(weights, metadata, path)

    def test_comparison_reports_rounding_control_and_rejects_nonfinite(self):
        reference = np.array([[1, 2]], np.float32)
        self.assertTrue(compare(reference, reference)['exact'])
        candidate = np.array([[1.001, 2]], np.float32)
        self.assertEqual(compare(reference, candidate)['unequal'], 1)
        with self.assertRaises(ValueError): compare(reference, candidate[:, :1])
        candidate[0, 0] = np.nan
        with self.assertRaises(ValueError): compare(reference, candidate)


if __name__ == '__main__':
    unittest.main()
