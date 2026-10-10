# SPDX-License-Identifier: AGPL-3.0-only
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from dense_activation_oracle import captured_input


class CaptureTests(unittest.TestCase):
    def test_exact_bf16_input_and_fail_closed_capture_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            raw, sidecar = Path(directory) / 'input.bin', Path(directory) / 'input.json'
            raw.write_bytes(np.array([0x3f80, 0xc000, 0x3f00, 0x8000], '<u2').tobytes())
            sidecar.write_text(json.dumps({'shape': [1, 4], 'dtype': 'bf16-le'}))
            digest = hashlib.sha256(raw.read_bytes()).hexdigest()
            actual = captured_input(raw, sidecar, digest)
            np.testing.assert_array_equal(actual.view(np.uint32), [[0x3f800000, 0xc0000000, 0x3f000000, 0x80000000]])
            with self.assertRaises(ValueError):
                captured_input(raw, sidecar, '0' * 64)
            for metadata in ({'shape': [1, 3], 'dtype': 'bf16-le'},
                             {'shape': [2, 4], 'dtype': 'bf16-le'},
                             {'shape': [1, 4], 'dtype': 'f32-le'}):
                sidecar.write_text(json.dumps(metadata))
                with self.assertRaises(ValueError):
                    captured_input(raw, sidecar, digest)


if __name__ == '__main__':
    unittest.main()
