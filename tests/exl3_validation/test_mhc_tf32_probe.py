# SPDX-License-Identifier: AGPL-3.0-only
import unittest
import tempfile
import json
from pathlib import Path
import numpy as np
from native_mhc_tf32_probe import validate_case, load_mhc_capture


class ProbeGeometryTests(unittest.TestCase):
    def test_rank3_bf16_residual_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"x.json"
            (path.parent/"x.bin").write_bytes(np.full((2,4,3),0x3f80,dtype="<u2").tobytes())
            path.write_text(json.dumps({"dtype":"bf16-le","shape":[2,4,3],"file":"x.bin"}))
            values,metadata,raw=load_mhc_capture(path)
            self.assertEqual(values.shape,(2,4,3))
            self.assertTrue(np.all(values==1.0))

    def test_malformed_residual_capture_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"x.json"
            (path.parent/"x.bin").write_bytes(np.full(24,0x3f80,dtype="<u2").tobytes())
            for metadata in ({"dtype":"bf16-le","shape":[2,3,4],"file":"x.bin"},
                             {"dtype":"i32-le","shape":[2,4,3],"file":"x.bin"},
                             {"dtype":"bf16-le","shape":[2,4,3],"file":"../x.bin"},
                             {"dtype":"bf16-le","shape":[2,4,4],"file":"x.bin"}):
                path.write_text(json.dumps(metadata))
                with self.assertRaises(ValueError):load_mhc_capture(path)
            path.write_text(json.dumps({"dtype":"bf16-le","shape":[2,4,3],"file":"x.bin"}))
            (path.parent/"x.bin").write_bytes(np.full(24,0x7f80,dtype="<u2").tobytes())
            with self.assertRaises(ValueError):load_mhc_capture(path)

    def test_registered_geometry(self):
        self.assertEqual(validate_case(10,16384,48),(1,48))
        self.assertEqual(validate_case(128,16384,24),(1,24))

    def test_invalid_fragment_or_split_geometry_rejected(self):
        for args in ((0,16384,48),(10,16385,48),(10,16384,0),(10,128,3)):
            with self.assertRaises(ValueError):validate_case(*args)


if __name__=='__main__':unittest.main()
