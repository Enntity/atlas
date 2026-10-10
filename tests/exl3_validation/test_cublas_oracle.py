# SPDX-License-Identifier: AGPL-3.0-only
"""Explicit classic BF16 dispatch geometry, without GPU emulation."""
import unittest
import runpy
import sys
from pathlib import Path
from unittest.mock import patch
from cublas_oracle import classic_arguments


class ClassicArgumentsTest(unittest.TestCase):
    def test_row_major_geometry_matches_traced_torch(self):
        self.assertEqual(classic_arguments(128, 2048, 4096, 68, 99),
                         (1, 0, 2048, 128, 4096, 14, 14, 14, 4096, 4096, 2048, 68, 99))

    def test_explicit_positive_geometry_required(self):
        for shape in [(0, 2048, 4096), (128, -1, 4096), (128, 2048, 2**31)]:
            with self.assertRaises(ValueError):
                classic_arguments(*shape, 68, 99)

    def test_candidate_only_replay_rejects_hidden_torch_import(self):
        with patch.dict(sys.modules, {'torch': object()}):
            with self.assertRaisesRegex(RuntimeError, 'Torch was imported before'):
                runpy.run_path(str(Path(__file__).with_name('shared_classic_oracle.py')), run_name='test_only')

    def test_no_unproven_compute_or_algorithm(self):
        for compute, algorithm in [(0, 99), (68, -1)]:
            with self.assertRaises(ValueError):
                classic_arguments(128, 2048, 4096, compute, algorithm)


if __name__ == '__main__':
    unittest.main()
