#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""CPU regression tests for the campaign's complete native-log fault gate."""
import unittest

from check_glm53_native_log import fault_lines


class NativeLogGateTests(unittest.TestCase):
    def test_benign_startup_error_word_and_cuda_info(self):
        self.assertEqual(fault_lines([
            '2026-09-08T00:00:00Z INFO bootstrap: Connection refused (os error 111)',
            '2026-09-08T00:00:01Z INFO CUDA backend ready',
        ]), [])

    def test_actual_trace_and_ledger_records(self):
        for marker in ('GLM MTP HIDDEN_TRACE', 'GLM MTP K5_LEDGER'):
            line = f'2026-09-08T00:00:00Z INFO model: {marker} generation=2'
            self.assertEqual(fault_lines([line]), [line])

    def test_plain_and_ansi_error_level(self):
        for level in ('ERROR', '\x1b[31mERROR\x1b[0m'):
            line = f'2026-09-08T00:00:00Z {level} comm: launch failed'
            self.assertEqual(fault_lines([line]), [line])

    def test_faults_without_error_level(self):
        for fault in ('unhealthy', 'thread panicked at', 'illegal memory access',
                      'out of memory', 'CUDA_ERROR_LAUNCH_FAILED',
                      'CudaError(IllegalAddress)', 'NCCL error', 'device-side assert'):
            with self.subTest(fault=fault):
                self.assertEqual(fault_lines([fault]), [fault])


if __name__ == '__main__':
    unittest.main()
