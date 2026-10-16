# SPDX-License-Identifier: AGPL-3.0-only
"""CPU unit tests for bench_tp2.archive_container_log.

The 400-line tail kept in container-r{rank}.log drops the NCCL init/transport
lines, so the full log is archived beside it. No docker, no GPU, stdlib only.
"""

import gzip
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import bench_tp2


def synthetic_nccl_log(payload_bytes):
    """A log shaped like NCCL_DEBUG=INFO output, `payload_bytes` bytes of body."""
    head = ('NCCL_DEBUG=INFO NCCL_2.21: ncclInitComms: 2 threads, 1 devices\n'
            'NCCL_DEBUG=INFO Using network IB\n')
    body = ('NCCL_DEBUG=INFO Channel 00/2: 0 1 [send] via NET/IB, conn 0, step 0\n'
            * (1 + payload_bytes // 64))[:payload_bytes]
    tail = 'NCCL_DEBUG=INFO comm 0x... rank 0 nranks 2 - Init COMPLETE\n'
    return head + body + tail


class ArchiveContainerLogTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.run = Path(self._tmp.name)

    def test_over_threshold_log_is_gzipped_and_round_trips(self):
        full = synthetic_nccl_log(bench_tp2.ARCHIVE_GZIP_THRESHOLD + 4096)
        path = bench_tp2.archive_container_log(0, full, self.run)
        self.assertTrue(str(path).endswith('.gz'), f'expected a .gz archive, got {path.name}')
        self.assertTrue(gzip.decompress(path.read_bytes()).decode('utf-8') == full,
                        'archive lost content: a tail-only write would pass this?')
        self.assertIn('ncclInitComms', gzip.decompress(path.read_bytes()).decode('utf-8'),
                      'the init line scrolls past any tail; it must survive archiving')
        self.assertIn('Init COMPLETE', gzip.decompress(path.read_bytes()).decode('utf-8'))
        self.assertLess(path.stat().st_size, len(full), 'archive was not actually compressed')

    def test_threshold_is_in_bytes_not_characters(self):
        # One byte over the threshold gzips even though the extra char is ASCII.
        full = 'x' * (bench_tp2.ARCHIVE_GZIP_THRESHOLD + 1)
        path = bench_tp2.archive_container_log(1, full, self.run)
        self.assertTrue(str(path).endswith('.gz'))
        self.assertEqual(gzip.decompress(path.read_bytes()).decode('utf-8'), full)

    def test_small_log_is_left_plain_and_uncompressed(self):
        full = 'NCCL_DEBUG=INFO Init COMPLETE\nshort rank1 log\n'
        path = bench_tp2.archive_container_log(1, full, self.run)
        self.assertFalse(str(path).endswith('.gz'), f'small log was gzipped: {path.name}')
        self.assertTrue(path.is_file())
        self.assertEqual(path.read_text(encoding='utf-8'), full)

    def test_archive_does_not_shadow_the_tail_summary(self):
        full = 'anything\n'
        path = bench_tp2.archive_container_log(0, full, self.run)
        self.assertNotEqual(path.name, 'container-r0.log',
                            'archive must not overwrite the tail consumers read')


if __name__ == '__main__':
    unittest.main()
