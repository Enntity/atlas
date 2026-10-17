# SPDX-License-Identifier: AGPL-3.0-only
"""CPU regressions for benchmark failure propagation and executable provenance."""
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import bench_tp2


class BenchLifecycleTests(unittest.TestCase):
    def run_bench(self, child_rc=0, hashes=('a' * 64, 'a' * 64), extra=None):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        calls = []

        def remote(rank, command, **kwargs):
            calls.append((rank, command))
            if 'docker ps' in command:
                return ''
            if 'image inspect' in command:
                return 'sha256:image'
            if 'sha256sum' in command:
                return hashes[rank] + '  /binary'
            return ''

        argv = ['bench', '--image', 'test', '--label', 'case', '--out', tmp.name,
                '--binary', '/binary', '--binary-source-commit', 'b' * 40]
        argv += extra or []
        with patch.object(sys, 'argv', argv), patch.object(bench_tp2, 'remote', side_effect=remote), \
                patch.object(bench_tp2, 'create_argv', return_value=['docker', 'create', 'test']), \
                patch.object(bench_tp2.subprocess, 'Popen'), \
                patch.object(bench_tp2.subprocess, 'run', side_effect=[
                    SimpleNamespace(returncode=0, stdout=bench_tp2.MODEL),
                    SimpleNamespace(returncode=child_rc, stdout='', stderr='')]):
            try:
                result = bench_tp2.main()
            except RuntimeError as exc:
                result = exc
        run = next(Path(tmp.name).iterdir())
        return result, run, calls

    def test_failed_child_returns_failure_after_both_containers_removed(self):
        result, run, calls = self.run_bench(child_rc=3)
        self.assertEqual(result, 3)
        self.assertEqual(json.loads((run / 'meta.json').read_text())['perf_rc'], 3)
        self.assertEqual(sum('docker rm -f' in cmd for _, cmd in calls), 2)

    def test_success_fingerprints_actual_binary_on_both_ranks(self):
        result, run, _ = self.run_bench()
        self.assertEqual(result, 0)
        fp = json.loads((run / 'fingerprint.json').read_text())
        self.assertEqual(fp['source_commit'], 'b' * 40)
        self.assertEqual(fp['binary_sha256_r0'], 'a' * 64)
        self.assertEqual(fp['binary_sha256_r1'], 'a' * 64)

    def test_mismatched_peer_binaries_fail_before_container_creation(self):
        result, run, calls = self.run_bench(hashes=('a' * 64, 'c' * 64))
        self.assertIsInstance(result, RuntimeError)
        self.assertIn('binary', str(result))
        self.assertFalse(any(cmd.startswith('docker create') for _, cmd in calls))
        self.assertFalse((run / 'fingerprint.json').exists())

    def test_bind_mount_requires_explicit_source_commit_before_any_remote_io(self):
        with patch.object(sys, 'argv', ['bench', '--image', 'test', '--label', 'case',
                                       '--out', '/tmp/unused', '--binary', '/binary']), \
                patch.object(bench_tp2, 'remote') as remote:
            with self.assertRaises(SystemExit) as exc:
                bench_tp2.main()
        self.assertEqual(exc.exception.code, 2)
        remote.assert_not_called()


class ProfileArgvTests(unittest.TestCase):
    DESCRIPTOR = ['docker', 'create', '--name', 'old', '--network', 'host',
                  '--env', 'NCCL_IB_HCA=rocep1s0f0', '--env', 'CUDA_CACHE_PATH=/tmp/cuda-driver-cache',
                  '--volume', '/h/cache:/tmp/cuda-driver-cache', '--env', 'ATLAS_KDA_FLA_REFERENCE=1',
                  '--env', 'ATLAS_EXL3_LOGITS_DIR=/tmp/capture',
                  '--volume', '/home/colum/glm53-local:/root/.cache/huggingface:ro',
                  '--volume', '/tmp/x/capture:/tmp/capture', bench_tp2.DESCRIPTOR_IMAGE,
                  'serve', bench_tp2.MODEL, '--tp-size', '2', '--kv-cache-dtype', 'fp8_g128']

    def argv(self, profile, **kwargs):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        (Path(tmp.name) / 'cand-rank1-create-command.json').write_text(json.dumps(self.DESCRIPTOR))
        with patch.object(bench_tp2, 'DESCRIPTORS', Path(tmp.name)):
            return bench_tp2.create_argv(1, 'img', 'new', ['ATLAS_X=1'], '/bin/spark', None, '15-19',
                                         profile, **kwargs)

    def test_exl3_profile_keeps_descriptor_serve_command(self):
        out = self.argv('glm53-exl3-tp2')
        self.assertIn('ATLAS_KDA_FLA_REFERENCE=1', out)
        self.assertEqual(out[out.index('img') + 1:], self.DESCRIPTOR[-6:])

    def test_nvfp4_profile_drops_exl3_env_and_serves_ep2(self):
        out = self.argv('glm53-nvfp4-ep2')
        envs = [out[i + 1] for i, a in enumerate(out) if a == '--env']
        self.assertEqual(envs, ['NCCL_IB_HCA=rocep1s0f0', 'ATLAS_X=1'])
        vols = [out[i + 1] for i, a in enumerate(out) if a == '--volume']
        self.assertEqual(vols, [f'{bench_tp2.NVFP4_HF_ROOT}:/root/.cache/huggingface:ro',
                                '/bin/spark:/usr/local/bin/spark:ro'])
        self.assertEqual(out[out.index('--cpuset-cpus') + 1], '15-19')
        serve = out[out.index('img') + 1:]
        self.assertEqual(serve, bench_tp2.nvfp4_ep2_serve(1))
        self.assertEqual(serve[serve.index('--ep-size') + 1], '2')
        self.assertEqual(serve[serve.index('--tp-size') + 1], '1')
        self.assertNotIn('--kv-cache-dtype', serve)


if __name__ == '__main__':
    unittest.main()
