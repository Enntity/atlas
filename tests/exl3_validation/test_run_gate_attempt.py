# SPDX-License-Identifier: AGPL-3.0-only
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("run_gate_attempt.sh")


class GateAttemptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.hosts = {rank: self.root / f"host{rank}" for rank in (0, 1)}
        for home in self.hosts.values():
            (home / "gate").mkdir(parents=True)
            (home / "tmp").mkdir()
        self.spark_log = self.root / "spark-calls"
        self._write_spark(0)
        self._write_spark(1)
        self._write_docker()
        self._write_cp()
        self.driver = self.root / "driver"
        self.driver.write_text(
            "#!/usr/bin/env bash\n"
            "mkdir -p \"$REMOTE0/gate/$2\"\n"
            "echo result > \"$REMOTE0/gate/$2/result.txt\"\n"
            "for remote in \"$REMOTE0\" \"$REMOTE1\"; do mkdir -p \"$remote/tmp/gate-ref\" \"$remote/tmp/gate-cand\"; echo raw > \"$remote/tmp/gate-ref/logits.f32\"; done\n"
            "if [[ ${DRIVER_SIGNAL:-0} == 1 ]]; then kill -TERM \"$PPID\"; sleep 0.1; exit 0; fi\n"
            "exit \"${DRIVER_STATUS:-0}\"\n"
        )
        self.driver.chmod(0o755)

    def tearDown(self):
        self.tmp.cleanup()

    def _write_spark(self, rank):
        path = self.bin / f"spark{rank}"
        path.write_text(
            "#!/usr/bin/env bash\n"
            "printf '%s %s\\n' \"$(basename \"$0\")\" \"$1\" >> \"$SPARK_LOG\"\n"
            f"HOME=\"$REMOTE{rank}\" PATH=\"$PATH\" bash -c \"$1\"\n"
        )
        path.chmod(0o755)

    def _write_docker(self):
        path = self.bin / "docker"
        path.write_text(
            "#!/usr/bin/env bash\n"
            "if [[ $1 == logs ]]; then echo \"logs $3\"; exit 0; fi\n"
            "if [[ $1 == inspect ]]; then echo '{\\\"container\\\":'\\\"$2\\\"'}'; exit 0; fi\n"
            "exit 2\n"
        )
        path.chmod(0o755)

    def _write_cp(self):
        path = self.bin / "cp"
        path.write_text(
            "#!/usr/bin/env bash\n"
            "if [[ ${FAIL_CAPTURE_COPY:-0} == 1 && $* == *'/gate-ref/.'* ]]; then exit 1; fi\n"
            "exec /bin/cp \"$@\"\n"
        )
        path.chmod(0o755)

    def _run(self, name, status=0, signal=False, fail_capture_copy=False):
        env = dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}",
                   REMOTE0=str(self.hosts[0]), REMOTE1=str(self.hosts[1]),
                   GATE_CAPTURE_ROOT=str(self.root / "host0" / "tmp"),
                   SPARK_LOG=str(self.spark_log), DRIVER_STATUS=str(status),
                   DRIVER_SIGNAL="1" if signal else "0",
                   FAIL_CAPTURE_COPY="1" if fail_capture_copy else "0")
        return subprocess.run(["bash", str(SCRIPT), "ref", name, str(self.driver)], env=env,
                              capture_output=True, text=True)

    def test_preserves_artifacts_captures_and_both_rank_logs_after_failure(self):
        result = self._run("ref-fail", 17)
        self.assertEqual(result.returncode, 17, result.stderr)
        for rank in (0, 1):
            archive = self.hosts[rank] / "gate/attempts/ref-fail"
            self.assertEqual((archive / f"docker-rank{rank}.log").read_text(), f"logs gate-ref-{rank}\n")
            self.assertIn(f"gate-ref-{rank}", (archive / f"inspect-rank{rank}.json").read_text())
            self.assertEqual((archive / "captures/gate-ref/logits.f32").read_text(), "raw\n")
        self.assertEqual((self.hosts[0] / "gate/attempts/ref-fail/artifacts/result.txt").read_text(), "result\n")

    def test_refuses_reused_attempt_or_run_directory(self):
        archive = self.hosts[0] / "gate/attempts/reused"
        archive.mkdir(parents=True)
        marker = archive / "keep.txt"
        marker.write_text("prior evidence")
        result = self._run("reused")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(marker.read_text(), "prior evidence")
        shutil.rmtree(archive)
        (self.hosts[0] / "gate/reused").mkdir()
        result = self._run("reused")
        self.assertNotEqual(result.returncode, 0)

    def test_capture_snapshot_failure_makes_successful_driver_fail(self):
        result = self._run("snapshot-fail", fail_capture_copy=True)
        self.assertNotEqual(result.returncode, 0)

    def test_signal_interruption_runs_preservation_trap(self):
        result = self._run("interrupted", signal=True)
        self.assertEqual(result.returncode, 143, result.stderr)
        archive = self.hosts[0] / "gate/attempts/interrupted"
        self.assertTrue((archive / "docker-rank0.log").exists())
        self.assertTrue((archive / "captures/gate-ref/logits.f32").exists())


if __name__ == "__main__":
    unittest.main()
