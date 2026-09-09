# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only launcher policy tests; execute only its pure validation prefix."""

import shlex
import subprocess
import unittest
from pathlib import Path


SOURCE = (Path(__file__).resolve().parents[1] / "start-glm53-ep2.sh").read_text()
PREFIX = SOURCE.split('if [[ "$MTP_SPEC_THINK"', 1)[0]


class Batch4LauncherTests(unittest.TestCase):
    def launch(self, **changes):
        env = dict(PATH="/usr/bin:/bin", GLM_C4_DECODE="1", TP_SIZE="2",
                   MAX_BATCH_SIZE="4", MAX_NUM_SEQS="4", MAX_SEQ_LEN="2048",
                   GLM_MLA_BATCH4="1", SPECULATIVE="0")
        env.update(changes)
        return subprocess.run(["/bin/bash", "-c", PREFIX], env=env,
                              capture_output=True, text=True, timeout=5)

    def test_valid_c4_and_default_off_legacy(self):
        self.assertEqual(self.launch().returncode, 0)
        self.assertEqual(self.launch(GLM_MLA_BATCH4="0").returncode, 0)
        self.assertEqual(self.launch(GLM_C4_DECODE="0", GLM_MLA_BATCH4="0",
                                     MAX_BATCH_SIZE="3", MAX_NUM_SEQS="3").returncode, 0)

    def test_flag_requires_c4_and_binary_value(self):
        self.assertNotEqual(self.launch(GLM_C4_DECODE="0", MAX_BATCH_SIZE="3",
                                        MAX_NUM_SEQS="3").returncode, 0)
        for flag in ("true", "2", "-1"):
            self.assertNotEqual(self.launch(GLM_MLA_BATCH4=flag).returncode, 0)

    def test_both_ranks_receive_default_off_flag(self):
        self.assertIn('GLM_MLA_BATCH4="${GLM_MLA_BATCH4:-0}"', SOURCE)
        self.assertEqual(SOURCE.count("-e ATLAS_GLM_MLA_BATCH4="), 2)


class IndependentLauncherTests(unittest.TestCase):
    def launch(self, **changes):
        env = dict(PATH="/usr/bin:/bin", GLM_INDEPENDENT_DECODE="1", TP_SIZE="2",
                   MAX_BATCH_SIZE="6", MAX_NUM_SEQS="6", MAX_SEQ_LEN="2048",
                   GLM_K5_HC_CUBLAS="0", SWAP_SPACE_GB="0",
                   HEAD_IP="192.0.2.1", WORKER_IP="192.0.2.2")
        env.update(changes)
        for name in [name for name, value in env.items() if value is None]:
            del env[name]
        # Function stubs intercept every external node command, including rm.
        # Execute the actual complete shell argument construction, not a copy.
        stubs = '''docker() { printf 'LOCAL '; printf '%q ' "$@"; printf '\\n'; }
ssh() { printf 'REMOTE '; printf '%q ' "$@"; printf '\\n'; }
'''
        return subprocess.run(["/bin/bash", "-c", stubs + SOURCE], env=env,
                              capture_output=True, text=True, timeout=5)

    def test_selected_all_capacities_forward_both_ranks(self):
        for width in range(2, 9):
            with self.subTest(width=width):
                result = self.launch(MAX_BATCH_SIZE=str(width), MAX_NUM_SEQS=str(width))
                self.assertEqual(result.returncode, 0, result.stderr)
                local = next(shlex.split(line)[1:] for line in result.stdout.splitlines()
                             if line.startswith("LOCAL run "))
                remote = next(shlex.split(line)[2] for line in result.stdout.splitlines()
                              if line.startswith("REMOTE ") and "docker" in line and "run" in line)
                for command in (local, shlex.split(remote)):
                    self.assertIn("ATLAS_GLM_INDEPENDENT_DECODE=1", command)
                    self.assertIn("ATLAS_GLM_K5_HC_CUBLAS=0", command)
                    self.assertIn("ATLAS_EP_PROTOCOL=v2", command)
                    self.assertEqual(command[command.index("--max-batch-size") + 1], str(width))
                    self.assertEqual(command[command.index("--max-num-seqs") + 1], str(width))
                    self.assertEqual(command[command.index("--kv-cache-dtype") + 1], "bf16")
                    self.assertNotIn("--speculative", command)

    def test_selected_conflicts_stop_before_node_commands(self):
        for changes in (dict(GLM_INDEPENDENT_DECODE=""), dict(GLM_INDEPENDENT_DECODE="true"),
                        dict(GLM_INDEPENDENT_DECODE="2"), dict(TP_SIZE="1"),
                        dict(MAX_BATCH_SIZE="1", MAX_NUM_SEQS="1"),
                        dict(MAX_BATCH_SIZE="9", MAX_NUM_SEQS="9"),
                        dict(MAX_BATCH_SIZE="06"), dict(MAX_NUM_SEQS="7"),
                        dict(MAX_SEQ_LEN="0"), dict(MAX_SEQ_LEN="2049"),
                        dict(SPECULATIVE="1"), dict(GLM_K5_HC_CUBLAS="1"),
                        dict(GLM_K5_HC_CUBLAS=None), dict(SWAP_SPACE_GB="3"),
                        dict(GLM_MULTI_SEQ_SPARSE="1"), dict(GLM_C4_SPARSE="1")):
            with self.subTest(changes=changes):
                result = self.launch(**changes)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("GLM_INDEPENDENT_DECODE", result.stderr)
                self.assertNotIn("LOCAL ", result.stdout)
                self.assertNotIn("REMOTE ", result.stdout)

    def test_selected_does_not_require_old_per_width_flags(self):
        result = self.launch(GLM_C4_DECODE="1", GLM_C4_GROUPED_MOE="1",
                             GLM_MLA_BATCH4="1", KDA_MULTI_SEQ="0", MLA_MULTI_SEQ="0")
        self.assertEqual(result.returncode, 0, result.stderr)
        invalid = self.launch(MAX_PREFILL_TOKENS="2049")
        self.assertNotEqual(invalid.returncode, 0)
        self.assertNotIn("LOCAL ", invalid.stdout)
        self.assertNotIn("REMOTE ", invalid.stdout)

    def test_off_keeps_legacy_width_gate_and_temporal_default(self):
        for flag in (None, "0"):
            rejected = self.launch(GLM_INDEPENDENT_DECODE=flag)
            self.assertNotEqual(rejected.returncode, 0)
            result = self.launch(GLM_INDEPENDENT_DECODE=flag, MAX_BATCH_SIZE="3",
                                 MAX_NUM_SEQS="5", GLM_K5_HC_CUBLAS=None)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("ATLAS_GLM_K5_HC_CUBLAS=1", result.stdout)


if __name__ == "__main__":
    unittest.main()
