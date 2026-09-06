# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only tests: execute the launcher's validation prefix, never node commands."""

import subprocess
import unittest
from pathlib import Path

SOURCE = (Path(__file__).resolve().parents[1] / "start-glm53-ep2.sh").read_text()
PREFIX = SOURCE.split('if [[ "$MTP_SPEC_THINK"', 1)[0]


class C4SparseLauncherTests(unittest.TestCase):
    def launch(self, **changes):
        env = dict(PATH="/usr/bin:/bin", GLM_C4_DECODE="1", GLM_C4_SPARSE="1",
                   GLM_MULTI_SEQ_SPARSE="1", GLM_MULTI_SEQ_SPARSE_GRAPHS="0",
                   NO_DECODE_GRAPHS_MULTISEQ="1", KV_OVERCOMMIT="0", TP_SIZE="2",
                   MAX_BATCH_SIZE="4", MAX_NUM_SEQS="4", MAX_SEQ_LEN="16384",
                   MAX_PREFILL_TOKENS="1024", SPECULATIVE="0")
        env.update(changes)
        return subprocess.run(["/bin/bash", "-c", PREFIX], env=env,
                              capture_output=True, text=True, timeout=5)

    def test_valid_long_c4_and_short_boundary(self):
        for context in ("1", "2048", "2049", "16384"):
            result = self.launch(MAX_SEQ_LEN=context, MAX_PREFILL_TOKENS="1")
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_fail_closed_prerequisites(self):
        for changes in (dict(GLM_C4_DECODE="0"), dict(GLM_MULTI_SEQ_SPARSE="0"),
                        dict(GLM_MULTI_SEQ_SPARSE_GRAPHS="1"),
                        dict(NO_DECODE_GRAPHS_MULTISEQ="0"), dict(SPECULATIVE="1"),
                        dict(TP_SIZE="1"), dict(MAX_BATCH_SIZE="3"),
                        dict(MAX_NUM_SEQS="5"), dict(MAX_SEQ_LEN="16385"),
                        dict(MAX_SEQ_LEN="0"), dict(KV_OVERCOMMIT="1"),
                        dict(MAX_PREFILL_TOKENS="1025"),
                        dict(KDA_MULTI_SEQ="0"), dict(MLA_MULTI_SEQ="0"),
                        dict(GLM_C4_SPARSE="true"), dict(KV_OVERCOMMIT="maybe")):
            with self.subTest(changes=changes):
                self.assertNotEqual(self.launch(**changes).returncode, 0)

    def test_defaults_unchanged_and_flags_forwarded(self):
        self.assertEqual(self.launch(GLM_C4_SPARSE="0", GLM_MULTI_SEQ_SPARSE="0",
                                     NO_DECODE_GRAPHS_MULTISEQ="0", MAX_SEQ_LEN="2048",
                                     KV_OVERCOMMIT="1").returncode, 0)
        self.assertEqual(self.launch(GLM_C4_DECODE="0", GLM_C4_SPARSE="0",
                                     MAX_BATCH_SIZE="3", MAX_NUM_SEQS="3",
                                     NO_DECODE_GRAPHS_MULTISEQ="0", KV_OVERCOMMIT="1").returncode, 0)
        self.assertIn('GLM_C4_SPARSE="${GLM_C4_SPARSE:-0}"', SOURCE)
        self.assertIn('KV_OVERCOMMIT="${KV_OVERCOMMIT:-1}"', SOURCE)
        self.assertEqual(SOURCE.count("-e ATLAS_GLM_C4_SPARSE="), 2)
        self.assertEqual(SOURCE.count("-e ATLAS_KV_OVERCOMMIT="), 2)


if __name__ == "__main__":
    unittest.main()
