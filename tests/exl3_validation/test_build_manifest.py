# SPDX-License-Identifier: AGPL-3.0-only
import json
import tempfile
import unittest
from pathlib import Path

from build_manifest import (
    CLOSURES_PREFIX,
    build_manifest,
    main,
    parse_target_closures,
)

TARGET_KEY = "gb10/glm-5.3-flash/exl3"
CLOSURES = (
    '{"gb10/glm-5.3-flash/exl3":{"hash":"h1","arch":"sm_121f",'
    '"compiler":"Build cuda_13.0.r13.0/compiler.36260728_0",'
    '"flags":["--fmad=false","-DTQ_PLUS_SIGNS"],"module_flags":{"exl3_mhc":["--fmad=true"]}}}'
)
PTX_MHC = b".version 8.0\n.visible .entry mhc() {}\n"
PTX_LIN = b".version 8.0\n.visible .entry lin() {}\n"


def make_tree(root, *, second_dir=False, spark_missing_ptx=False, example_partial=False):
    release = root / "release"
    out = release / "build/atlas-kernels-abc/out"
    out.mkdir(parents=True)
    (out / "target_ptx.rs").write_text('pub const M: &str = "glm-5.3-flash";')
    (out / "t0__exl3_mhc.ptx").write_bytes(PTX_MHC)
    (out / "t0__exl3_linear_mma.ptx").write_bytes(PTX_LIN)
    (out.parent / "output").write_text(
        f"cargo:rerun-if-env-changed=X\n{CLOSURES_PREFIX}{CLOSURES}\n")
    if second_dir:
        out2 = release / "build/atlas-kernels-def/out"
        out2.mkdir(parents=True)
        (out2 / "target_ptx.rs").write_text("stub")
        (out2.parent / "output").write_text(f"{CLOSURES_PREFIX}{CLOSURES}\n")
    spark = b"AAA" + PTX_MHC + b"BBB" + PTX_LIN
    if spark_missing_ptx:
        spark = b"AAA"
    (release / "spark").write_bytes(spark)
    (release / "examples").mkdir()
    example = PTX_MHC if example_partial else PTX_MHC + PTX_LIN
    (release / "examples/exl3_moe_scratch_check").write_bytes(example)
    fingerprint = root / "fingerprint"
    fingerprint.mkdir()
    (fingerprint / "rustc.txt").write_text("rustc 1.97.1\n")
    (fingerprint / "cargo.txt").write_text("cargo 1.97.1\n")
    (fingerprint / "nvcc.txt").write_text("nvcc 13.0\n")
    return release, fingerprint


class TargetClosureTests(unittest.TestCase):
    def test_extracts_single_target(self):
        text = f"x\n{CLOSURES_PREFIX}{CLOSURES}\n"
        closure = parse_target_closures(text, TARGET_KEY)
        self.assertEqual(closure["arch"], "sm_121f")
        self.assertEqual(closure["module_flags"], {"exl3_mhc": ["--fmad=true"]})

    def test_missing_or_duplicate_lines_fail(self):
        with self.assertRaises(ValueError):
            parse_target_closures("no closures here", TARGET_KEY)
        text = f"{CLOSURES_PREFIX}{CLOSURES}\n{CLOSURES_PREFIX}{CLOSURES}\n"
        with self.assertRaises(ValueError):
            parse_target_closures(text, TARGET_KEY)

    def test_missing_target_key_fails(self):
        other = '{"gb10/other/exl3":{}}'
        with self.assertRaises(ValueError):
            parse_target_closures(f"{CLOSURES_PREFIX}{other}\n", TARGET_KEY)


class BuildManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def build(self, **kwargs):
        release, fingerprint = make_tree(self.root, **kwargs)
        return build_manifest(release, fingerprint, TARGET_KEY, "g" * 40, "s" * 64)

    def test_manifest_fields(self):
        manifest = self.build()
        self.assertEqual(manifest["git_commit"], "g" * 40)
        self.assertEqual(manifest["source_snapshot_sha256"], "s" * 64)
        self.assertEqual(manifest["target_key"], TARGET_KEY)
        self.assertIn("rustc", manifest["toolchain"])
        self.assertEqual(manifest["compiler_flags"]["hash"], "h1")
        self.assertEqual(
            set(manifest["binaries"]), {"spark", "exl3_moe_scratch_check"})
        self.assertEqual(
            set(manifest["embedded_ptx"]), {"exl3_mhc", "exl3_linear_mma"})

    def test_ptx_offsets_point_into_binaries(self):
        manifest = self.build()
        spark = (self.root / "release/spark").read_bytes()
        for module, entry in manifest["embedded_ptx"].items():
            offset = entry["embedded_byte_offsets"]["spark"]
            self.assertEqual(spark[offset:offset + entry["bytes"]], (
                PTX_MHC if module == "exl3_mhc" else PTX_LIN))

    def test_main_writes_manifest_and_fingerprints(self):
        release, fingerprint = make_tree(self.root)
        out = self.root / "build-manifest.json"
        rc = main(["--release-dir", str(release), "--fingerprint-dir", str(fingerprint),
                   "--target-key", TARGET_KEY, "--git-sha", "g" * 40,
                   "--source-snapshot-sha256", "s" * 64, "--out", str(out)])
        self.assertEqual(rc, 0)
        manifest = json.loads(out.read_text())
        spark_sha = manifest["binaries"]["spark"]["sha256"]
        line = (fingerprint / "spark.sha256").read_text()
        self.assertEqual(line.split()[0], spark_sha)
        self.assertIn("examples/exl3_moe_scratch_check",
                      (fingerprint / "exl3_moe_scratch_check.sha256").read_text())

    def test_unembedded_ptx_fails(self):
        with self.assertRaises(ValueError):
            self.build(spark_missing_ptx=True)

    def test_example_may_embed_a_subset(self):
        manifest = self.build(example_partial=True)
        offsets = manifest["embedded_ptx"]["exl3_linear_mma"]["embedded_byte_offsets"]
        self.assertIsNone(offsets["exl3_moe_scratch_check"])
        self.assertIsNotNone(offsets["spark"])

    def test_ambiguous_kernel_out_dir_fails(self):
        with self.assertRaises(ValueError):
            self.build(second_dir=True)


if __name__ == "__main__":
    unittest.main()
