# SPDX-License-Identifier: AGPL-3.0-only
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from attest_runtime_maps import (
    DEFAULT_PINS,
    EXPECTED_CUBLAS,
    main,
    parse_maps,
    read_pins,
    resolve_pid,
    verify,
)

INSTALL = "/opt/atlas/cublas-13.1.1.3"
CLASSIC = f"{INSTALL}/libcublas.so.13"
LT = f"{INSTALL}/libcublasLt.so.13"
TOOLKIT_LT = "/usr/local/cuda/lib64/libcublasLt.so.13"
CLASSIC_BYTES = b"classic-blob"
LT_BYTES = b"lt-blob" * 10


def sha(data):
    return hashlib.sha256(data).hexdigest()


def make_pins(path, classic_sha=None, lt_sha=None):
    path.write_text(
        f"{classic_sha or sha(CLASSIC_BYTES)}  {CLASSIC}\n"
        f"{lt_sha or sha(LT_BYTES)}  {LT}\n"
        f"{'a' * 64}  {INSTALL}/package-binding-v1.json\n"
        f"{'b' * 64}  {INSTALL}/dist-info/licenses/License.txt\n")


class ParseMapsTests(unittest.TestCase):
    def test_parses_entries(self):
        text = (
            "00400000-00401000 r-xp 00000000 00:37573 123 /usr/local/bin/spark\n"
            f"7f00-7f10 r--p 00000000 00:1 456 {LT}\n"
            "7f10-7f20 rw-p 00001000 00:1 456 \n"
            f"7f30-7f40 r-xp 00000000 00:1 789 {TOOLKIT_LT}{ ' (deleted)'}\n"
        )
        entries = parse_maps(text)
        self.assertEqual(len(entries), 4)
        self.assertEqual(entries[0]["start"], 0x400000)
        self.assertEqual(entries[0]["path"], "/usr/local/bin/spark")
        self.assertEqual(entries[1]["path"], LT)
        self.assertIsNone(entries[2]["path"])
        self.assertTrue(entries[3]["path"].endswith("(deleted)"))

    def test_unparsable_line_fails(self):
        with self.assertRaises(ValueError):
            parse_maps("garbage line without fields\n")


class ReadPinsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.pins_path = Path(self.tmp.name) / "cublas131.sha256"

    def test_reads_expected_pair(self):
        make_pins(self.pins_path)
        pins = read_pins(self.pins_path)
        self.assertEqual(set(pins), set(EXPECTED_CUBLAS) | {
            "package-binding-v1.json", "License.txt"})
        self.assertEqual(pins["libcublas.so.13"]["path"], CLASSIC)

    def test_duplicate_or_wrong_cublas_set_fails(self):
        self.pins_path.write_text(f"{'c' * 64}  {CLASSIC}\n{'d' * 64}  /elsewhere/libcublas.so.13\n")
        with self.assertRaises(ValueError):
            read_pins(self.pins_path)
        self.pins_path.write_text(f"{'c' * 64}  {CLASSIC}\n")
        with self.assertRaises(ValueError):
            read_pins(self.pins_path)


class VerifyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.classic = self.root / "libcublas.so.13"
        self.lt = self.root / "libcublasLt.so.13"
        self.classic.write_bytes(CLASSIC_BYTES)
        self.lt.write_bytes(LT_BYTES)
        self.pins = {
            "libcublas.so.13": {"sha256": sha(CLASSIC_BYTES), "path": str(self.classic)},
            "libcublasLt.so.13": {"sha256": sha(LT_BYTES), "path": str(self.lt)},
        }

    @staticmethod
    def hash_file(path):
        return sha(Path(path).read_bytes())

    def test_clean_pair_verifies(self):
        report = verify([str(self.classic), str(self.lt), str(self.lt)], self.pins, self.hash_file)
        self.assertTrue(report["ok"], report["issues"])
        self.assertEqual({e["name"] for e in report["libraries"]}, set(EXPECTED_CUBLAS))
        self.assertTrue(all(e["ok"] for e in report["libraries"]))

    def test_unexpected_mapping_fails(self):
        report = verify([str(self.classic), str(self.lt), TOOLKIT_LT], self.pins, self.hash_file)
        self.assertFalse(report["ok"])
        self.assertTrue(any(TOOLKIT_LT in i for i in report["issues"]))

    def test_missing_member_fails(self):
        report = verify([str(self.lt)], self.pins, self.hash_file)
        self.assertFalse(report["ok"])
        self.assertTrue(any("not mapped" in i for i in report["issues"]))

    def test_wrong_directory_fails(self):
        report = verify([str(self.classic), f"/other/libcublasLt.so.13"], self.pins, self.hash_file)
        self.assertFalse(report["ok"])
        self.assertTrue(any("expected" in i for i in report["issues"]))

    def test_deleted_mapping_fails(self):
        report = verify([str(self.classic), f"{self.lt} (deleted)"], self.pins, self.hash_file)
        self.assertFalse(report["ok"])
        self.assertTrue(any("deleted" in i for i in report["issues"]))

    def test_hash_mismatch_fails(self):
        self.classic.write_bytes(b"tampered")
        report = verify([str(self.classic), str(self.lt)], self.pins, self.hash_file)
        self.assertFalse(report["ok"])
        self.assertTrue(any("!= pinned" in i for i in report["issues"]))


class ResolvePidTests(unittest.TestCase):
    def test_numeric_pid(self):
        self.assertEqual(resolve_pid("1234"), (1234, None))
        with self.assertRaises(ValueError):
            resolve_pid("0")

    def test_container_name_via_docker(self):
        class Result:
            returncode = 0
            stdout = "4321\n"
            stderr = ""

        self.assertEqual(resolve_pid("glm53-exl3-atlas-0", run=lambda *a, **k: Result()),
                         (4321, "glm53-exl3-atlas-0"))


class MainTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.pins = self.root / "cublas131.sha256"
        make_pins(self.pins)
        self.fs = self.root / "fs"
        (self.fs / INSTALL.lstrip("/")).mkdir(parents=True)
        (self.fs / CLASSIC.lstrip("/")).write_bytes(CLASSIC_BYTES)
        (self.fs / LT.lstrip("/")).write_bytes(LT_BYTES)
        (self.fs / INSTALL.lstrip("/") / "package-binding-v1.json").write_bytes(b"{}")
        license_dir = self.fs / INSTALL.lstrip("/") / "dist-info/licenses"
        license_dir.mkdir(parents=True)
        (license_dir / "License.txt").write_text("license")
        self.maps = self.root / "maps"
        self.maps.write_text(
            "00400000-00401000 r-xp 00000000 00:37573 1 /usr/local/bin/spark\n"
            f"7f00-7f10 r-xp 00000000 00:1 2 {CLASSIC}\n"
            f"7f20-7f30 r-xp 00000000 00:1 3 {LT}\n"
            f"7f40-7f50 r-xp 00000000 00:1 4 {LT}\n")

    def run_main(self, extra=()):
        out = self.root / "report.json"
        argv = ["123", "--pins", str(self.pins), "--maps", str(self.maps),
                "--fs-root", str(self.fs), "--out", str(out), *extra]
        rc = main(argv)
        return rc, json.loads(out.read_text())

    def test_clean_maps_pass(self):
        rc, report = self.run_main()
        self.assertEqual(rc, 0)
        self.assertTrue(report["ok"])
        self.assertEqual(report["pid"], 123)
        self.assertEqual(report["fs_root"], str(self.fs))

    def test_offline_maps_without_fs_root_fails_closed(self):
        with self.assertRaises(ValueError):
            main(["123", "--pins", str(self.pins), "--maps", str(self.maps)])

    def test_default_pins_cover_committed_pair(self):
        pins = read_pins(DEFAULT_PINS)
        self.assertEqual({b for b in pins if b.startswith("libcublas")},
                         set(EXPECTED_CUBLAS))

    def test_foreign_mapping_fails(self):
        self.maps.write_text(self.maps.read_text()
                             + f"7f60-7f70 r-xp 00000000 00:1 5 {TOOLKIT_LT}\n")
        rc, report = self.run_main()
        self.assertEqual(rc, 1)
        self.assertFalse(report["ok"])


if __name__ == "__main__":
    unittest.main()
