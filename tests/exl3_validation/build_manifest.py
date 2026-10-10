# SPDX-License-Identifier: AGPL-3.0-only
"""Build manifest for the GLM-5.3 EXL3 validation image (runs inside the builder).

Records the bound git commit and source snapshot hash, the toolchain versions,
the atlas-kernels target closure (arch/compiler and per-module compiler flags),
the SHA256 of every shipped binary, and the SHA256 of every compiled PTX module
after proving each PTX blob is byte-embedded in both binaries (the same
evidence layout the external attest-native-aux-classic131.py check uses).

Usage (see the docker/gb10/glm-5.3-flash-exl3/exl3/Dockerfile header):

    python3 build_manifest.py --release-dir /build/target/release \
        --fingerprint-dir /build/fingerprint \
        --target-key gb10/glm-5.3-flash-exl3/exl3 \
        --git-sha <sha> --source-snapshot-sha256 <sha> \
        --out /build/fingerprint/build-manifest.json

All arguments are explicit; there are no defaults. The script fails closed on
an ambiguous atlas-kernels build directory, a missing target closure, a PTX
module name collision, or a PTX blob that is not embedded in a binary.
"""
import argparse
import hashlib
import json
from pathlib import Path

SERVER_BINARY = "spark"
BINARIES = {
    "spark": "spark",
    "exl3_moe_scratch_check": "examples/exl3_moe_scratch_check",
}
CLOSURES_PREFIX = "cargo:rustc-env=ATLAS_TARGET_CLOSURES="
TOOLCHAIN_FILES = ("rustc.txt", "cargo.txt", "nvcc.txt")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def parse_target_closures(output_text, target_key):
    """Extract the one target closure for `target_key` from build-script output."""
    lines = [l[len(CLOSURES_PREFIX):] for l in output_text.splitlines()
             if l.startswith(CLOSURES_PREFIX)]
    if len(lines) != 1:
        raise ValueError(f"expected exactly one {CLOSURES_PREFIX} line, got {len(lines)}")
    closures = json.loads(lines[0])
    if target_key not in closures:
        raise ValueError(f"target closure {target_key!r} missing; found {sorted(closures)}")
    return closures[target_key]


def find_kernel_out_dir(release_dir, target_key):
    """Locate the single atlas-kernels OUT_DIR whose closure covers `target_key`."""
    matches = []
    for ptx_rs in sorted(release_dir.glob("build/atlas-kernels-*/out/target_ptx.rs")):
        out_dir = ptx_rs.parent
        output = out_dir.parent / "output"
        if output.is_file() and f'"{target_key}"' in output.read_text():
            matches.append(out_dir)
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one atlas-kernels out dir for {target_key}, got {matches}")
    return matches[0]


def ptx_modules(out_dir):
    """Map module name -> t<idx>__<module>.ptx path; collision-free and non-empty."""
    modules = {}
    for ptx in sorted(out_dir.glob("t*__*.ptx")):
        module = ptx.name.split("__", 1)[1][: -len(".ptx")]
        if not module:
            raise ValueError(f"unparsable PTX filename {ptx.name!r} in {out_dir}")
        if module in modules:
            raise ValueError(f"duplicate PTX module {module!r} in {out_dir}")
        modules[module] = ptx
    if not modules:
        raise ValueError(f"no t*__*.ptx modules in {out_dir}")
    return modules


def load_toolchain(fingerprint_dir):
    missing = [n for n in TOOLCHAIN_FILES if not (fingerprint_dir / n).is_file()]
    if missing:
        raise ValueError(f"missing toolchain fingerprints {missing} in {fingerprint_dir}")
    return {n[: -len(".txt")]: (fingerprint_dir / n).read_text().strip()
            for n in TOOLCHAIN_FILES}


def build_manifest(release_dir, fingerprint_dir, target_key, git_sha, source_snapshot_sha256):
    release_dir = Path(release_dir)
    fingerprint_dir = Path(fingerprint_dir)
    out_dir = find_kernel_out_dir(release_dir, target_key)
    binaries = {}
    binary_bytes = {}
    for name, rel in BINARIES.items():
        path = release_dir / rel
        data = path.read_bytes()
        binary_bytes[name] = data
        binaries[name] = {"path": str(path), "sha256": sha256_bytes(data), "bytes": len(data)}
    embedded = {}
    for module, ptx in sorted(ptx_modules(out_dir).items()):
        blob = ptx.read_bytes()
        offsets = {}
        for name, data in binary_bytes.items():
            offset = data.find(blob)
            if offset < 0 and name == SERVER_BINARY:
                raise ValueError(f"PTX module {module!r} not embedded in binary {name!r}")
            # Other binaries (the scratch-check example) may legitimately embed
            # only the modules they reference; record absence as None.
            offsets[name] = offset if offset >= 0 else None
        embedded[module] = {
            "file": ptx.name,
            "sha256": sha256_bytes(blob),
            "bytes": len(blob),
            "embedded_byte_offsets": offsets,
        }
    return {
        "schema": "atlas-exl3-build-manifest-v1",
        "git_commit": git_sha,
        "source_snapshot_sha256": source_snapshot_sha256,
        "target_key": target_key,
        "toolchain": load_toolchain(fingerprint_dir),
        "compiler_flags": parse_target_closures(
            (out_dir.parent / "output").read_text(), target_key),
        "binaries": binaries,
        "embedded_ptx": embedded,
    }


def write_binary_fingerprints(fingerprint_dir, manifest):
    """Write sha256sum-format <name>.sha256 files from the manifest (single hash source).

    Paths are builder paths (record-only evidence); they are not sha256sum -c
    targets in the runtime image, where the binaries live under /usr/local/bin.
    """
    for name, entry in manifest["binaries"].items():
        (Path(fingerprint_dir) / f"{name}.sha256").write_text(
            f"{entry['sha256']}  {entry['path']}\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--fingerprint-dir", required=True)
    parser.add_argument("--target-key", required=True)
    parser.add_argument("--git-sha", required=True)
    parser.add_argument("--source-snapshot-sha256", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    manifest = build_manifest(
        args.release_dir, args.fingerprint_dir, args.target_key,
        args.git_sha, args.source_snapshot_sha256)
    write_binary_fingerprints(args.fingerprint_dir, manifest)
    Path(args.out).write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"out": args.out,
                      "binaries": {n: e["sha256"] for n, e in manifest["binaries"].items()},
                      "embedded_ptx_modules": sorted(manifest["embedded_ptx"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
