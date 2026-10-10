# SPDX-License-Identifier: AGPL-3.0-only
"""Attest a running container's cuBLAS mappings from /proc/<pid>/maps.

Given a container PID or docker container name, reads /proc/<pid>/maps and
verifies that the only mapped libcublas/libcublasLt objects are the pinned
nvidia-cublas 13.1.1.3 pair installed at /opt/atlas/cublas-13.1.1.3 (classic
libcublas.so.13 + libcublasLt.so.13), that nothing else named libcublas* is
mapped (notably the CUDA-toolkit copy), and that both mapped files hash to the
committed pins in docker/gb10/glm-5.3-flash-exl3/exl3/cublas131.sha256.
Fails closed on any deviation; "(deleted)" mappings always fail.

Usage (host side; needs read access to /proc/<pid>/maps and /proc/<pid>/root):

    python3 tests/exl3_validation/attest_runtime_maps.py <pid|container-name>
    python3 tests/exl3_validation/attest_runtime_maps.py <pid> --maps captured.maps \
        --fs-root /path/to/container/root   # offline / captured evidence

The report is JSON on stdout (or --out); exit status is 0 only if the
verdict is clean.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath

DEFAULT_PINS = (Path(__file__).resolve().parents[2]
                / "docker/gb10/glm-5.3-flash-exl3/exl3/cublas131.sha256")
EXPECTED_CUBLAS = ("libcublas.so.13", "libcublasLt.so.13")
DELETED_SUFFIX = " (deleted)"


def parse_maps(text):
    """Parse /proc/<pid>/maps text into entries with int addresses and paths."""
    entries = []
    for line in text.splitlines():
        line = line.rstrip()
        if not line:
            continue
        parts = line.split(None, 5)
        if len(parts) < 5:
            raise ValueError(f"unparsable maps line: {line!r}")
        start, end = parts[0].split("-")
        entries.append({
            "start": int(start, 16),
            "end": int(end, 16),
            "perms": parts[1],
            "offset": parts[2],
            "dev": parts[3],
            "inode": parts[4],
            "path": parts[5].strip() if len(parts) > 5 else None,
        })
    return entries


def read_pins(path):
    """Parse a sha256sum-format pins file into {basename: {"sha256", "path"}}."""
    pins = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        digest, name = line.split(None, 1)
        name = name.strip()
        base = PurePosixPath(name).name
        if base in pins:
            raise ValueError(f"duplicate pin for {base!r} in {path}")
        pins[base] = {"sha256": digest, "path": name}
    cublas = {b for b in pins if b.startswith("libcublas")}
    if cublas != set(EXPECTED_CUBLAS):
        raise ValueError(f"pins must cover exactly {EXPECTED_CUBLAS}, got {sorted(cublas)}")
    return pins


def _clean_name(raw_path):
    path = raw_path[:-len(DELETED_SUFFIX)] if raw_path.endswith(DELETED_SUFFIX) else raw_path
    return PurePosixPath(path).name, path


def verify(mapped_paths, pins, hash_file):
    """Verify mapped cuBLAS paths against pins. hash_file(path) -> sha256 hex."""
    expected = {b: p for b, p in pins.items() if b.startswith("libcublas")}
    if set(expected) != set(EXPECTED_CUBLAS):
        raise ValueError(f"pins must cover exactly {EXPECTED_CUBLAS}, got {sorted(expected)}")
    issues = []
    seen = {}
    for raw in mapped_paths:
        base, path = _clean_name(raw)
        if not base.startswith("libcublas"):
            continue
        if raw.endswith(DELETED_SUFFIX):
            issues.append(f"deleted mapping: {raw}")
            continue
        seen.setdefault(base, set()).add(path)
    libraries = []
    for name in sorted(set(seen) | set(expected)):
        paths = seen.get(name, set())
        if name not in expected:
            issues.append(f"unexpected cuBLAS mapping: {sorted(paths)}")
            continue
        if not paths:
            issues.append(f"pinned {name} is not mapped")
            continue
        if paths != {expected[name]["path"]}:
            issues.append(f"{name} mapped from {sorted(paths)}, "
                          f"expected {expected[name]['path']}")
            continue
        path = expected[name]["path"]
        try:
            digest = hash_file(path)
        except OSError as e:
            issues.append(f"cannot hash {path}: {e}")
            continue
        ok = digest == expected[name]["sha256"]
        if not ok:
            issues.append(f"{name} sha256 {digest} != pinned {expected[name]['sha256']}")
        libraries.append({"name": name, "path": path, "sha256": digest,
                          "pinned_sha256": expected[name]["sha256"], "ok": ok})
    return {"libraries": libraries, "issues": issues, "ok": not issues}


def resolve_pid(target, run=subprocess.run):
    """Return (pid, container_or_None) for a numeric PID or docker container name."""
    if target.isdigit():
        pid = int(target)
        if pid <= 0:
            raise ValueError(f"invalid PID {target!r}")
        return pid, None
    proc = run(["docker", "inspect", "--format", "{{.State.Pid}}", target],
               capture_output=True, text=True)
    if proc.returncode != 0:
        raise ValueError(f"docker inspect failed for {target!r}: {proc.stderr.strip()}")
    pid = int(proc.stdout.strip())
    if pid <= 0:
        raise ValueError(f"container {target!r} is not running")
    return pid, target


def _file_hash(fs_root):
    root = Path(fs_root) if fs_root else None

    def hash_file(path):
        target = (root / path.lstrip("/")) if root is not None else Path(path)
        return hashlib.sha256(target.read_bytes()).hexdigest()

    return hash_file


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", help="container PID or docker container name")
    parser.add_argument("--pins", default=str(DEFAULT_PINS))
    parser.add_argument("--maps", help="read maps from this file instead of /proc/<pid>/maps")
    parser.add_argument("--fs-root", default=None,
                        help="filesystem root for hashing mapped paths "
                             "(default: /proc/<pid>/root in live mode)")
    parser.add_argument("--out", help="write the JSON report here as well as stdout")
    args = parser.parse_args(argv)
    pid, container = resolve_pid(args.target)
    if args.maps:
        if not args.fs_root:
            raise ValueError("--maps requires --fs-root: the captured maps must be "
                             "hashed against the filesystem they describe, never "
                             "the host's")
        maps_text = Path(args.maps).read_text()
        maps_source = args.maps
        fs_root = args.fs_root
    else:
        maps_path = Path(f"/proc/{pid}/maps")
        maps_text = maps_path.read_text()
        maps_source = str(maps_path)
        fs_root = args.fs_root or f"/proc/{pid}/root"
    pins = read_pins(args.pins)
    result = verify([e["path"] for e in parse_maps(maps_text) if e["path"]],
                    pins, _file_hash(fs_root))
    report = {"target": args.target, "pid": pid, "container": container,
              "maps_source": maps_source, "fs_root": fs_root,
              "pins": args.pins, **result}
    text = json.dumps(report, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(text)
    print(text, end="")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
