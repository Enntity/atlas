# SPDX-License-Identifier: AGPL-3.0-only
"""Strict compiled-PTX presence/signature gate for the staged GLM B-tile family."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import stat
import sys

MAX_PTX_BYTES = 128 * 1024 * 1024
MODULES = ("grouped", "decode", "repack")


def signatures():
    dense = ["u64"] * 8 + ["u32"] * 3
    fused = ["u64"] * 12 + ["u32"] * 3 + ["u64", "u64", "u32"]
    decode = ["u64"] * 12 + ["f32", "u64", "u64", "u64", "f32", "u64"] + ["u32"] * 3
    grouped = {"glm_moe_gate_up_btile" + suffix: fused for suffix in
               ("", "_vecscale", "_m64", "_m64_vecscale")}
    for suffix in ("dense", "vecscale_dense", "compact", "vecscale_compact"):
        grouped["glm_moe_btile_m64_" + suffix] = dense + (
            ["u64", "u64", "u32"] if suffix.endswith("compact") else [])
    return {
        "grouped": grouped,
        "decode": {f"glm_btile_decode_{variant}{rows}": decode
                   for variant in ("word", "vec") for rows in (1, 2, 3)},
        "repack": {"glm_native_to_btile_u8": ["u64", "u64", "u32", "u32"]},
    }


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_ptx(texts):
    """Require every staged export exactly once with the exact compiled ABI."""
    require(set(texts) == set(MODULES), "require grouped, decode and repack PTX")
    result = {}
    for module, expected in signatures().items():
        source = texts[module]
        require(isinstance(source, str) and 0 < len(source) <= MAX_PTX_BYTES,
                f"{module}: empty or oversized PTX")
        source = re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.S)
        require(len(re.findall(r"(?m)^\s*\.version\s+\d+\.\d+\s*$", source)) == 1,
                f"{module}: missing/duplicate PTX version")
        targets = [value.strip() for value in re.findall(r"(?m)^\s*\.target\s+([^\n]+)$", source)]
        require(targets == ["sm_121a"],
                f"{module}: require exact sm_121a target")
        require(re.findall(r"(?m)^\s*\.address_size\s+(\d+)\s*$", source) == ["64"],
                f"{module}: require 64-bit PTX addresses")
        entries = {}
        for match in re.finditer(r"\.entry\s+([A-Za-z_$][\w$]*)\s*\(([^)]*)\)", source):
            name, parameters = match.groups()
            require(name not in entries, f"{module}: duplicate entry {name}")
            if name in expected:
                require(re.search(r"\.visible\s+$", source[max(0, match.start() - 80):match.start()])
                        is not None, f"{module}: entry {name} is not visible")
                require(re.match(r"\s*(?:\.(?:maxntid|reqntid|minnctapersm|maxnctapersm|maxnreg)"
                                 r"\s+[0-9,\s]+)*\{", source[match.end():]) is not None,
                        f"{module}: entry {name} has no definition")
            entries[name] = parameters
        result[module] = {}
        for name, wanted in expected.items():
            require(name in entries, f"{module}: missing entry {name}")
            actual, parameter_names = [], set()
            for declaration in entries[name].split(","):
                parameter = re.fullmatch(r"\s*\.param\s+\.(u64|u32|f32)\s+([A-Za-z_$][\w$]*)\s*",
                                         declaration)
                require(parameter is not None, f"{module}: unsupported parameter in {name}")
                require(parameter.group(2) not in parameter_names,
                        f"{module}: duplicate parameter in {name}")
                parameter_names.add(parameter.group(2))
                actual.append(parameter.group(1))
            require(actual == wanted, f"{module}: ABI mismatch {name}: {actual}, expected {wanted}")
            result[module][name] = actual
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in MODULES:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        sources, evidence, paths = {}, {}, set()
        for module in MODULES:
            path = getattr(args, module).resolve()
            require(path not in paths, "each module requires its distinct PTX artifact")
            paths.add(path)
            require(stat.S_ISREG(path.stat().st_mode), f"{module}: require regular PTX file")
            with path.open("rb") as source:
                data = source.read(MAX_PTX_BYTES + 1)
            require(0 < len(data) <= MAX_PTX_BYTES, f"{module}: empty or oversized PTX")
            sources[module] = data.decode("ascii", errors="strict")
            evidence[module] = {"path": str(path), "bytes": len(data),
                                "sha256": hashlib.sha256(data).hexdigest()}
        result = check_ptx(sources)
        print(json.dumps({"artifacts": evidence, "signatures": result,
                          "scope": "PTX presence/ABI only; no numerical GPU validation"}, indent=2))
    except (OSError, ValueError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
