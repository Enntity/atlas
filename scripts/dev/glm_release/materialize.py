# SPDX-License-Identifier: AGPL-3.0-only
"""Offline recipe/launch materialization; never connects to nodes.

No environment or watchdog inference: complete operator inputs are required.
The unchanged canonical generator and production supervisor validate authority.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
            and value != "0" * 64, "explicit full nonzero SHA256 required")
    return value


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON/environment key: " + key)
        result[key] = value
    return result


def supplied(value):
    if isinstance(value, dict):
        for key, item in value.items():
            supplied(key)
            supplied(item)
    elif isinstance(value, list):
        for item in value:
            supplied(item)
    elif isinstance(value, str):
        require("REQUIRED_" not in value and "\0" not in value,
                "unresolved operator input or NUL")


def inputs(recipe, launch, output, workload, workload_bytes):
    recipe, launch = copy.deepcopy(recipe), copy.deepcopy(launch)
    require(output.is_absolute() and workload.is_absolute(), "absolute local paths required")
    require(recipe["owner_capacity"] == 8, "this workload requires explicit capacity eight")
    require([r["rank"] for r in recipe["ranks"]] == [0, 1]
            and [n["rank"] for n in launch["nodes"]] == [0, 1], "ordered two ranks required")
    digest(recipe["server_elf_sha256"])
    digest(recipe["guard_elf_sha256"])
    for rank in recipe["ranks"]:
        digest(rank["image_sha256"])
        for flag, expected in (("--max-batch-size", "8"), ("--max-num-seqs", "8"),
                               ("--max-seq-len", "2044"), ("--max-prefill-tokens", "1024"),
                               ("--ssm-cache-slots", "0"), ("--ssm-checkpoint-interval", "0")):
            actual = [a for a in rank["argv"] if a == flag or a.startswith(flag + "=")]
            require(actual == [flag + "=" + expected], "explicit selected CLI profile mismatch")
        env = unique(rank["environment"])
        require(len(env) <= 128 and env.get("ATLAS_GLM_C2_PAIRED_VERIFY") == "1"
                and env.get("ATLAS_GLM_C2_PAIR_FFN") == "joint-shared-m10"
                and env.get("ATLAS_GLM_OWNER_VERIFY") in ("0", "joint"),
                "explicit paired control/owner mode required within environment bound")
    require(dict(recipe["ranks"][0]["environment"])["ATLAS_GLM_OWNER_VERIFY"]
            == dict(recipe["ranks"][1]["environment"])["ATLAS_GLM_OWNER_VERIFY"],
            "rank owner modes differ")
    recipe["output_directory"] = str(output / "recipes")
    for node in launch["nodes"]:
        digest(node["supervisor_sha256"])
        digest(node["relay_sha256"])
        node["recipe_file"] = str(output / "recipes" / ("rank%d.recipe.bin" % node["rank"]))
        node["recipe_sha256"] = "generated after canonical encoding"
    work = launch["workload"]
    digest(work["program_sha256"])
    require(0 < len(workload_bytes) <= work["limits"]["stdin_bytes"] <= 65536,
            "workload exceeds pinned stdin bound")
    work["input_file"] = str(output / "workload.input")
    work["input_sha256"] = hashlib.sha256(workload_bytes).hexdigest()
    supplied(recipe)
    supplied(launch)
    return recipe, launch


def read(path, bound):
    with path.open("rb") as source:
        data = source.read(bound + 1)
    require(len(data) <= bound, "input exceeds byte bound: " + str(path))
    return data


def save(path, value):
    data = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    require(len(data) <= 200 * 1024, "generated JSON exceeds controller bound")
    save_bytes(path, data)


def save_bytes(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as target:
        os.fchmod(target.fileno(), 0o600)
        target.write(data)
        target.flush()
        os.fsync(target.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("recipe-input", "launch-input", "generator", "workload", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--generator-sha256", required=True)
    args = parser.parse_args()
    for path in (args.recipe_input, args.launch_input, args.generator, args.workload, args.output):
        require(path.is_absolute() and path == path.resolve(),
                "canonical absolute paths without symlinks required")
    require(not args.output.exists(), "output must be a new private directory")
    repository = Path(__file__).resolve().parents[3]
    require(not args.output.is_relative_to(repository), "generated records must remain outside checkout")
    require(hashlib.sha256(read(args.generator, 64 * 1024 * 1024)).hexdigest()
            == digest(args.generator_sha256), "generator ELF hash mismatch")
    workload_bytes = read(args.workload, 65536)
    recipe, launch = inputs(
        json.loads(read(args.recipe_input, 200 * 1024), object_pairs_hook=unique),
        json.loads(read(args.launch_input, 200 * 1024), object_pairs_hook=unique),
        args.output, args.workload, workload_bytes)
    args.output.mkdir(mode=0o700)
    save_bytes(Path(launch["workload"]["input_file"]), workload_bytes)
    save(args.output / "recipe-input.json", recipe)
    subprocess.run([str(args.generator), str(args.output / "recipe-input.json")],
                   check=True, timeout=30, env={})
    for node in launch["nodes"]:
        node["recipe_sha256"] = hashlib.sha256(read(Path(node["recipe_file"]), 65536)).hexdigest()
    save(args.output / "launch.json", launch)
    for directory in (args.output, args.output.parent):
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    print(args.output / "launch.json")


if __name__ == "__main__":
    main()
