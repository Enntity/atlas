# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded CPU-only GLM hidden-trace evidence checker."""
import argparse
import json
import math
from pathlib import Path
import re
import stat
import struct
import sys

MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_LINE_BYTES = 16 * 1024
MAX_MANIFEST_BYTES = 64 * 1024
MARKER = "GLM MTP HIDDEN_TRACE"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
FIELD = re.compile(r"([a-zA-Z_][a-zA-Z0-9_.]*)=(Some\(\[[^\]]*\]\)|[^\s]+)")
INTEGERS = {"rank", "slot", "generation", "attempt", "position", "seed", "hidden_row",
            "step", "input_token", "self.cache_before", "self.cache_after", "draft"}
BOOLS = {"eh_nvfp4", "head_nvfp4"}
HASHES = {"input_sha256", "final_sha256"}
FIELDS = INTEGERS | BOOLS | HASHES | {"argmax_pair_bytes"}
LIMITATIONS = (
    "Equal hidden hashes do not prove equal private KV, equal target causal prefixes, "
    "or a root cause. Matching position/seed only identifies comparison candidates. "
    "Argmax pairs contain both shards; no equality of rank-local maxima is assumed. "
    "Missing rows cannot distinguish a failed proposal from log truncation. "
    "Trace synchronization invalidates throughput comparisons."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, name, lower=0, upper=2**64 - 1):
    require(type(value) is int and lower <= value <= upper, f"invalid {name}")
    return value


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON field {key}")
        result[key] = value
    return result


def parse_line(line):
    """Strict producer text format, including dotted tracing shorthand names."""
    line = ANSI.sub("", line)
    if "HIDDEN_TRACE" not in line:
        return None
    if (MARKER not in line and line.count("HIDDEN_TRACE") == 1
            and re.search(r"\b(?:ATLAS_)?GLM_MTP_HIDDEN_TRACE=[01](?:\s|$)", line)):
        return None  # Explicit valid startup flag receipt, not a trace event.
    require(line.count(MARKER) == 1, "malformed trace marker")
    tail = line.split(MARKER, 1)[1].strip()
    result = {}
    while tail:
        match = FIELD.match(tail)
        require(match is not None, "malformed trace field syntax")
        key, raw = match.groups()
        require(key in FIELDS and key not in result, f"unknown/duplicate trace field {key}")
        if key in INTEGERS:
            require(re.fullmatch(r"0|[1-9][0-9]{0,19}", raw) is not None,
                    f"malformed integer {key}")
            result[key] = integer(int(raw), key)
        elif key in BOOLS:
            require(raw in ("true", "false"), f"invalid boolean {key}")
            result[key] = raw == "true"
        elif key in HASHES:
            require(re.fullmatch("[0-9a-f]{64}", raw) is not None, f"invalid digest {key}")
            result[key] = raw
        elif raw == "None":
            result[key] = None
        else:
            require(raw.startswith("Some([") and raw.endswith("])"), "invalid argmax option")
            values = json.loads(raw[5:-1])
            require(isinstance(values, list) and len(values) == 16, "argmax payload must be16 bytes")
            for value in values:
                integer(value, "argmax byte", 0, 255)
            v0, _, v1, _ = struct.unpack("<fIfI", bytes(values))
            require(math.isfinite(v0) and math.isfinite(v1), "nonfinite argmax pair")
            result[key] = values
        rest = tail[match.end():]
        require(not rest or rest[0].isspace(), "missing field separator")
        tail = rest.lstrip()
    require(set(result) == FIELDS, f"missing fields {sorted(FIELDS - set(result))}")
    for key, lower, upper in (("rank", 0, 1), ("generation", 1, 2**64 - 1),
                              ("attempt", 1, 8), ("step", 0, 3), ("hidden_row", 0, 4),
                              ("position", 1, 2048), ("self.cache_before", 0, 2047),
                              ("self.cache_after", 1, 2048), ("seed", 0, 2**32 - 1),
                              ("input_token", 0, 2**32 - 1), ("draft", 0, 2**32 - 1)):
        integer(result[key], key, lower, upper)
    return result


def validate_chain(records, expected):
    require(len(records) == expected * 4, "missing/extra trace records")
    expected_order = [(attempt, step) for attempt in range(1, expected + 1) for step in range(4)]
    actual_order = [(row["attempt"], row["step"]) for row in records]
    require(len(set(actual_order)) == len(actual_order), "duplicate attempt/step")
    require(actual_order == expected_order, "missing/out-of-order attempt/step")
    for index, row in enumerate(records):
        require(row["self.cache_after"] == row["self.cache_before"] + 1, "broken cursor increment")
        if row["step"] == 0:
            require(row["input_token"] == row["seed"], "first input token differs from seed")
            continue
        previous = records[index - 1]
        for field in ("position", "seed", "hidden_row", "eh_nvfp4"):
            require(row[field] == previous[field], f"within-attempt {field} changed")
        require(row["self.cache_before"] == previous["self.cache_after"], "broken cursor chain")
        require(row["input_token"] == previous["draft"], "broken draft/input token chain")
        require(row["input_sha256"] == previous["final_sha256"], "broken hidden hash chain")


def record_key(label, row):
    return {"request": label, **{key: row[key] for key in
            ("rank", "slot", "generation", "attempt", "position", "seed", "step")}}


def decoded_pairs(row):
    data = row["argmax_pair_bytes"]
    if data is None:
        return None
    v0, i0, v1, i1 = struct.unpack("<fIfI", bytes(data))
    return [{"shard": 0, "value": v0, "local_index": i0},
            {"shard": 1, "value": v1, "local_index": i1}]


def compare(left_label, left, right_label, right):
    metadata = {field: [left[field], right[field]] for field in ("position", "seed", "hidden_row", "step",
                "self.cache_before", "self.cache_after", "eh_nvfp4", "head_nvfp4")
                if left[field] != right[field]}
    if any(left[key] != right[key] for key in ("position", "step", "input_token", "input_sha256")):
        classification = "input_difference"
    elif left["final_sha256"] != right["final_sha256"]:
        classification = "final_hidden_difference"
    elif (left["argmax_pair_bytes"] is not None and right["argmax_pair_bytes"] is not None
          and left["argmax_pair_bytes"] != right["argmax_pair_bytes"]):
        classification = "argmax_pair_difference"
    elif left["draft"] != right["draft"]:
        classification = "draft_difference"
    elif (left["argmax_pair_bytes"] is None) != (right["argmax_pair_bytes"] is None):
        classification = "argmax_pair_availability_difference"
    else:
        classification = "observed_agreement"
    return {"left": record_key(left_label, left), "right": record_key(right_label, right),
            "classification": classification, "metadata_differences": metadata,
            "input_equal": left["input_sha256"] == right["input_sha256"],
            "final_equal": left["final_sha256"] == right["final_sha256"],
            "input_sha256": [left["input_sha256"], right["input_sha256"]],
            "final_sha256": [left["final_sha256"], right["final_sha256"]],
            "drafts": [left["draft"], right["draft"]],
            "argmax_pair_evidence": [decoded_pairs(left), decoded_pairs(right)]}


def compare_requests(left_label, left, right_label, right):
    output = {"requests": [left_label, right_label], "ranks": {}}
    for rank in range(2):
        def index(records):
            result = {}
            for offset in range(0, len(records), 4):
                attempt = records[offset:offset + 4]
                key = (attempt[0]["position"], attempt[0]["seed"])
                require(key not in result, "ambiguous repeated position/seed comparison key")
                result[key] = attempt
            return result
        lindex, rindex = index(left[rank]), index(right[rank])
        common = sorted(lindex.keys() & rindex.keys())
        output["ranks"][str(rank)] = {
            "unmatched_left": [list(key) for key in sorted(lindex.keys() - rindex.keys())],
            "unmatched_right": [list(key) for key in sorted(rindex.keys() - lindex.keys())],
            "steps": [compare(left_label, lrow, right_label, rrow) for key in common
                      for lrow, rrow in zip(lindex[key], rindex[key])]}
    return output


def analyze(manifest, base):
    """Analyze explicitly owned request records from complete rank logs."""
    require(isinstance(manifest, dict) and set(manifest) == {"requests", "comparisons"},
            "manifest requires only requests and comparisons")
    requests, comparisons = manifest["requests"], manifest["comparisons"]
    require(isinstance(requests, list) and 1 <= len(requests) <= 8, "require1..8 requests")
    require(isinstance(comparisons, list) and len(comparisons) <= 28, "too many comparisons")
    owned, request_rows, expected_counts, paths = {}, {}, {}, {}
    for request in requests:
        require(isinstance(request, dict) and set(request) == {"id", "expected_attempts", "ranks"},
                "invalid request manifest fields")
        label = request["id"]
        require(isinstance(label, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", label),
                "invalid request label")
        require(label not in request_rows, "duplicate request label")
        expected_counts[label] = integer(request["expected_attempts"], "expected_attempts", 1, 8)
        require(isinstance(request["ranks"], list) and len(request["ranks"]) == 2, "require two rank selectors")
        request_rows[label] = {}
        for selector in request["ranks"]:
            require(isinstance(selector, dict) and set(selector) == {"rank", "path", "slot", "generation"},
                    "invalid selector fields")
            rank = integer(selector["rank"], "rank", 0, 1)
            slot = integer(selector["slot"], "slot")
            generation = integer(selector["generation"], "generation", 1)
            require(rank not in request_rows[label], "duplicate selector rank")
            path = selector["path"]
            require(isinstance(path, str) and 0 < len(path) <= 4096, "invalid log path")
            resolved = (Path(base) / path).resolve()
            paths.setdefault(resolved, path)
            key = (resolved, rank, slot, generation)
            require(key not in owned, "selector assigned to multiple requests")
            owned[key] = label
            request_rows[label][rank] = []
    files = {}
    for path, display_path in paths.items():
        byte_count, trace_count = 0, 0
        require(stat.S_ISREG(path.stat().st_mode), f"{display_path}: require a regular log file")
        with path.open("rb") as source:
            for number in range(1, MAX_FILE_BYTES + 2):
                raw = source.readline(MAX_LINE_BYTES + 1)
                if not raw:
                    break
                byte_count += len(raw)
                require(len(raw) <= MAX_LINE_BYTES, f"{display_path}:{number}: line exceeds bound")
                require(byte_count <= MAX_FILE_BYTES, f"{display_path}: file exceeds bound")
                try:
                    row = parse_line(raw.decode("utf-8", errors="strict"))
                except (ValueError, UnicodeError) as error:
                    raise ValueError(f"{display_path}:{number}: {error}") from error
                if row is None:
                    continue
                key = (path, row["rank"], row["slot"], row["generation"])
                require(key in owned, f"{display_path}:{number}: undeclared trace owner")
                selected = request_rows[owned[key]][row["rank"]]
                require(len(selected) < 32, "more than32 records for request/rank")
                selected.append(row)
                trace_count += 1
        files[display_path] = {"bytes": byte_count, "trace_records": trace_count}
    result = {"files": files, "requests": {}, "comparisons": [], "limitations": LIMITATIONS}
    for label, ranks in request_rows.items():
        for records in ranks.values():
            validate_chain(records, expected_counts[label])
        result["requests"][label] = {"record_count": sum(map(len, ranks.values())),
            "cross_rank": [compare(label, left, label, right)
                           for left, right in zip(ranks[0], ranks[1])]}
    seen_comparisons = set()
    for pair in comparisons:
        require(isinstance(pair, list) and len(pair) == 2 and all(isinstance(v, str) for v in pair),
                "comparison requires two request labels")
        left, right = pair
        require(left != right and left in request_rows and right in request_rows, "invalid comparison labels")
        require(tuple(sorted(pair)) not in seen_comparisons, "duplicate comparison")
        seen_comparisons.add(tuple(sorted(pair)))
        result["comparisons"].append(compare_requests(left, request_rows[left], right, request_rows[right]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        require(stat.S_ISREG(args.manifest.stat().st_mode), "require a regular manifest file")
        with args.manifest.open("rb") as source:
            raw = source.read(MAX_MANIFEST_BYTES + 1)
        require(len(raw) <= MAX_MANIFEST_BYTES, "manifest exceeds64KiB")
        manifest = json.loads(raw, object_pairs_hook=unique_object)
        result = analyze(manifest, args.manifest.resolve().parent)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, OSError, RecursionError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
