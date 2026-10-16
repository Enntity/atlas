#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
Vendor a header-only extract of nvidia/Qwen3.8-Flash-Next-FP4-Local for the
`qwen4_exp` probe/format tests (`weight_loader/qwen4_exp/format_tests.rs`).

Reads every shard's safetensors HEADER over HTTP range requests (no tensor
data), keeps every tensor EXCEPT routed experts other than 0 and 256 (the
first local expert of EP ranks 0 and 1 at EP=2) in the main layers and the
MTP block, and writes `{name: [dtype, shape]}` to
test_data/qwen4_exp_fp4local_headers.json. The tests rebuild a WeightStore
from it exactly as the loaders would (CSF pairs decoded, n-gram shards
deferred), so the probe and format detection run against the real namespace.

USAGE
    python3 scripts/qwen4_exp_fp4local_header_extract.py [--headers headers.json]

`--headers` reuses an already-fetched `{shard: header}` JSON.
"""

import argparse
import concurrent.futures as cf
import json
import re
import struct
import urllib.request
from pathlib import Path

REPO = "nvidia/Qwen3.8-Flash-Next-FP4-Local"
REV = "b9ce18a6887e18c089fdea8509a33e63429a3300"
OUT = Path(__file__).resolve().parent.parent / "test_data" / "qwen4_exp_fp4local_headers.json"
KEEP_EXPERTS = {0, 256}


def fetch_headers():
    url = f"https://huggingface.co/{REPO}/resolve/{REV}/model.safetensors.index.json"
    shards = sorted(set(json.loads(urllib.request.urlopen(url).read())["weight_map"].values()))

    def get(shard):
        u = f"https://huggingface.co/{REPO}/resolve/{REV}/{shard}"
        r = urllib.request.Request(u, headers={"Range": "bytes=0-7"})
        n = struct.unpack("<Q", urllib.request.urlopen(r).read())[0]
        r = urllib.request.Request(u, headers={"Range": f"bytes=8-{7 + n}"})
        return shard, json.loads(urllib.request.urlopen(r).read())

    with cf.ThreadPoolExecutor(16) as ex:
        return dict(ex.map(get, shards))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headers")
    args = ap.parse_args()
    headers = json.load(open(args.headers)) if args.headers else fetch_headers()
    out = {}
    for header in headers.values():
        for name, info in header.items():
            if name == "__metadata__":
                continue
            m = re.search(r"\.mlp\.experts\.(\d+)\.", name)
            if m and int(m.group(1)) not in KEEP_EXPERTS:
                continue
            out[name] = [info["dtype"], info["shape"]]
    OUT.write_text(json.dumps(dict(sorted(out.items())), separators=(",", ":")) + "\n")
    print(f"{OUT}: {len(out)} tensors")


if __name__ == "__main__":
    main()
