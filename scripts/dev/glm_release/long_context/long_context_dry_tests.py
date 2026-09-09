# SPDX-License-Identifier: AGPL-3.0-only
"""Pure mocked safety boundaries. NO node, Docker, HTTP or lifecycle qualification."""
import json
import threading
import time
from unittest.mock import patch

import long_context_node as node
from long_context_profile import ENV, argv, create_args, environment


def rejects(operation):
    try:
        operation()
    except (RuntimeError, ValueError):
        return
    raise AssertionError("unsafe case accepted")


def run_safety_tests(runner_class):
    explicit = {"rank": 0, "image_sha256": "a" * 64,
                "weights_host_path": "/fixture/weights",
                "fabric": {"interface": "fixture0", "hca": "fixture_hca", "gid_index": "7"}}
    created = create_args(explicit, "b" * 32, 4096, "192.0.2.10", 8890, False)
    assert "NCCL_SOCKET_IFNAME=fixture0" in created
    assert "NCCL_IB_HCA=fixture_hca" in created
    assert "NCCL_IB_GID_INDEX=7" in created
    off, on = environment(explicit, False), environment(explicit, True)
    assert off["ATLAS_GLM_PAGED_PREFILL_BF16_GEMM"] == "0"
    assert on["ATLAS_GLM_PAGED_PREFILL_BF16_GEMM"] == "1"
    assert [k for k in off if off[k] != on[k]] == ["ATLAS_GLM_PAGED_PREFILL_BF16_GEMM"]
    for invalid in (None, 0, 1, "0", "1"):
        rejects(lambda: environment(explicit, invalid))
    assert ENV["ATLAS_NO_DECODE_GRAPHS"] == "1"
    for context in (4096, 8192, 16384):
        for rank in (0, 1):
            profile = argv(context, rank, "192.0.2.10", 8890)
            assert "--ssm-cache-slots=0" in profile
            assert "--ssm-checkpoint-interval=0" in profile
            assert not any(v == "--enable-prefix-caching"
                           or v.startswith("--enable-prefix-caching=") for v in profile)
    r = object.__new__(runner_class)
    r.failed = threading.Event()
    r.end = time.monotonic() + 30
    r.last = [{"controller_begin": time.monotonic()} for _ in range(2)]
    r.healthy()
    r.last[1]["controller_begin"] -= 11
    rejects(r.healthy)
    r.last[1] = None
    rejects(r.healthy)
    r.last = [{"controller_begin": time.monotonic()} for _ in range(2)]
    r.failed.set()
    rejects(r.healthy)
    for available, swapfree, accepted in ((4194304, 1024, True), (4194303, 1024, False), (5000000, 1023, False)):
        text = f"MemAvailable: {available} kB\nSwapTotal: 1024 kB\nSwapFree: {swapfree} kB\n"
        with patch.object(node.Path, "read_text", return_value=text):
            if accepted:
                assert node.memory()["available_kib"] == available
            else:
                rejects(node.memory)
    meta = {"name": "atlas-longctx-fixture-r0", "session": "a" * 32,
            "rank": 0, "image_sha256": "b" * 64}
    cid = "c" * 64
    data = {"Id": cid, "Name": "/" + meta["name"], "Image": "sha256:" + meta["image_sha256"],
            "Config": {"Labels": {"atlas.longctx": meta["session"]}}}
    # Exercise actual identity predicate, replacing ONLY external reads.
    with patch.object(node, "command", side_effect=[cid + "\n", json.dumps([data])]), \
            patch.object(node.Path, "exists", return_value=True), patch.object(node, "read", return_value=cid):
        assert node.inspect(meta)["Id"] == cid
    with patch.object(node, "command", side_effect=[cid + "\n", json.dumps([data])]), \
            patch.object(node.Path, "exists", return_value=True), patch.object(node, "read", return_value="d" * 64):
        rejects(lambda: node.inspect(meta))
    print("PASS: mocked memory equality/swap, stale/absent pair samples, terminal latch, immutable-ID replacement")
