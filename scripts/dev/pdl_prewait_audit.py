#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""SASS audit of the GLM-5.3-Flash PDL kernels: nothing but weight touches before the wait.

A kernel on PDL_KERNELS (crates/spark-runtime/src/cuda_backend/gpu_impl/pdl.rs)
starts while its predecessor still runs, so nothing it reads before
`griddepcontrol.wait` (SASS ACQBULK) may be a predecessor's output. The
source-contract test checks the source; this checks what ptxas schedules.
ptxas treats `ld.global.nc` (what NVVM emits for a `const __restrict__` read)
as invariant and may hoist it above ACQBULK: the persistent MoE gate/up twin
read its routes' item count that way, a layer stale (fixed with __ldcg in
glm_moe_decode_stream.cuh). The only global accesses allowed before the wait
are atlas_pdl_touch.cuh's discarded byte loads of weights
(LDG.E.U8.STRONG.SYS).

    python3 scripts/dev/pdl_prewait_audit.py <atlas-kernels OUT_DIR with t0__*.ptx> [arch]

Needs ptxas and cuobjdump on PATH; exits 1 and lists every violation.
"""
import concurrent.futures as cf
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PDL_RS = os.path.join(ROOT, "crates/spark-runtime/src/cuda_backend/gpu_impl/pdl.rs")
MEM = re.compile(r"\b(LDG|LD|STG|ST|ATOMG|ATOM|RED|LDGSTS)\b")
TOUCH = "U8.STRONG.SYS"


def listed():
    src = open(PDL_RS).read()
    body = src.split("pub(super) const PDL_KERNELS")[1].split("];")[0]
    return set(re.findall(r'"([A-Za-z0-9_]+)"', body))


def sass(ptx, arch, tmp):
    cubin = os.path.join(tmp, os.path.basename(ptx) + ".cubin")
    subprocess.run(["ptxas", f"-arch={arch}", "-O3", ptx, "-o", cubin], check=True)
    return subprocess.run(["cuobjdump", "-sass", cubin], check=True, capture_output=True,
                          text=True).stdout


def prewait(listing, names):
    """{kernel: (has_wait, [global accesses before its first ACQBULK, touches excepted])}."""
    out, cur, seen, pre = {}, None, False, []
    for line in listing.splitlines() + ["\tFunction : <end>"]:
        m = re.match(r"\s+Function : (\S+)", line)
        if m:
            if cur in names:
                out[cur] = (seen, pre)
            cur, seen, pre = m.group(1), False, []
            continue
        if cur is None or seen:
            continue
        if "ACQBULK" in line:
            seen = True
            continue
        ins = re.search(r"\*/\s+(.*?);", line)
        if ins and MEM.search(ins.group(1)) and TOUCH not in ins.group(1):
            pre.append(ins.group(1).strip())
    return out


def main():
    ptx_dir = sys.argv[1]
    arch = sys.argv[2] if len(sys.argv) > 2 else "sm_121a"
    names = listed()
    files = []
    for f in sorted(os.listdir(ptx_dir)):
        if f.endswith(".ptx"):
            text = open(os.path.join(ptx_dir, f)).read()
            if any(f".entry {n}(" in text for n in names):
                files.append(os.path.join(ptx_dir, f))
    found, bad = {}, []
    with tempfile.TemporaryDirectory() as tmp, cf.ThreadPoolExecutor(os.cpu_count()) as pool:
        for res in pool.map(lambda p: prewait(sass(p, arch, tmp), names), files):
            found.update(res)
    for name in sorted(names):
        if name not in found:
            bad.append(f"{name}: listed, but no PTX module defines it")
            continue
        has_wait, pre = found[name]
        if not has_wait:
            bad.append(f"{name}: no ACQBULK (griddepcontrol.wait) in its SASS")
        for ins in pre:
            bad.append(f"{name}: before the wait: {ins}")
    print(f"{len(found)} PDL-listed kernels in {len(files)} modules ({arch}); {len(bad)} violation(s)")
    for b in bad:
        print("  " + b)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
