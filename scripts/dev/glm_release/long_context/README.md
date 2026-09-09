# Ordinary eager sparse long-context qualification

This is the separate TP2/EP2-v2 **nonspeculative C1–C4** lane at4096,8192 or
16384 configured context. It is not C8 independent decode, paired MTP, or the
LIVE paired guard protocol. The runner's operational watchdog is not a T3
quiescence/receipt-release proof. Packaging these scripts establishes no
fresh native quality, throughput, fit or teardown PASS.

## Provenance and boundaries

These are portable counterparts of the retained release-context campaign's
`long-context-runner.py`, `long_context_profile.py`, `long_context_node.py`,
`release-context-quality.py`, `release-context-boundaries.py`,
`release-context-suite.py`, and `long_context_dry_tests.py`. The node helper,
quality validators and suite are copied literally; their recorded source hashes
are in `provenance.json`. The boundary client has one documented adaptation:
`answers()` now uses the same real-chat helper, empty tools, thinking budget16,
cap128 and one-token calibration instead of a bare completion capped at32.
The original8K repetition/length failure is retained in provenance. Exact
answer/stop validation, raw context400 and cancellation SSE checks are unchanged;
the corrected client still requires fresh native qualification. Runner/profile changes only
replace workstation configuration with explicit input and carry the numerical
switch below. The source and configuration must be frozen together.

The fixed profile retains memory=swap114GiB, cpuset0-19, shm1GiB, BF16 KV,
prefill1024, capacity4, utilization0.90,4GiB reserve, snapshot rollback,
zero prefix-cache slots/checkpoint interval, no prefix reuse, and no decode
graphs. `env -i` discards inherited image/ambient selected-MTP flags. Explicit
per-node RDMA interface/HCA/GID are supplied in JSON; all other reviewed
environment settings remain literal. Linux/cgroup-v2, ARM64 GB10 images,
`/dev/infiniband`, the configured model mount, and the documented resource
layout must actually exist. This is not an automatically adapting launcher.

`paged_prefill_bf16_gemm` is a mandatory JSON boolean in this portable input:
`false` sets `ATLAS_GLM_PAGED_PREFILL_BF16_GEMM=0` on both ranks, and `true`
sets it to1. Numbers/strings/null are refused. No other numerical flag changes.
The explicit OFF assignment replaces the old external omission without
changing the OFF numerical policy. ON is an opt-in candidate requiring its own
fresh exact-image quality and safety receipts; never infer equivalence or gain
from the option's existence.

## Supply configuration and pins

Copy `input.example.json` to a trusted private directory outside Git. Replace
every `REQUIRED_...` value with actual data; choose one supported context and
the explicit boolean. Supply full nonzero image IDs and the actual server ELF
SHA256 (both ranks must contain that ELF). Keep image tags/labels distinct from
these hashes. Supply both literal `user@IPv4` SSH destinations, the head's RDMA
address, matching HTTP head/port, weights paths, existing key/known-hosts paths,
controller SSH executable and node Python executable. Python3.11+ is required
by the retained `hashlib.file_digest` calls on controller and nodes.

The SSH account must already be explicitly authorized for Docker and the
required host observations. The retained node script executes Docker directly;
this bundle does not silently add sudo, install keys, grant permissions, pull
images or build anything. Docker-administrator/controller trust is assumed.
Keep credentials and generated/session receipts outside Git. Root-owned private
operator directories require the same authorized local-root execution used by
the release operator; do not loosen permissions to avoid that requirement.

Leave `workload` as the literal `GENERATED_LOCAL_WORKLOAD`. The local-only
configurator fills the adjacent suite/client paths, hashes each actual source
and the explicitly pinned Python executable, and derives API/context arguments
from the same input. It never creates a runner or invokes SSH/Docker/HTTP.
It preserves the1800s workload/suite budgets, and requires explicit tested
concurrency widths; starting with1 before extending through4 is allowed.

```bash
/absolute/pinned/python3 -B /absolute/checkout/scripts/dev/glm_release/long_context/configure.py \
  --input /private/operator/long-context-input.json \
  --output /private/operator/new-long-context-launch.json \
  --python /absolute/pinned/python3 --python-sha256 FULL_PYTHON_ELF_SHA256 \
  --concurrency 1 2 3 4

/absolute/pinned/python3 -B /absolute/checkout/scripts/dev/glm_release/long_context/long-context-runner.py \
  --dry-run /private/operator/new-long-context-launch.json

# Only in the exclusive, explicitly authorized native window:
/absolute/pinned/python3 -B /absolute/checkout/scripts/dev/glm_release/long_context/long-context-runner.py \
  --run /private/operator/new-long-context-launch.json
```

Do not run at4096,8192 and16384 concurrently. Each needs a fresh private
output directory, separately generated launch record and actual headroom check.
Dry-run validates inputs/local pins and prints planned argv/environment; it
does not prove nodes are idle, images fit, or the model works. No source edits
between pinning, dry-run and execution.

## Operational safety and evidence

Serialize all GPU/model/build jobs across both nodes before launching. The
runner arms node-local watchdogs before create/start, pins the full container
ID/image/actual copied ELF, and checks fresh pair observations, at least4GiB
host headroom, zero swap and actual cgroup limits/OOM events. Node lease30s,
node campaign2400s, controller2300s, readiness300s,10s freshness, workload1800s
and existing bounded stop/cleanup policy are unchanged. These are operational
deadlines subject to OS/Docker scheduling, not uninterruptible enforcement.

Near-full quality and tool-call/result checks precede the boundary supplement.
The latter requires exact HTTP400 at the configured input boundary, actual
streamed text/response ID before client close, then distinct answer/tool-result
reuse. Client close/HTTP health do not prove immediate GPU reclamation; server
traces establish occupancy and actual teardown. A child timeout/failure stops
the suite; do not enlarge budgets or remove validators to obtain success.

Success also requires the runner's final record: both retained containers
stopped with exit0, no OOM/watchdog cleanup, and no configured native fault
marker. Retain controller receipts, raw HTTP records, both container logs,
node `/tmp/atlas-longctx-<session>-rN` watchdog receipts and image/source pins.
Containers are stopped and preserved, never removed by these scripts. Cleanup
uncertainty is a failure, not permission to delete evidence or start overlap.

Offline controls (no node/HTTP/subprocess dispatch):

```bash
python3 -B long-context-runner.py --selftest
python3 -B configure.py --help
```

The selftest uses the existing explicitly mocked memory/identity controls and
pure profile checks. It is not a connected lifecycle or numerical test.
