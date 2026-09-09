# v27: same-binary C2 compact MoE off/on

Root-only native experiment; this author runs no build, SSH, Docker or GPU work.
This is ordinary nonspeculative active4/admitted4, NOT paired speculative C2.
Build one reviewed binary and use it unchanged for both arms. Do not benchmark
an uncommitted author tree. This is a full committed-source native build, not
a claim that v26's Rust or kernel revision is unchanged.

## Build once, retain v26

Retained head source/build root: `/tmp/atlas-sparse-test.ZeMcYA/source/` (not
present on controller). Root's read-only container inspection establishes the
exact retained `atlas-glm53-build-20260906` recipe:

```text
image sha256:79bcc40f7d5cf3aa749c41e83ebd0d20c2154920dedf19907b4e3bbb262f4bbb
CMD bash -c 'cargo build --offline --release -p spark-server -j 2'
runc; network=none; cpuset=0,1; memory=memory-swap=8589934592; no DeviceRequests
RO mounts source/{Cargo.lock,Cargo.toml,crates,kernels,vendor} -> /build equivalents
PATH=/root/.cargo/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
LIBRARY_PATH=/usr/local/cuda/lib64/stubs
CUDA_HOME=/usr/local/cuda; CUTLASS_HOME=/opt/cutlass
ATLAS_TARGET_HW=gb10; ATLAS_TARGET_MODEL=glm-5.3-flash-nvfp4; ATLAS_TARGET_QUANT=nvfp4
```

The retained container owns incremental `/build/target`; root can restart that
same stopped container after the frozen source replacement. Do not rebuild a lookalike,
substitute the controller ATLAS_SKIP_BUILD environment, or infer its Env from
the current host. `v26-native-build.log` records189 kernels and2m13s release.

Root/reviewer selected FULL committed HEAD plus the compact change, Rust AND
CUDA, because current MoE depends on resident storage guards, private Down
returns and factory traversal since4416ede2. Do not attempt a dependency-surgical
overlay onto v26. Root first preserves the old native source tree (~33MB tar),
then replaces tracked source with one exact committed-tree snapshot, including
its matching Cargo manifests/lock, crates, kernels and tracked vendor inputs.
Record commit and snapshot hash, not an assumed HEAD identity or dirty worktree.
Retain the stopped build container's incremental target cache; let Cargo rebuild
all invalidated Rust/kernel dependencies. No claim of retaining native kernel
revision189db87e: the snapshot deliberately includes committed CUDA changes
(including the M10 export where present). No speculative factory/admission flag
is enabled by this experiment. Restart the retained release command once,
without GPU access. Copy `/build/target/release/spark` out as one `spark-v27`
binary to both hosts and compare SHA256 before packaging. Follow v26's tiny
runtime layer: `FROM atlas-glm53-flash:kernel-20260908-v26`, then
`COPY --chmod=755 spark-v27 /usr/local/bin/spark`; tag both
`atlas-glm53-flash:kernel-20260909-v27`. Verify the installed binary SHA on both
hosts (image IDs can differ across independently packaged layers). Keep v26.

## Minimal launch delta, not a 935-line fork

Controller artifacts are now prepared in campaign20260908: `v27-launcher.sh`
is a literal retained-launcher copy plus only the three-line OPTIONAL_ENV
insertion below; `run-v27-c4.sh` is the small clean v26 recipe with explicit
bit and multisequence-graph arguments. Root copies these two to head phase7.
No runner has been executed by this author; bash syntax validation only.

Use the retained `v25-launcher-a85b1a7b.sh` / head
`start-glm53-ep2-hidden-a85b1a7b.sh`. There is no generic external-env passthrough.
Its existing OPTIONAL_ENV array is quoted into the remote worker Docker command
and passed as an array to local Docker. Apply only this insertion AFTER the
existing FP4_PREFILL conditional (otherwise its assignment overwrites the entry):

```bash
GLM_C2_COMPACT_MOE=${GLM_C2_COMPACT_MOE:-0}
[[ "$GLM_C2_COMPACT_MOE" == 0 || "$GLM_C2_COMPACT_MOE" == 1 ]] || exit 1
OPTIONAL_ENV+=(-e "ATLAS_GLM_C2_COMPACT_MOE=$GLM_C2_COMPACT_MOE")
```

The derived `run-v27-c4.sh` sets image to v27; accepts
one positional arm bit0/1 (default0), explicitly preserve that argument through
its `env -i ... bash "$0"` re-exec, and export GLM_C2_COMPACT_MOE to the patched
launcher. Merely prefixing an env var to the existing clean runner loses it.
Keep all other recipe values:2048 context/1024 prefill, active4/admitted4,
TP2/EP2,v2,BF16KV,.90 memory,114GiB equal memory+swap cap,4096MiB guard,
no overcommit/swap/speculation/MTP/diagnostics, existing groupedC3/C4 and MLA flags.

Before each arm root checks both GPUs have no other compute process/container
(including isolated microbenchmarks), obtains the exclusive window, and uses
the existing v26 gate's absence/preserve checks and stop/log/rename trap.
That trap must be armed before launch; do not invoke the launcher directly
against existing container names because it contains `docker rm -f`.
Keep the existing 35s idle interval and >=4096MiB available/no-used-swap gates.

## Measurements and acceptance

Root invokes controller `run-v27-c2-compact-gated.sh ARM SHA256 GRAPHS`, where
ARM is exactly off/on/off-repeat/on-repeat, SHA256 is the approved immutable
binary hash, and GRAPHS is0(eager multisequence) or1(graphs, default).
Require V27_ROOT_EXCLUSIVE_GPU_WINDOW=1. It takes a local flock, refuses either
existing service/preservation name, verifies both image binaries, checks both
nodes have no GPU apps/build processes/build containers, and arms the inherited
stop/log/error-scan/rename trap before launching. It verifies loaded memory and
zero swap, both live binary hashes, commands/rank/env and compact/graph bits.
Partial launch failures also go through the stop trap; failed cleanup is FAIL.

Run off0 and on0 quality-only first (no redundant timing or second quality
pass). Eager means C2..4 multisequence graphs OFF, not C1 scalar graphs OFF.
Each graph arm requires its same-binary base arm's successful eager receipt.
Every ON arm also requires OFF-eager success with that exact binary on both ranks.
Then run off1 and on1; only if warranted run off-repeat1/on-repeat1. Each call
stops both ranks, checks no GPU process remains and preserves named containers.
Tags are `v27-compact-ARM-{eager,graphs}`; existing receipts cannot be overwritten.
Root captures the same actual Config.Cmd/env/memory receipts as v26 into
`v27-compact-ARM-PROFILE-live-config-rank{0,1}.log`; check the unchanged active4
recipe plus identical installed binary SHA and exact compact bit on BOTH ranks.
The campaign `run-v27-c2-compact-measure.sh off|on TAG quality|matrix` performs
client requests only, called by the gate after those receipts exist. Root sets V27_ROOT_EXCLUSIVE_GPU_WINDOW=1
only while holding the real exclusive window; this is an operator attestation,
not an automatic proof of GPU isolation or a replacement shutdown trap.

Each graph arm runs existing C2 concurrent-answer checks (checker2155b752 executes
four checks in pairs) and uneven C4 NIAH BEFORE and AFTER the matrix. The
explicit C2 answer check must pass before timing, not merely a C4 tail check.
Every quality phase also requires two concurrent actual tool-call requests with
distinct natural-language tasks, function schemas and exact string arguments.
The checker requires one parsed function call, exact name/duplicate-free JSON
arguments and finish_reason=tool_calls; it never executes a tool or accepts
reasoning/prose as a substitute. Initial gate uses supported specific tool_choice:
forced named-tool correctness only, not automatic tool selection. A separate
auto-choice gate remains necessary for a broader tool-selection claim.
The existing benchmark uses literal148-token LRU workload,
temperature0/seed1, ordinary EOS,256 requested tokens, C1..4, one retained warmup
per width and three measured waves. Validate every warmup/measured request
reaches256; preserve texts, workload token hash, cap/quality failures unchanged.
Compare only matching workload hashes and same-binary recipes. Report existing
aggregate-E2E median plus TTFT/decode-window metrics; C2 is the hypothesis,
C1/C3/C4 are regression controls. OFF versus ON isolates the compact switch in
the same new binary; historical v26 comparisons include other source changes.
A speedup with failed quality is not a win.
After graph-run post-matrix quality, reuse the v26 cancellation request
`v23-hidden-cancel-request.json`: curl max-time2 must return28 with nonempty SSE,
then the existing C2 concurrent-answer checker must pass recovery. Retain those
receipts as ordinary cancellation/slot-reuse evidence; eager quality-only runs
skip this phase. The matrix parser's workload.prompt_tokens key is verified
against current benchmark workload_metadata and retained v26 matrix JSON (148).
After each arm use existing `check_glm53_native_log.py` on both final logs,
verify stopped/non-OOM containers and no used swap, then preserve/rename them.
Do not overlap arms, builds, microbenchmarks or unrelated serving GPU work.
