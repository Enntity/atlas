# Portable GLM dual-Spark release qualification

This is an operator handoff, **not a native PASS receipt**. Check the exact
source/image qualification in
[the release checkpoint](../../../docs/experiments/glm53-20260906/release-candidate-20260910.md)
and [merge evidence](../../../docs/experiments/glm53-20260906/upstream-release-integration.md).
Never label a newly built image qualified using an older image's results.

The supplied profile is TP2/EP2-v2, eager selected MTP, capacity eight,
context2044, prefill1024, four drafts, BF16 KV, snapshot rollback and zero swap.
It is not long-context MTP. The paired M10 control uses owner mode `0`; the
wider transaction uses owner mode `joint`. Change that literal on **both**
ranks for a same-binary A/B; do not change other numerical settings or budgets.

Both ranks explicitly set `--ssm-cache-slots=0` and
`--ssm-checkpoint-interval=0`. Selected admission rejects prefix reuse, and
the2044-token cap is below the ordinary4096 checkpoint interval: the default
16 Marconi prefix-cache slots therefore reserve about1.2GiB without serving
this profile. This adjustment removes only that unused prefix-cache reserve;
it preserves all live SSM/MTP state, snapshot rollback, the4GiB CUDA reserve,
utilization0.90, memory=swap114GiB, and all numerical/health/lease/watchdog
settings. The initial merged native attempt safely refused at818 worker KV
blocks versus the required8×128; the controller stopped both ranks with
exit137, OOM=false and zero swap. **Native retry with these explicit flags is
pending; this recipe change is not a successful serving or fit receipt.**

## Contents and provenance

- `recipe-generator/`: the retained bounded native recipe generator, with a
  relative dependency on the real `atlas-glm-pair-wire` codec. Only its package
  and usage name changed. No Docker, SSH, GPU or model operations.
- `materialize.py`: local-only generation from complete explicit inputs. It
  verifies the supplied generator hash, creates a new private directory, copies
  the exact workload bytes to `workload.input` with mode0600, and derives
  recipe/workload hashes. It does not mint sessions or launch nodes. Checkout
  file modes remain unchanged; launch input never points into the checkout.
- `recipe-input.example.json`: complete C8 argument/environment table derived
  from the retained owner-eight campaign. Only operator network/path/digest
  values are placeholders. The two literal-zero CHECK diagnostics omitted by
  that campaign remain omitted; no ambient environment is merged.
- `launch.example.json`: the same finite C8 controller/lease/workload policy,
  with operator-specific values removed. No recorded IDs or credentials.
- `native-workload-c8.py`: byte-identical retained C8 stdlib stdin workload.
  The coding literal/token hashes, temperature0, seed1, 148-input/256-output
  requests, one warmup and three measured waves at each C1..C8 are unchanged.
  The new tool-result followups allow128 total output tokens: their32-token
  thinking budget is soft and can defer closure until about96 tokens. The
  initial64-token envelope truncated four valid own-result reasoning paths;
  those failed receipts are retained. Exact visible result and normal-stop
  checks remain mandatory. This does not change the coding benchmark.
- `test_materialize.py`: CPU-only transformation controls; no fake serving or
  authority fixture. These do not replace codec, controller or native tests.

## Build and pin artifacts

Use a clean recorded source revision. Build server/kernels using the repository's
GB10/CUDA instructions; do not set `ATLAS_SKIP_BUILD` for native artifacts.
Record the source commit, toolchain/build recipe and full hashes. Build each
helper for the machine that will execute it: controller helpers on the
controller, node supervisor/relay and container guard for the Spark's ARM64.
The recipe generator is controller-only and GPU-free:

```bash
cargo build --locked --release --manifest-path scripts/dev/glm_release/recipe-generator/Cargo.toml
cargo build --locked --release --manifest-path scripts/dev/glm_pair_guard/Cargo.toml
sha256sum scripts/dev/glm_release/recipe-generator/target/release/glm-release-recipe-generator
```

Run those commands only in your assigned build window. The helper package
contains the existing `glm-pair-supervisor`, `glm-pair-relay`, and
`glm-pair-guard` binaries; no replacement safety implementation is bundled.
Install immutable root-owned, non-writable node helpers and construct/pin both
runtime images containing the actual guard/server at their recipe paths. The
guard remains exec-form PID1, with the server inheriting its private channel.
No image pull/build/provisioning is performed by this handoff.

## Supply explicit local inputs

Copy the two JSON examples to a private operator directory outside the checkout.
Replace every `REQUIRED_...` value except fields the materializer generates:
recipe `output_directory`; node `recipe_file`/`recipe_sha256`; workload
`input_file`/`input_sha256`. Those are replaced from explicit local CLI inputs.
Unresolved remaining placeholders are refused.

Required values include:

- Full lowercase, nonzero image IDs and actual server/guard/supervisor/relay
  ELF SHA256s, not mutable tags or commit abbreviations. Node helpers and image
  contents must match those hashes. Images may have distinct rank image IDs.
- Canonical absolute model path (same host path on both ranks for this fixed
  generator), root-owned immutable node helper paths, SSH destinations, an
  existing private-key path and pinned known-hosts file. Never put key contents
  or passwords in JSON/Git. Use the unchanged controller's dedicated SSH
  connection policy and an explicitly authorized administrator account.
- The actual head RDMA address in both server argument vectors; each node's
  actual HCA, socket interface and GID index. Do not guess NCCL values from a
  different installation. The remaining complete environment describes the
  retained runtime image; verify it against the actual image's inherited
  environment. Extra image-injected entries cause exact-inspection refusal.
- The HTTP host in readiness and workload arguments, and the actual controller
  Python executable path/hash. Python3.10+ and stdlib suffice. Keep `-I -`
  before the literal base URL and expected model. The model's container path
  remains `/var/tmp/models/glm53-flash-nvfp4` throughout this fixed profile.

The generator deliberately fixes resources: memory=swap114GiB (no swap
allowance), cpuset0-19, shm1GiB, all-GPU DeviceRequest, `/dev/infiniband` rwm,
memlock unlimited, IPC_LOCK/SYS_NICE, label disable/seccomp unconfined, root
uid/gid, host network, private PID/IPC, no restart/init, no-new-privileges.
The production adapter pins runc. These are explicit reviewed deployment
choices, not portable hardware defaults. If they do not match your nodes,
stop and review a different recipe; do not weaken inspection or health gates.

## Materialize, prepare, then run once

All paths below are illustrative placeholders that the operator must replace.
The output and prepared directories must be new, outside the checkout. Parent
directories must already exist and be private/trusted. The materializer refuses
symlink/noncanonical paths and overwrites nothing. Its generator timeout is30s.
Run local materialization, `prepare` and `run` as the same authorized local
root identity using pinned executables (for example `sudo -n --` below).
The prepared directory and private records are owned by the preparing uid/gid;
switching identities between preparation and execution is refused. This
documented deployment uses root-owned mode0700 operator directories, so local
root is required to access them; the directory check itself binds the current
uid/gid rather than silently changing ownership. This is separate from the
root-only node verbs and node-side `sudo -n` authorization. Do not solve file
permission errors by granting group write or chmod-ing the checkout.

```bash
sudo -n -- /absolute/pinned/python3 -B /absolute/checkout/scripts/dev/glm_release/materialize.py \
  --recipe-input /private/operator/recipe.json \
  --launch-input /private/operator/launch.json \
  --generator /absolute/checkout/scripts/dev/glm_release/recipe-generator/target/release/glm-release-recipe-generator \
  --generator-sha256 FULL_GENERATOR_SHA256 \
  --workload /absolute/checkout/scripts/dev/glm_release/native-workload-c8.py \
  --output /private/operator/new-materialization

sudo -n -- /absolute/pinned/glm-pair-supervisor prepare \
  /private/operator/new-materialization/launch.json /private/operator/new-prepared

sudo -n -- /absolute/pinned/glm-pair-supervisor run \
  /private/operator/new-prepared FULL_BUNDLE_SHA256_PRINTED_BY_PREPARE
```

The production `prepare` validates the launch, pins its actual inputs, writes
the fresh bundle and prints its digest. `run` consumes that exact bundle once;
it performs create→inspect→seal→start, real guard/child identity and challenge
exchange, health/lease supervision, readiness, workload, matched drain,
quiescence/release and final exit checks. Never bypass it with a legacy launcher
for this selected profile. A workload `passed:true` alone is not campaign
success: both actual final containers must have exit0, OOM=false and restart0,
and the controller must complete successfully.

Before running, serialize all native jobs across both nodes and verify no
unrelated GPU workload/build or competing controller is active. Preserve at
least4GiB actual loaded headroom and zero swap. Do not infer fit from weight
size. Retain failed records/containers/logs and cleanup uncertainty; never
delete prior evidence to make a new run pass. Use fresh bundles for repeats.
Retain known-good rollback images (the checkpoint records `a069efc3`) until
new qualification completes. Restore privileges according to operator policy
after the campaign; this bundle does not provision or revoke SSH/sudo access.

## Evidence and limits

Retain exact source/artifact/bundle/workload hashes, controller output, both
rank logs, workload JSON and final Docker observations outside Git. The
workload gates eight distinct answers, eight structured **auto** tool calls,
eight followups using actual call IDs and distinct literal tool results, and
eight own/no-foreign needle checks before timing. No external tool executes.
Named-forced grammar is explicitly unsupported in this selected profile.
Needle containment is not exact-format or broad language-quality proof.

Coding totals use full wall `sum(tokens)/(last_end-first_start)` and separately
decode window `sum(tokens)/(last_end-first_text)`; post-first subtracts one
token per stream. These are client clocks, not GPU timers. Raw capped texts,
per-stream counts and TTFT remain available; generated programs are not
certified. Client concurrency alone does not prove active slots: retain actual
rank C3..C8 transaction traces. Warmups are excluded from measured medians.

The workload retains30s blocking I/O and180s checked request deadlines. Its
many sequential phases can exceed the1190s whole-workload bound under repeated
slow responses; the unchanged controller must fail then, not extend a lease or
declare success. This is a qualification budget, not a worst-case completion
guarantee. The watchdogs are unchanged from the retained C8 campaign.

Long-context4K/8K/16K qualification is a **separate** independent eager sparse
C1..C4 profile and needs its own fresh image/quality receipts. This C8 bundle
does not generate or qualify that lane, nor imply long-context C8/MTP support.

Local checks (no nodes or model execution):

The packaged generator has also been built with `--locked --offline`. Its
actual recipe output matched the retained C8 recipe bytes, and the production
controller accepted the portable materializer's mode0600 workload into a fresh
prepared bundle. This proves materialization/preparation, not serving success.

```bash
cd scripts/dev/glm_release
python3 -B -m unittest test_materialize.py
python3 -B materialize.py --help
```
