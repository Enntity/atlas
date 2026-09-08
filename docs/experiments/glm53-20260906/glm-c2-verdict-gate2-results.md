# GLM C2 request-owned verdict and continuation: Gate 2

2026-09-08. The complete controller CPU model suite passed 1,045/1,045
in 82.35 seconds. The focused handoff suite passed 69/69. This remains an
unactivated implementation: no factory, scheduler admission, launcher, CUDA
arithmetic, serving image or model throughput change is included.

## What this closes

After each actual four-draft proposal, the selected paired head seals its
five verification inputs, absolute position, slot generation and committed
token prefix. An actual K5 target operation owns shared normalized scratch
until its matching verdict detaches the accepted rows and bonus into that
request's existing six-row slab. A peer cannot overwrite an undetached result.

For acceptance count a in 0..4, detachment copies a accepted hidden rows and
one bonus row: at most two D2D copies, 40,960 bytes per owner. It allocates no
new GPU storage. The existing two-owner indexed private-cache and slab total
remains 10,846,208 bytes at context 2044, excluding target state, weights and
scratch. One host token-prefix receipt is reserved to the validated context
per slot, at most 8,176 payload bytes per owner, excluding allocator overhead.

Verdict recording, proposer trimming and target SSM commit are distinct
acknowledgements. Repair requires both trim and commit, orders the actual
secondary-stream event before consuming recurrent state, and uses only the
request's retained accepted/bonus rows. The existing pair planner and KV writer
remain the arithmetic authority. Zero acceptance keeps the seed pair; full
acceptance repairs the fourth pair beyond the old speculative cache end.

Raw verifier argmax is deliberately not treated as the bonus-token authority.
The actual scheduler may apply repetition penalties or other logit processing.
Gate 2 accepts the valid caller-selected next seed and seals the resulting
next proposal; binding the actual scheduler selection to the worker's E1
payload remains an explicit serving-integration requirement.

## Actual-path CPU evidence

Tests call the real Model K5 wrappers, real paired head, actual private KV
writer, rollback/commit hooks and worker F5 dispatch. Numerical kernels and
collectives are substituted at the backend boundary. Recognizable token,
hidden and physical K/V bytes are checked independently of event counts.
These are ownership and ordering tests, not a KDA/MLA numerical oracle.

- All 25 two-owner acceptance combinations, both rank configurations and both
  owner orders, plus repeated alternating zero/full/asymmetric continuation.
- Actual worker F5 with the legacy repair flag off, repeated transactions,
  owned accepted rows and internal next proposal. Public C2 E1 still refuses
  before its payload; successful public paired E1 is not claimed.
- Interleaved global scratch overwrite after detachment, exact shifted
  token/hidden pair checks, a selected bonus different from raw argmax,
  retirement/reuse and a continuing peer across capture-generation changes.
- Actual tiny SSM pools and claimed slots, all acceptance counts, live and
  intermediate pointer identity checks, same-index foreign-pool guard refusal,
  commit copies and event ordering. This does not establish production KDA
  arithmetic or native graph replay.
- Eighty control-derived fault injections across target/normalization/readback,
  accepted/bonus copies and completion, accepted-pair KV writing, first/last
  reproposal body steps, and actual worker verdict receive/read/count errors.
  Additional tests cover SSM copies/events/waits and teardown failures.
- Alternate legacy producers, rollback/restore/normalization and compaction
  refuse selected calls before backend work. Their FullAttention fixture
  proves early refusal, not the numerical correctness of legacy SSM restore.

An error while verification scratch is still owned blocks all subsequent
selected producers until abort/close; peer bytes remain preserved, but peer
progress is not promised. A detached private repair failure quarantines only
that owner; tests execute the healthy peer's real continuation. Successful
later synchronization never reopens a failed lease.

Selected retirement disarms the actual target SlotGuard even when early
retirement fails, preventing Drop from returning a quarantined slot. Healthy
retirement waits for target commit before zeroing/releasing. Model close joins
the actual secondary stream as well as default: a failed event record can
leave writes outside the last event's history. Failed completion forbids
Model-path frees and sweep. The existing native CUDA backend Drop independently
sweeps its ledger; unknown-completion native recovery is not proved or changed.
The close contract assumes drained, serialized host lifetime.

## Receipts and remaining gates

Development receipts are in `atlas-campaigns/20260908/glm-c2-verdict/`.
Behavioral RED receipts are retained separately from compile-only failures and
invalid fixture expectations. In particular, a worker test initially expected
the refusal text to contain `C1`; production correctly said
`max_batch_size=1`. That assertion correction is not a production fix.
The final test-only compile correction explicitly copied fixture cache-config
fields because `KvCacheConfig` is not Clone.

The frozen 38-file manifest (36 Rust files and both implementation plans) has
SHA256 `eac4236c538bc527cdc18d29186ad4fab90b2a5131c7246858d216edec0f81e5`.
This result document is outside that manifest. After the full test run, only
one safety-comment line was combined to retain the existing verifier's
500-line limit; there was no executable delta. Source was committed as
`f8a0bdb9faa38797021a720a4ecf6cea79db59d3`; the clean post-commit full CPU
suite also passed 1,045/1,045 in 81.56 seconds.

`glm-c2-verdict-f8a0bdb9-receipts.tar` contains the frozen source, committed
result document and complete development receipts, including the post-commit
run. Controller and head phase7 copies have SHA256
`ed3136b7a79afcb7b03f2ff353e518ba511e7543afa1d7754744b6acf095ef51`.
The archive predates this provenance paragraph and contains no serving binary
or controller CPU shim.

Non-test library checking passed in 9.01 seconds. Workspace formatting,
scoped SPDX first-line checks and `git diff --check` passed. All 16 new Rust
files are at most 482 lines. Clippy stops before model analysis on the same
four inherited runtime Metal-stub `too_many_arguments` errors. The inherited,
non-allowlisted `glm5_mtp.rs` file-size violation remains (885 versus 882
lines at the base); other oversized changed files are allowlisted. No
whole-workspace CI-green or native serving qualification is claimed.

The next chunk is a serialized, slot-addressed C2 serving control. It must
validate before issuing F5/E1, finish both ranks' transaction even when output
ends, preserve the actual penalty-aware selected token, reject unsupported
requests before allocation, and stop paired serving after a protocol failure.
Only after that control is correct should two K5 target passes be replaced
with a segmented ten-row pass and the qualified standalone kernel candidates.

The measured warm baseline is unchanged: roughly 28.6–28.7 full-wall tok/s for
the separate optimized C1 MTP4 profile, 19.0 aggregate nonspeculative C2, and
47.2–47.4 aggregate nonspeculative C4. The 30 C1 / 60 C4 target is not yet met.
See the [implementation plan](../../../scripts/dev/glm_c2_speculation_plan.md)
and [Gate 2 contract](../../../scripts/dev/glm_c2_verdict_gate2_plan.md).
