# GLM MTP first-attempt private-KV probe

Status: plan only, awaiting root's complete read and approval. No Rust, Python,
Cargo, nodes, native builds or GPU actions in this partition yet. The kernel
family agent owns integrated Rust/Cargo. Root alone owns activation and hardware.

## Question and exact boundary

v23 established identical full BF16[4096] post-EH rows for all48 selected
cross-rank step0 pairs, but different final proposer hidden rows. This does not
establish equality of the private causal KV prefix. The next narrow probe adds
two byte observations under the existing default-off hidden-trace flag:

1. Canonical full valid K/V prefix immediately before the proposer body.
2. The single new K/V row immediately after that body succeeds.

Only **attempt1, step0 of each request generation** is eligible. This means the
first admitted trace attempt, not the first successful copy or first successful
proposal. If its admission, input, EH, prefix, body or appended-row observation
fails, the budget stays spent. Attempts2..8 and steps1..3 must not perform any
additional KV lookup, allocation, capture query, copy, synchronization or hash.
The existing hidden/input/post-EH/final observations retain their current budget.
No new serving flag, cache writes, zeroing, repacking, kernel changes, model
fallback, KV repair change or performance claim.

The first-attempt restriction avoids later diverging draft histories as a
causal confound and bounds extra D2H to less than4MiB per request/rank. It does
not prove that the initial target prompt capture itself was identical.

## Source audit and first-prefix accounting

- `weight_loader/glm5/mtp.rs` constructs the body with full64 Q/KV heads,
  TP=EP1; `layers.rs` pins dimension overrides. Its runtime context carries
  target-local counts but `comm=None`. Scalar MoE uses loaded expert tables;
  TP/EP collective branches require a communicator. Scalar absorbed MLA calls
  `paged_decode_attn_bf16` directly, without external split-K workspace.
- `PagedKvCache::alloc_block/free_blocks` changes ownership but does not zero
  pools. This is not itself an invalid read: current BF16 attention bounds its
  reads by logical sequence length. Compare only those valid bytes.
- `kv_rows.rs::prefill_kv_batched` writes shifted pairs from tokens[1..P] and
  target hidden rows[0..P-1], publishing `state.seq_len=P-1` after completion.
- `repair.rs::plan_repair`, using the real `Limits::bootstrap`, then writes the
  missing terminal prompt pair and publishes `state.seq_len=P`. The first
  proposal is at target positionP+1; it appends logical cache rowP.
- In v23's repeated benchmark, P=148: the log reports147 primer rows, then the
  first hidden record has position149/cache_before148/cache_after149. The probe
  must include all148 rows, including bootstrap repair, not just147.
- v23 quality receipt has P=1984, so its first prefix is1984 rows and new row
  index1984. First-proposal repair requires position<=context_tokens; with the
  admitted context2044 the largest first prefix is2043, not2044 or2047.

The current full arena has1025 rows, so the scalar full-head Q/attention scratch
requirements (64KiB absorbed rows) fit the measured32MiB buffers. This probe is
not a substitute for an independent generic scalar-arena sizing fix.

## Exact observations and byte budget

Cache geometry is one layer, BF16, one latent head, width512, block size16:
one side's row is1024 bytes and one side's block is16,384 bytes. Keep one
reusable32,768-byte host scratch buffer. Its two16KiB halves hold K and V for
one logical block; the appended-row reads reuse those halves. No GPU allocation.

For each prefix block, read only `min(16, remaining_rows)*1024` bytes per side.
Do not use `read_block`, since it also copies unwritten tail rows. Use the
actual pool pointers plus checked physical-block offsets and the existing
`copy_d2h_on_stream`. Never read reserved future blocks merely because they
appear in `state.block_table`. Exact added transfer budget per request/rank:

| Case | Valid prefix rows | Prefix bytes | Appended row bytes | Total extra D2H |
| --- | ---: | ---: | ---: | ---: |
| Repeated v23 benchmark |148|303,104|2,048|305,152|
| v23 quality prompt |1984|4,063,232|2,048|4,065,280|
| Largest admitted first proposal |2043|4,184,064|2,048|4,186,112|

The maximum is4MiB minus8192 bytes. Whole-block rounding must not increase
these totals. In the benchmark, the last block has four prefix rows, not16.
In the worst case it has11 prefix rows. For the1984-row quality case the prefix
ends exactly on a block boundary; appended row0 is in the next allocated block.
Maximum copy calls are2*ceil(P/16)+2, hence258 at P2043,250 at P1984 and22 at
P148. Their synchronization overhead invalidates throughput comparisons.

Other transient host storage is bounded separately: at most128 physical block
IDs (512 bytes), pool/shape/cursor witness fields and SHA256 states. No full
prefix host vector, no per-row digest list and no model-sized metadata. Avoid
placing the32KiB buffer inside a record copied by value; allocate/own one
bounded host scratch for the selected invocation and reuse it for both hooks.

## Checked production binding before the first KV copy

Use the real head-owned `self.kv_cache` lock and the real proposer state's
block table. Never accept an arbitrary raw pool pointer or caller-asserted
geometry as sufficient authority. Existing `arm_prepared` checks must remain
intact: request generation/slot, cold capture, fixed C1 TP2/EP2 MTP4 repair,
no adapters/carry/prefix reuse, correct saved hidden owner and eager stream.

For the selected first attempt additionally validate, before any KV copy:

- The actual `RepairPhase::Proposed` generation and position agree with the
  trace record; first hidden_row is0; `state.seq_len == cache_before`,
  `cache_before+1 == request.position`, and the repair plan's speculative end
  still matches `state.seq_len+4`. All arithmetic is checked.
- `1 <= cache_before <= 2043` for this exact bounded profile; appended row lies
  within actual pool capacity. The number of layers is1, block size16, latent
  heads1, width512, actual dtype BF16; reject incompatible per-layer dimensions,
  non-BF16 overrides and rolling-window configuration. Require sparse-index
  selection to remain inactive (`cache_before+1 <= ctx.config.index_topk`), so
  equal K/V is not mislabeled as equality of all sparse-attention history.
- Actual K/V block strides are exactly16,384 bytes, pool bases non-null/aligned,
  checked pool byte extents agree with actual capacity, and K/V spans are
  disjoint. Bind these actual owners, not similarly shaped arena buffers.
- Every allocated entry in the bounded block table is in range, unique and
  exclusively referenced (`ref_count==1`). It covers prefix plus appended row;
  any preallocated future entries are validated but not read. Store an exact
  bounded copy of this table plus pool/geometry/cursor identity as the witness.
- Derived read spans stay inside their respective actual pool spans and are
  exactly the intended valid logical rows, including nonzero/shuffled physical
  IDs. Block0 is valid in this private pool; do not inherit target reserved0.
- Both `ctx.graph_capture` and actual stream capture state reject before I/O.
  Flag-off/unselected records return before these checks and capture queries.

Validate the same witness after the body before reading the appended row:
pool bases/capacities/strides, exact block table, exclusivity and geometry must
not change. The actual body receives the bound block table; no backend address
ledger fallback is inferred. The post hook runs before `state.seq_len += 1`,
so it checks the old cursor plus the expected appended logical row explicitly.
The existing final-hidden hook still checks the incremented cursor afterward.

## Hook order, failure closure and hashing definition

Inside `forward_body_one`, retain existing EH and post-EH behavior. After the
body's existing block-capacity preparation and metadata upload, but before
residual memset/body decode, perform the selected prefix hook. Then execute the
unchanged memset and body. On body success, while holding the same cache lock
and before cursor increment, validate the witness and read the appended row.
All copies use the proposal stream. No default-stream read, global synchronize,
additional numerical kernel or pointer mutation. An earlier pending operation
on the same stream completes before its corresponding host read returns.

An error propagates normally. Do not catch it to serve a draft or emit a
complete success trace with absent data. The trace attempt was already spent
by `HiddenTrace::begin` before admission/copies, and does not get refunded.
Duplicate prefix/post calls reject; post without prefix rejects. No new counter
may reset on the same generation or permit a second eligible attempt. Request
free/new-generation behavior follows existing trace ownership exactly.

Define canonical SHA256 byte streams precisely, using little-endian integers:

```text
prefix = ASCII("atlas/glm53/mtp-kv/prefix/v1\0")
       || u64(prefix_rows) || u32(512) || u32(2)
       || K_row0[1024] || V_row0[1024] || ... || K_row(P-1) || V_row(P-1)
appended = ASCII("atlas/glm53/mtp-kv/appended/v1\0")
         || u64(logical_row_index) || u32(512) || u32(2)
         || K_new[1024] || V_new[1024]
block_map = ASCII("atlas/glm53/mtp-kv/block-map/v1\0")
          || u32(number_of_bound_entries) || physical_id0:u32 || ...
```

`2` is the BF16 byte width. Hash raw storage bits, not converted floats; read
both K and V even though NoPE normally makes them equal. Semantic hashes
exclude physical IDs, addresses, reserved rows and block tail bytes. Map digest
is separate provenance, never required to equal across ranks/requests. These
hashes compare observations, not mathematical equivalence or causality.

## Versioned producer/analyzer and exact file ownership

Proposed edits only after root approval and module/Cargo release:

- `layers/glm5_mtp.rs`: pass mutable trace through the two new hooks at the
  specified boundaries; no body, repair, cache allocation or final norm changes.
- `layers/glm5_mtp/hidden_trace.rs`: private record fields/first-attempt selector,
  version3 emission and narrow delegation. New private child
  `hidden_trace_kv.rs` holds checked witness/read/hash code, <=500 Rust lines.
- New `hidden_trace_kv_tests.rs` and optional `hidden_trace_kv_test_gpu.rs` for
  actual hook/owner/fault tests, plus minimal extension to the existing trace
  fixture if required. Existing adapter and post-EH tests must stay active.
- `scripts/dev/analyze_glm_mtp_hidden_trace.py` and its tests: strict version3
  parsing and availability-aware comparisons; existing v1/v2 receipts remain
  readable with KV evidence explicitly unavailable.
- This plan and the existing hidden-trace plan status/cross-reference only.

Version3 emits the existing exact fields plus `kv_prefix_sha256`,
`kv_appended_sha256`, `kv_block_map_sha256`. All three are64 lowercase hex only
for attempt1/step0; all are literal `None` on every other record. Their counts
and appended index derive from existing `self.cache_before`, not duplicate
unverified producer fields. Strictly reject missing/unknown/duplicate fields,
partial digest sets, version mismatches and digests on ineligible records.
Mixed schema within a request/rank remains an error. The analyzer must report
KV evidence availability separately for cross-rank and repeated-request pairs;
it must not fill unavailable evidence with equality, or treat map inequality
as semantic-prefix inequality.

For comparable first-attempt observations only:

- Different prefix: evidence points upstream of current body (primer/repair/
  prompt-hidden capture); it does not alone prove a cache bug.
- Equal prefix/post-EH but different appended row: isolate input norm/WKV/
  cache assembly/write or unexpected mutation, not later MoE arithmetic.
- Equal prefix/appended/post-EH but different final hidden: next distinguish
  Q/attention/O projection versus FFN. Current hashes do not verify static
  weight equality or all other body scratch state.
- Missing/failed/incomplete probe: explicitly unavailable; no causal verdict.

## TDD, review and root-owned native gates

Before implementation, execute behavioral RED against real production hooks
and checked readers, not a test-only digest helper. CPU recorder supplies real
bounded BF16 pool byte storage and records exact D2H addresses/length/stream;
the body stand-in writes distinct appended bytes through its actual callback.
It does not simulate MLA numerics or claim CUDA correctness.

Tests must cover exact148/1984/2043 byte totals, boundary rows1/15/16/17 and
nonzero/shuffled/block0 physical maps; identical semantic bytes with different
maps; changed first/last valid K/V byte; changed unused tail/future-block bytes
leaving hashes unchanged; exact appended-row index and no adjacent read.
Test actual wrong shape/dtype/layer/stride/capacity/null/alignment/overflow,
duplicate/out-of-range/nonexclusive blocks, cursor/repair/request mismatch,
insufficient map coverage and changed post-body pool/map. All malformed-owner
cases reject before the first KV copy. Inject failure at every prefix copy and
both appended copies; verify unchanged budget, no later KV reads, no success
record, and no GPU allocation/mutation/implicit default-stream calls. Include
flag-off, attempts2..8, exhausted trace, steps1..3, stale/same/new generation,
actual stream capture, missing/duplicate/out-of-order hooks and body failure.

Analyzer RED/GREEN covers exact v3 schema and availability, all malformed
combinations, old v1/v2 compatibility, mixed versions, and differing physical
maps with equal logical bytes. Full model library CPU suite, non-test check,
workspace formatting, SPDX, file caps and independent exact-hash source review
precede root commit. Persist receipts under
`atlas-campaigns/20260908/hidden-trace-first-kv/`.

Root then builds/deploys the frozen diagnostic slice and repeats cold C1 paired
requests, the1984-token quality case and cancel/recovery with existing memory
guards. Acceptance, answers and model health remain gates, not predicted
outcomes. This trace is not eligible for throughput promotion or performance
comparison. No model activation is authorized by this plan alone.
