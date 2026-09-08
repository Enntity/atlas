# GLM first-cold-prompt source probe

Status: source and immediate-writer addendum implemented; CPU full suite912/912
and analyzer24/24 pass. Independent final source review is completing; non-test
check receipt is recorded below. No native claim. Root alone owns execution and will not deploy this
diagnostic until the separately identified target-to-primer stream ordering
defect has its own reviewed fix and behavioral tests.

## Narrow question

v24 observed six first-attempt prefixes differing on rank0, with stable rank1
prefixes, identical physical block maps, and identical current input/post-EH
and newly appended row148. That does not establish equality of captured prompt
rows0..147, or of the bootstrap-written KV row147. Distinguish the actual
captured source from the later KV writer by observing **consumption-time source
bytes**, not a later snapshot that might have changed since consumption.

Add only two source observations under the existing default-off hidden trace:

1. Before eager primer consumption: all captured BF16 rows `[0,P-1)`.
2. Before first bootstrap write: captured BF16 row `P-1`.

The existing first-attempt v3 KV prefix and appended-row probes remain intact.
The addendum reads immediate written KV without adding a projection hook,
timing or numerical change. Different sources locate divergence in target
capture or earlier work; they do not prove a capture-copy defect.

## Scope and precise transfer envelope

Only cold C1 TP2/EP2 MTP4 repair requests with prompt length **2..=256** get
source observations. Require the existing whole-prompt eager primer; no serial
primer, carry, prefix reuse, mixed decoding, adapters or first-propose fallback
may silently substitute for that execution. First attempt only, not each draft
or later repair. The established v3 first-prefix limit1..2043 remains unchanged.

For P>256, source evidence is explicitly unavailable and performs no source
copy/capture query/host-buffer allocation. Existing v3 KV probes still run on
their admitted profile, including the1984-row quality case. P1 likewise has
source evidence unavailable because it has no eager primer span. Do not refuse
an otherwise valid long quality prompt solely for this source-probe limit.

| Prompt P | Primer hidden bytes | Bootstrap hidden bytes | Immediate written KV bytes | Existing v3 KV bytes | Combined diagnostic D2H |
| --- | ---: | ---: | ---: | ---: | ---: |
|148|1,204,224|8,192|303,104|305,152|1,820,672|
|256|2,088,960|8,192|524,288|526,336|3,147,776|
|1984|0|0|0|4,065,280|4,065,280|
|2043|0|0|0|4,186,112|4,186,112|

Every row is BF16[4096]=8192B. One32KiB reusable host payload buffer reads at
most four whole rows per copy; no GPU allocations or prefix-sized host vector.
At P148, source copies are37 primer copies plus1 bootstrap copy; at P256,
64 primer copies plus1 bootstrap copy. Reuse the source buffer for the phase's
immediate K/V reads, then drop it; no payload persists across phases/requests.
Existing KV32KiB buffer is allocated later. Digests, fixed-size owner/generation
metadata and a SHA256 continuation persist, not a full prefix or block map.
The maximum across all admitted profiles stays4,186,112B, below4MiB; this is
diagnostic traffic only, and invalidates throughput comparisons.

## Actual ownership and hook placement

The capture allocation belongs to TransformerModel (`mtp_prefill_hidden`,
`mtp_prefill_capacity`), separate from BufferArena; `try_mtp_prefill_capture_from`
normalizes GLM target rows directly into it on the prefill stream. At chunk0 it
claims `mtp_prefill_capture_gen`; only contiguous same-generation chunks append.
The eager wrapper calls `ensure_drafter_context` on its supplied consumer
stream. Audit found that the TP2/EP2 non-mixed dispatch shadows its incoming
stream to default for target/capture, while that wrapper previously retained
the original scheduler stream. This diagnostic does not add a fence or fix;
its source hash observes the actual consumer stream and does not itself prove
capture-producer completion. A separate ordering fix must precede deployment.

Introduce a narrow model-private binding in a new
`model/glm_mtp_prompt_trace.rs` child. Its descriptor constructor is private to
that owning model implementation, using actual capture base/capacity, actual
generation/length atomics and actual SequenceState. Do not admit arbitrary raw
pointer plus caller-asserted capacity. The descriptor can expose checked
read/identity comparison methods to the GLM diagnostic child, not a general raw
pointer accessor or new generic DraftProposer capability.

Before source I/O, validate the full existing trace profile as applicable at
prefill (same adapter installation-history closure, exact model/config/ranks,
C1, supported repair storage/environment, cold no-reuse ownership). Actual
repair must be enabled, its existing environment guard must pass, and actual
slot verify capacity must be at least4. Prefill SequenceState does not expose
future requested draft depth or grammar: do not fabricate those facts. Source
observations may precede knowledge of unsupported future depth/grammar, remain
bounded/spent, and cannot be emitted unless later arm_prepared validates actual
requested4/no grammar. Unlike arm_prepared, this
phase requires RepairPhase::Capture and private cache rows0, not Proposed.
The live model communicator must supply rank; the existing primer compute
context intentionally has comm=None and cannot supply rank provenance.

Require actual capture length=P, tokens length/seq_len/prompt_len=P at eager
consume, nonzero equal sequence/capture generation and correct slot identity,
capacity covering P rows, non-null BF16-aligned checked span, and source span
disjoint from all mutable writer arena spans and actual private KV pools.
Use existing validate_kv_inputs to preserve the writer's scratch/weight checks.
Read only the selected valid capture bytes; never whole capacity or unwritten
tails. Validate both caller graph_capture and actual stream capture before I/O.

Model binding invokes a private GLM trace entry immediately before the existing
`proposer.prefill_drafter` call in `ensure_drafter_context`, after its actual
cold-owner eligibility is known and before any writer operation. Record a
request-owned source phase as Spent before validation/copy; success records the
primer digest plus owner, generation, P and exact shifted token digest. Confirm
the real primer subsequently commits exactly P-1 rows; failure cannot create
complete source evidence or be retried as a fresh observation.

At `repair.rs::prepare`, after the existing full plan validation but immediately
before bootstrap `write_kv_rows`, require the stored primer witness to match
RepairInput.capture base/capacity, generation, prompt length and Capture phase.
Require its write exactly `(cache_start=P-1, hidden_start=P-1, token_start=P,
rows=1)` and prior cache rows=P-1. Hash that exact source row and the actual
bootstrap token. The scalar writer then runs unchanged. Commit the bootstrap
evidence only after write success and immediate KV read, matching the existing
failed-phase policy.
Later Pending repairs are source-probe inert. First arm_prepared binds the
completed witness to attempt1; failures spend it, with no retry on attempts2..8.

### Addendum: immediate writer evidence

Read primer KV `[0,P-1)` after execute_kv_rows completes its existing stream
synchronization, before publishing phase success. Read bootstrap KV `[P-1,P)`
after its unchanged writer finishes, before repair commits Proposed. Both use
the real head-owned private cache under lock and the phase's actual stream.

Extract private read-only owner validation from hidden_trace_kv into a reusable
child if needed. Require an explicit valid read interval, not the current
Probe::before assumption that its future appended row is allocated. A P17
primer writes16 rows in one block: no second block may be required until repair.
Post-bootstrap reads exactly rowP-1. Validate ALL provided block map entries
(range, uniqueness, exclusivity, bounded size), but read only the valid interval.
Preserve actual BF16 one-layer/head512/block16 geometry, checked aligned/
disjoint pool spans and eager-capture guards. Do not relax any existing v3
Probe::before/after contract. No arbitrary raw pool-owner constructor.

At most one active map snapshot<=512B and one32KiB host payload; drop phase
snapshots before the next phase. Retain no primer map vector across bootstrap,
which may legitimately append allocated blocks. Retain fixed-size identity,
digests and one SHA256 continuation only.

Compute three digests from these same bytes, with no additional reads:

- `prompt_primer_kv_sha256`: existing canonical prefix domain/countP-1.
- `prompt_bootstrap_kv_sha256`: existing appended-row domain/indexP-1.
- `prompt_written_prefix_sha256`: initialize a SHA256 continuation with the
  existing v3 prefix domain/countP, feed immediate primer K/V in canonical
  order, retain the fixed-size state, then feed bootstrap K/V and finalize.

The composed digest is directly comparable to the later v3 kv_prefix_sha256.
Separate147-row and one-row hashes cannot be combined after finalization; the
continuation avoids that gap without copying/storing the prefix. It represents
values observed at TWO writer completion times, not one atomic snapshot.
If composed and later same-rank prefix differ, an observed byte changed after
its writer observation (including possible bootstrap clobber of older rows).
Equal sources/tokens but different immediate writer outputs locate writer/
output-state divergence. Equal composed prefixes followed by different body
prefixes locate intervening storage change. Neither proves mathematical error
or identifies a specific kernel.

No requirement that primer and bootstrap streams have equal numeric handles:
primer completion already synchronizes its actual prefill stream. Bootstrap
uses the actual repair stream after the normal scheduler handoff. Store the
phase's actual stream identity for validation, not in semantic hashes.

The immediate observations retain a fixed-size pool/capacity and logical-prefix
map digest. Revalidate the P-1-row ownership at bootstrap, then the completed
P-row ownership before the first body's v3 prefix read. Additional future blocks
may be allocated without invalidating that logical-prefix witness; remapping an
already observed row or replacing its pool is refused before KV copies. This
binding is necessary for interpreting unequal composed/later same-rank hashes
as observed storage mutation rather than a different owner/map.

## Fail-closed diagnostic propagation

Today `ensure_drafter_context` and `try_eager_drafter_prefill` return unit, and
the former swallows `prefill_drafter` errors. A source diagnostic must not use
that fallback to continue serving after a failed observation or writer.

Change those two internal functions to Result<()> and their existing callers
to propagate with `?`: four Model prefill wrappers in trait_impl/mod.rs,
model/impl_b3.rs, and the batched speculative call in trait_impl/speculative.rs.
With the diagnostic off or source-ineligible, retain the exact existing
swallow/log/continue behavior for ordinary primer/carry failures and return Ok.
With selected source evidence armed, propagate both source-copy and actual
primer errors without swallowing, and reject a noncommitting/short primer.
This is a narrow diagnostic behavior change, not general primer error policy.

Require eager execution for selected short-source profiles before any source
copy. `ATLAS_NO_MTP_EAGER_DRAFTER` must not produce a seemingly complete first
trace lacking the selected source. Refuse the selected diagnostic profile
explicitly; larger source-ineligible requests retain prior behavior.

The earlier mixed-forward stream mismatch is outside this slice: fused mixed
decode shadows the stream, but GLM is excluded by both comm and MLA guards.
Do not change non-GLM concurrency code as part of this diagnostic.

## Hashes and strict version4 schema

Use SHA256 over exact storage bits, with fixed ASCII domain plus LE metadata:

```text
primer_source = "atlas/glm53/mtp-source/primer/v1\0"
             || u64(P) || u32(4096) || u32(2)
             || capture rows[0,P-1) in row-major order
bootstrap_source = "atlas/glm53/mtp-source/bootstrap/v1\0"
                || u64(P-1) || u32(4096) || u32(2) || capture row[P-1]
primer_tokens = "atlas/glm53/mtp-source/primer-tokens/v1\0"
             || u64(P-1) || each actual token[1,P) as u32LE
bootstrap_token = "atlas/glm53/mtp-source/bootstrap-token/v1\0"
                || u64(P) || actual token[P] as u32LE
```

Emit four source/token fields plus three immediate-KV fields only on version4
attempt1/step0 with2<=P<=256; all seven literal None on later records and P
outside source scope. P derives
from first cache_before, whose established relation to position is unchanged.
All existing v3 KV fields retain exact eligibility1..2043 and hash domains.
Versions1/2/3 parse unchanged, with source evidence explicitly unavailable.
Strictly reject partial sets, wrong eligibility, malformed hashes, unknown/
duplicate fields, mixed request schemas and selected absent evidence.

Compare actual token hashes before assigning a source/writer interpretation.
Different token inputs are not a writer divergence. Compare source digests
independently of physical map hashes. A source-available pair with equal token
and source hashes uses immediate/composed KV to distinguish writer observations
from later mutation, not as a correctness conclusion. Old unavailable evidence
never becomes equality. Preserve all v22/v23/v24 reports after stripping only
the new source availability/digest/comparison fields.

## Owned files and staged gates

Production edits: model/trait_impl/{drafter_prefill,speculative,mod}.rs,
model/impl_b3.rs, new model/glm_mtp_prompt_trace.rs plus model module declaration,
layers/glm5_mtp/{hidden_trace,hidden_trace_kv,kv_rows,repair}.rs and new private prompt-source
child/test files. Minimal Glm5MtpProposerState field initialization only if the
phase state cannot live under HiddenTrace. No generic trait or runtime backend
changes. Existing hidden_trace.rs is442 lines; put substantial new logic in
children, keep new Rust files<=500. Analyzer and its existing tests plus this
plan are owned. Family author's MoE/kernel files remain untouched.

TDD must execute real owner-bound/model consume hooks and actual primer/repair
dispatch, not only a digest helper. Test exact P2/148/256 budgets, P1/257/1984
source-inert behavior with v3 KV unchanged, every source-copy failure position,
full first/last valid bytes and unused tails, token shifts, stale generations,
capture capacity/alias/cursor, duplicate/failed source phases, no eager fallback,
actual stream/capture guards, and first-attempt/reset behavior. Specifically
inject a diagnostic failure through the actual eager wrapper and assert its
Model prefill caller returns Err, while flag-off ordinary primer failure keeps
the prior fallback. Verify bootstrap rowP-1 separately from current rowP.
Include P17/P33/P129 primer block boundaries without future blocks, every
immediate-KV copy fault, old-row clobber during bootstrap and post-bootstrap
mutation. Compare the composed digest against a fresh canonical whole-prefix
reference when unmodified and detect injected changes between observations.

Execute actual runtime RED then focused/full shared-tree CPU GREEN and non-test
check, owned formatting/SPDX/file caps; label concurrent family WIP accurately.
Strict analyzer tests and complete legacy reanalysis, independent exact-hash
review, then root commit/native gates. Persistent receipts belong under
`atlas-campaigns/20260908/hidden-trace-prompt-source/`. No native performance or
numerical improvement claim follows from CPU tests or these observations.

## CPU implementation receipts (2026-09-08)

Persistent directory:
`/home/abc/storage/models/atlas-campaigns/20260908/hidden-trace-prompt-source/`.
The final full suite exercises this slice atop the committed family baseline
(current root tree20e9d224), with the stream-regression and next MoE reader files
still unreferenced and excluded from compilation. It is a worktree receipt,
not an assertion that a later commit was already tested in isolation.

- `interval-red.log`→`interval-green.log`: actual bounded reader stub fails,
  then8/8 including old v3 and exact-boundary no-future-block tests pass.
- `phase-red.log`→`phase-green.log`: production source phase assertion fails,
  then real byte source+composed-prefix observation passes.
- `eager-behavioral-red.log`: restoring swallowed primer errors loses the
  original error and fails the real eager caller assertion. `eager-red.log`
  is an earlier compile failure, not behavioral RED.
- `writer-hook-red.log`: removing the actual primer source hook fails the
  real writer→repair test with missing scratch; restored hooks are GREEN.
- `public-wrapper-red.log`: dropping propagation at actual Model::prefill_chunk
  returns Ok for selected injected failure and fails the public caller test.
- `final-full-suite.log`:912/912 PASS33.93s with all hooks restored, including
  the public caller, canonical sentinel writer outputs, all99 added copy fault
  positions, model provenance/capacity faults, map/pool binding, and state reset.
  Repair-environment tests launch isolated child processes; no process-global
  environment mutation races with the ordinary default-environment suite.
- `final-lib-check.log`: non-test no-default-feature library check PASS8.58s.
- `analyzer-final.log`:24/24 PASS. `legacy-immediate-reanalysis.log` compares
  all three complete v22/v23/v24 reports after stripping exactly17 new fields;
  all existing evidence/classifications remain identical.
- `final-format.log`, `final-diff-check.log`, `final-spdx.log`: scoped style,
  whitespace and license checks pass. All new Rust children remain<=500 lines.
- `final-clippy.log`: blocked by four existing too-many-arguments errors in
  runtime no-CUDA stubs before checking this model slice; no lint suppression
  or unrelated runtime change was introduced, and this is not a Clippy pass.

No serving activation, GPU allocation, target/MTP numerical change, CUDA build,
native result, timing claim, or stream-ordering fix is included in this slice.
