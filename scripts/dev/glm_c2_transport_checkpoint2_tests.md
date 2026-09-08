# B1 checkpoint 2 actual transport test split

2026-09-08, root-approved checkpoint 2 implementation after exact-tip gates for
checkpoint 1 `335c9956` and checked selection `67640368`: model 1051 PASS and
server 2359 PASS. Root explicitly released model source/Cargo for this slice.
Factory, scheduler selection and admission remain OFF. Cargo handoffs to the
independent server-only T1 author retain explicit compiled-model source freezes.

Initial actual transport RED is recorded in
`glm-c2-predispatch/checkpoint2-red.log`: all three tests compiled and failed at
the intended existing boundaries (disabled head E1, disabled head F5, and the
paired worker's legacy C1 E1 guard). No production implementation was present.

## Fixture and positive controls

Use actual `TransformerModel` + `Glm5MtpHead::new_paired`, real Gate 1/2 owner
states and the existing numerical-boundary recorder. Add a bounded test-only
command recorder/replayer: rank 0 captures words from the real command buffer
at actual broadcasts, rank 1 replays those exact words through actual
`Model::ep_worker_step`. Distinguish command broadcast from the private body's
existing all-gather; preserve deterministic vocabulary handling. No network or
NCCL agreement is inferred from replay.

New children should separate command-fixture boilerplate, positive transport
and payload validation, and transport-failure tests, each below 500 lines.
Any main-fixture constructor extension must build a real legacy head through
`Glm5MtpHead::new` for legacy traces, not relabel a paired head as C1. Existing
paired fixture behavior and event ordinals remain unchanged by default.

- Actual first E1 for both owners, opposite producer order and distinct prompt
  tails: head capability returns four drafts; actual worker receives 8 words,
  consumes its own retained tail/bonus and reaches the same private cursor/KV.
- Actual capability F5 emits the existing preamble/width/five tokens only after
  shared preflight. Append the actual selected acceptance result through the
  existing head broadcast, then replay the whole transaction on the worker.
- All25 acceptance pairs and continued unequal histories: actual head/worker
  record, trim, commit, then transported E1. Compare complete canonical and
  speculative private K/V/slab state to an actual direct-call Gate 2 control
  (same seed and inputs), not a second production math implementation.
- Deliberately choose a valid non-raw seed; assert actual EH input uses the
  selected packet seed and the next real K5 accepts the resulting issued IDs.
- Retire/reuse a real owner, then retain and replay the old generation/attempt
  packet. The new owner refuses it before private compute; no lease is forged.
- Actual legacy C1 head and worker preserve the four-word E1 payload, global
  hidden-save copy and command/event order. Legacy F5 width/token/acceptance
  transport stays unchanged. Generic legacy max-batch refusal remains tested.

## Genuine RED gates

First run the positive actual capability E1/F5 tests while checkpoint 1's
transport methods still return `paired command transport is not enabled`.
The current selected worker E1 also refuses at the C1 guard; retain that runtime
RED separately from later payload-mutation negatives. Only then implement the
approved selected eight-word format and the actual F5 transport owner.

Predictable prewire failures (wrong token owner, invalid profile/position,
grammar/width, active producer, exhausted following target budget, stale
generation) must leave the rank-0 command recorder empty and preserve an
otherwise healthy real operation. An optional capability cannot turn invalid
selected state into legacy fallback.

Mutate version, either generation half, either attempt half, position and fixed
width in the actual E1 packet. Invalid seed is vocabulary-bound; a different
*valid* seed is not falsely called a sampler violation. The actual worker's
owner/phase/position validation must reject malformed identities before private
EH/body/KV work. Raw receive H2D/D2H/synchronization is transport, not forbidden
numerical work, and must be counted honestly.

## First-header terminal behavior

Capture controls for the first header's H2D, first broadcast, remaining header,
payload transfer and actual selected execution. Inject failures at the fresh
control-derived event/communicator ordinal; assert the actual event identity,
not only a generic error string. Include a failure before any Verification
exists, which existing owner quarantine alone cannot make session-terminal.

The new transport failure operation validates actual backend identity and
irreversibly marks the paired Pool terminal, including both slots. This blocks
later selected commands and allocation even after a successful synchronization;
there is no cleanup-based revival or fallback. Failed selected E1/F5 worker
payload/read/parse/execute paths also mark the session terminal. Existing
pre-dispatch worker slot/preamble failure handling remains a B2 process-fatal
boundary when no valid request is known; do not invent a safe local retry.

This metadata policy is not native containment. B2's no-unwind fatal guard must
be armed before the first header attempt, with an external exact-peer supervisor.
No native fault injection, backend-Drop safety or graceful recovery is claimed.
