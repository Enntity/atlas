# Next performance chunk: one layer traversal across three/four MTP owners

2026-09-09 source audit; proposed implementation, not a serving capability.
The selected C4 residency path serializes two E6 pairs. Its throughput cannot
be inferred from the standalone M15/M20 FFN win. The substantive next change
is one bounded transaction retaining all selected owners through a single
layer-major traversal, with independent attention/state and accepted counts.

## Implementation sequence

1. Introduce a checked owner-batch workspace and FFN interface alongside the
   literal existing pair modes in `layer/glm_pair_verify.rs` and
   `layers/moe/forward_pair_{verify,validate,shared}.rs`. Let N=5*owners.
   Normalized working rows occupy0..N and saved rows N..2N; M20 therefore
   requires40 rows, not the existing20. Working mHC rows remain0..5; saved
   owner rows occupy5..5+N. Hidden/attention/MoE output buffers need N rows.
   Derive every byte span, router tile count and worklist bound from checked
   N. Preserve existing mode semantics and explicit opt-in selection.
2. Extend actual target traversal in `model/glm_c2_pair_verify.rs`,
   `layers/glm5_kda/paired_verify.rs` and
   `qwen3_attention/trait_impl/multi_seq/pair.rs`: run each owner's unchanged
   K5 attention, save its normalized/mHC state, execute one packed FFN, then
   restore and apply each owner's mHC. Keep independent K5 final normalization
   and vocabulary projection. Validate all metadata, logits and cross-owner
   alias spans before issue. No host/device allocations inside the transaction.
3. Add a bounded producer carrying an explicit live owner count and separate
   packed ordinal/physical slot identity. Current `paired.rs` implements only
   Single and Pair producers. Validate every verdict before detaching any
   owner, retain the producer until all selected commits complete, and retain
   all ownership terminally on any issued failure.
4. Preserve the existing fixed26-word E6 packet. Design a separate bounded
   wider wire operation, with explicit3/4 count, canonical distinct physical
   slots, validated inactive fields and a whole-message worker preflight before
   any writer. The proposed48-word payload/six-word verdict still needs its
   own protocol/ownership proof; it is not an existing command.
5. Connect wider scheduler selection last, before any issuance. Gather all
   selections by packed ordinal, complete all commits, then emit or issue E1.
   Cancellation cannot remove an owner mid-transaction. Keep pair and scalar
   drain paths. Qualify mixed acceptance, unselected-owner preservation and
   failure retention before native activation.

The native harness already exercises the generic router, compact gate/up,
dense M64 down projection, shared T-row path and scalar mHC at15/20 rows.
Reuse those implementations through checked production interfaces; do not
claim the approximately2x arithmetic result as an end-to-end speedup.

Implementation update: `bc3637fd` provides the shared layer/FFN workspace and
`77a58abb` adds the distinct producer, actual model traversal/local verdict and
fixed E7 codec. Host composed checks pass; see `owner-batch-model-integration.md`.
Next work is explicit cold selection plus live E7 exchange/worker dispatch and
scheduler selection, not another standalone arithmetic harness. The internal
model path is not yet reachable through serving.

Fresh native qualification must include current-build C1/C2 regressions,
C3/C4 warmed full-wall rates and TTFT, distinct coherence/tool/needle requests,
measured no-swap headroom and actual paired release. C6/C8 concurrent MTP and
the C1>=30 target remain required: this four-owner increment does not complete
the goal. Later capacity/row/metadata extensions need their own native bounds
and quality qualification. Keep the separate large-context follow-up in
`mtp-c3-c4-next-plan.md`; do not bypass its semantic-index prerequisites.
