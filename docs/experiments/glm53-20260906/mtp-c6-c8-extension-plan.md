# Concurrent MTP through eight owners: next dependency plan

2026-09-09, read-only design review against source `a069efc3`.
This follows native qualification of the three/four-owner E7 path; it does
not claim C6/C8 MTP support or authorize raising limits before implementation.

## Required implementation

1. Extend authenticated recipe, preflight, factory, private-storage and actual
   pool capacity together. Derive reserve from `PrivateStoragePlan`, target
   state and arena accounting, not a copied memory fraction. At context2044,
   indexed private cache plus hidden slab alone costs5,423,104bytes/owner;
   this is not the total additional model reserve. Preserve the4GiB floor.
2. Extend the explicit owner shapes and support every draining width3..8,
   including5/7 and noncontiguous survivors. Preserve scalar/pair fallback,
   cold-owner progress and physical identity. Extend separate wider statistics.
3. Rework the bounded metadata region before increasing owner counts. Its
   current start32768 plus stride3328 reaches52736 atC6 and59392 atC8, beyond
   the49152 fence. Prove the enlarged region disjoint from all live scratch;
   update fixed host staging and token/argmax extents from the same geometry.
4. Validate actual wider workspaces: C6 needs60 norm rows and35 mHC rows;
   C8 needs80 and45. Existing1024-prefill allocation may suffice but must be
   checked against real buffers. Expand KDA's fixed44-span alias scratch to
   cover88 before any slicing; retain whole-cohort validation before writers.
5. Extend stateless FFN admission toM25/M30/M35/M40. Existing generic-T shared
   kernels and compact routed gate/up plus dense M64 down are candidates,
   not yet qualified kernels for these shapes. Unique top8 routing bounds
   each expert to at most40 rows, but expanded worklists/buffers must be
   checked and rows32..39 explicitly numerically validated. Attention can
   remain exact independent K5 calls for this first extension.
6. Explicitly distinguish the widened wire format from E7's current48-word
   payload/six-word verdict. Extend fixed results, retained producer records,
   detachment/commit bookkeeping, worker registry and retirement as one
   coordinated protocol change. All accepted counts must validate before
   any detach, and all commits before any visible token or new proposal.

## Gates in dependency order

- Native bounded M25..M40 arithmetic comparison against qualified smaller
  cohorts, varied EP masks/routing and memory checking; not a serving claim.
- Actual eight-slot model/protocol continuation and drain checks, including
  malformed last-owner refusal before mutation and normal paired retirement.
- Committed native image, supervised C1/C2/C3/C4/C6/C8 coherence, real tool
  calls and needle retrieval, then matched warmed timing and fresh-process
  repetition. Report full-wall and decode-window rates, TTFT, per-rank minimum
  available memory, observed swap and independently inspected clean exits.

Do not substitute this plan for finishing the current native C3/C4 A/B.
Do not combine context expansion with initial C6/C8 qualification. Fresh
large-context checks remain separately required after semantic-index and
memory prerequisites in `mtp-c3-c4-next-plan.md`.
