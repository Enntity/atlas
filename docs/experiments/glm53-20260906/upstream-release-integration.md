# Upstream integration for the September 10 release candidate

Deadline: 2026-09-10 02:42 UTC. Upstream `origin/main` was fetched at
`6c5f17dab9c27ee2396aef1ac2501a17b201c715`. The pre-merge fork checkpoint
is `3ff217b1`, also retained as `release/checkpoint-20260909-c8`.
The already qualified serving images from `a069efc3` remain the rollback.

## Integration boundaries

Preserved upstream Qwen4/PLE/QSA, n-gram and DSpark functionality alongside
the GLM paired/independent lanes. In particular:

- New DSpark allocation identities are created inside the GLM retained-slot
  initialization error boundary. GLM's own private-owner generations remain
  authoritative for its paired protocol.
- Upstream graph vetoes and prestaging coexist with GLM eager and sparse
  policies. Selected paired failures cannot enter legacy graph cleanup/retry.
- Actual attention allocation now returns `AttnLayerState`. GLM temporal MLA
  admits only its stateless `qsa=None` form (or legacy empty state), rejecting
  foreign/QSA state and low-rank HC paths before compute.
- New arena allocations are included in the total-allocation oracle. The
  upstream permanent 64 KiB MoE zero slab is measured after weight loading;
  it must not be charged twice to the later inference reserve.

The first full CPU model run exposed the attention-state mismatch and the
old MoE allocation oracle: 1,337 passed, two failed, 14 ignored. These are
host-side integration checks, not model quality or GPU numerical evidence.
Both failing focused cases were subsequently corrected and passed.
The final full model rerun passes **1,339**, fails zero, with 14 ignored.
Runtime buffer allocation checks also pass (18 cases).

The full server run with `NO_COLOR` unset passes 2,427, with 12 ignored and
one pre-existing, time-sensitive TUI selection-render failure. That unchanged
test passes when rerun alone with one test thread. An earlier run inherited
`NO_COLOR=1` and failed three color-style assertions instead; no inference or
safety checks failed. This is not a claim that all workspace/CI gates passed.
Formatting passes; pre-existing over-cap Rust files and lint debt remain.

## Distributed greedy tie correctness

Upstream scalar/batched argmax now chooses the highest vocabulary index for
equal maxima. The fork's shard-value kernel and rank merge still used older
tie rules. Merely changing the rank merge to `>=` would mishandle two invalid
shards, whose canonical fallback is token zero.

The fix reuses the upstream comparator in both shard reduction stages and
chooses the later shard on a tied **valid** maximum. BF16 cannot represent
the FP32 `-1e30` sentinel exactly (`0xf149f2ca`), so tied sentinel values remain
the all-invalid fallback. Scalar/batch/FP32 bodies and the 8-byte shard record
are unchanged.

Evidence retained under the external campaign directory
`atlas-campaigns/20260909/glm-native-controller`:

- Actual extracted Rust merge: three failing tie tests before, all five pass
  after. Source contract: missing two shard comparator calls before, passes
  after. These checks invoke the production merge helper/source.
- `scripts/dev/glm_argmax_shard_parity.cu`: actual scalar, batched and two-shard
  CUDA reductions over 24 deterministic cases, 619,728 device bytes, output
  canaries and unchanged input bytes. Ten checks failed before; all pass
  after. Includes within/across-stride ties, shard-boundary ties, NaNs,
  infinities, invalid shards and values below the canonical floor.
- The C++ host merge in that microtest is explicitly a mirror, not a linked
  Rust invocation; the production Rust merge is separately exercised above.
- Corrected-kernel memcheck: zero errors and zero leaked bytes. Both native
  passing processes exit zero, `OOMKilled=false`; head swap use remains zero.

Microtest SHA256:
`cc53ed6e69bcdb67906a226884db8ac66f99791e23c1eee49ee8fbb649143c42`.
Corrected kernel SHA256:
`95fe82ec52e9da0533d3b56bb1add660dddd63f7251ac7a143b2febc76f4ae8a`.
Native microtest binary SHA256:
`e75df999a1be1562cea65b8f5bdffe7974f9fbc7e0b03ef47b0c8a7fe7d0f003`.

These are integration/microkernel receipts, not a qualification of merged
C8 serving, full-context quality, or the reference-performance goal. Those
require the subsequent immutable merged-image runs in the release plan.
