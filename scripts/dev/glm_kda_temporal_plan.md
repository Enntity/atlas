# Frozen scalar KDA temporal regression gate

This gate is independent of the frozen indexed harness. It protects existing
scalar exports before extracting shared arithmetic into production helpers.
Only new files in `scripts/dev` are created at this stage; no GPU runs by agents.

1. Define the temporal oracle harness against two renamed frozen reference
   exports before creating the reference snapshot.
2. Snapshot exactly the existing `causal_conv1d_update_prefill` and
   `kda_recurrent_bf16` functions into `glm_kda_legacy_reference.cuh`, changing
   only their exported names. Verify source identity after undoing that rename.
   The snapshot must NEVER follow future helper extraction changes.
3. Compare frozen eager with current eager and captured/replayed current scalar
   kernels for temporal tokens1/2/3/5/17, heads32/D128, conv channels12288/width4.
   Temporal tokens are successive updates to ONE history, not independent rows.
4. Use standard, zero-history, strong-state/gate, and exact-zero Q/K profiles;
   zero Q/K also zeroes convolution weights/history/bias for those channels.
   Include optional convolution bias and padded input/output strides.
5. Compare every FP32 H/conv element and every BF16 convolution/recurrent output.
   Poison unused rows and stride padding; check all outer allocation canaries,
   read-only inputs, live finite values, and zero-Q/K preconditions.
6. Bound total device allocation below16MiB (expected below10MiB). Grids: scalar
   conv48/block256; QKV gather ceil(tokens*12288/256)/block256; recurrence32/
   block128, shared0. No communication, model weights, or performance claims.
   Current 17 guarded allocations total7,266,944 bytes. Graph runtime overhead
   is additional; only one three-node graph exists at a time.
7. Root first compiles/runs with production `--fmad=false` and memcheck, then
   the default FMA build to protect other common-kernel consumers. After any
   shared extraction, rerun these gates against this unchanged snapshot.

Each exact-width graph is captured before input initialization and replayed
with two different sets of inputs/state, comparing each to the eager frozen
reference. No allocations or metadata uploads are captured.
