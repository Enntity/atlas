# R-a — `ATLAS_DECODE_ARENA_FILL=skip` safety audit

**Item:** todo.md `R-a` — arena-skip safety audit
**Branch:** `followup/ra-arena-skip-audit` (off `perf/decode-vllm-parity-20261015` @ `794e9400`)
**Date:** 2026-10-16
**Mode:** READ-ONLY source audit. No GPU work, no source edits, no server runs.
**Report author:** worker subagent (single writer thread).

Reference state at audit time:
`crates/spark-model/src/model/trait_impl/decode_a.rs:101-108` (fill gate) and
`decode_a.rs:499-505` (`decode_arena_fill`). The perf stack takes
`ATLAS_DECODE_ARENA_FILL=skip` (`tests/exl3_validation/serve_perf.py:41`, the
`EXACT` list) together with `--enable-prefix-caching` and
`ATLAS_MLA_PREFIX_SKIP=1` (`serve_perf.py:71-73`).

---

## 0. What the flag actually gates (blast radius, established from code)

`decode_arena_fill()` maps the env var to `None` (skip) / `Some(0xFF)` (poison) /
`Some(0)` (default), and the single call site is:

```rust
// crates/spark-model/src/model/trait_impl/decode_a.rs:104-108
if self.config.kv_lora_rank > 0 && self.config.o_lora_rank == 0 {
    match decode_arena_fill() {
        Some(0) => self.buffers.zero_all(self.gpu.as_ref(), stream)?,
        Some(value) => self.buffers.fill_all(self.gpu.as_ref(), value, stream)?,
        None => {}
    }
}
```

Consequences that bound the whole audit:

1. **The gate is MLA-absorbed-only.** `kv_lora_rank > 0 && o_lora_rank == 0`
   excludes DeepSeek-V4-Flash (`o_lora_rank > 0`, direct-KV path) and every
   non-MLA model. For GLM-5.3-Flash EXL3 (`kv_lora_rank = 512`,
   `qk_rope_head_dim = 0`, `o_lora_rank = 0`) the gate **is** taken.
2. **It is `TransformerModel::decode_dispatch` only.** Entry points that reach it:
   - `decode_a.rs:44` — single-sequence decode;
   - `decode_a2.rs:87` — `decode_batch_dispatch` with `n == 1`;
   - `decode_a2.rs:177` — the per-sequence fallback loop (`mla_perseq_fallback`
     or `hc_perseq`, `decode_a2.rs:132`/`155`);
   - `verify_a.rs:41-45` — decode-verify with `k == 1`;
   - `prefill_a.rs:49-52` — prefill with `n <= 1`.
3. **Nothing else changes.** Prefill still calls `zero_all` unconditionally
   (`prefill_a.rs:75`, `prefill_b.rs:200`, `prefill_c.rs:104-106`; the
   `zero_prefill_essentials` fast path is a *separate*, flag-free reduction at
   `accessors.rs:315-329`). The batched-decode body
   (`decode_batch_compute_main`, `decode_a2.rs:212`) has **never** cleared the
   arena — its own invariant is "every scratch buffer is overwritten by each
   sub-call" (`decode_b.rs:101-103`).
4. **The clear is pre-graph.** The fill runs at `decode_a.rs:104-108`, before the
   graph-cache lookup/replay at `decode_a.rs:328-340`, so capture, replay and
   eager steps all see the same fill state. Graph-captured kernels are therefore
   unaffected by which fill is chosen.

`fill_all` (`crates/spark-runtime/src/buffers/accessors.rs:337-363`) covers exactly
18 buffers: `hidden_states`, `residual`, `norm_output`, `qkv_output`,
`attn_output`, `gate_logits`, `moe_output`, `ssm_qkvz`, `ssm_ba`,
`ssm_deinterleaved`, `ssm_gates`, `ssm_conv_out_f32`, `splitk_workspace`,
`expert_gate_out`, `expert_up_out`, `expert_down_out`, `logits`, `scratch`.

**Explicitly outside the fill's blast radius** (never cleared, therefore never a
skip/zero differential): `hc_streams`, `hc_post`, `hc_comb`, `gate_logits_f32`,
`moe_router_in_f32`, `ffn_act_q8`, `ffn_act_a`, `ffn_act_scale`, `fp8_act`,
`fp8_act_scale`, `q2_dequant_scratch`, `q2_act_q8`, `gdn_fla_scratch`,
`ssd_scratch`, `o_latent`, `lora_*`, `qsa_select_scratch`, `glm_latent_*`,
`hc_lowrank_scratch`, `norm_unit_w`, `token_ids`. `token_ids` is re-uploaded
every step before graph replay (`decode_a.rs:158-160`).

> **Hazard class being audited:** a reader of one of the 18 buffers that assumes
> freshly-written-or-zeroed memory. Under `skip` the assumption becomes
> "freshly-written **or the previous step's / previous sequence's / previous
> request's content**". Note the two failure modes differ in visibility: `=0`
> makes an unwritten lane read as `0.0` (usually benign, occasionally the argmax
> winner), `=poison` makes it read as NaN (always visible), `=skip` makes it read
> as *plausible stale numbers* (visible only as an output-hash divergence).

---

## 1. Reader inventory

`Arena in fill_all` × `single-seq decode` × `batched multi-seq decode` × `verify` ×
`prefill`. "Reads" = `buffers.<name>()` obtained for a kernel `__restrict__ const`
input or an in-place operand.

| buffer | single-seq `decode()` | batched `decode_multi_seq` | verify (K>1) | prefill |
|---|---|---|---|---|
| `hidden_states` | W `embed_ctx` (`decode_a.rs:115`), R/W per layer | W per-row (`decode_a2.rs:442-447`, padding zeroed 447-452) | W `verify_a.rs:59-63` | W `prefill_*` |
| `residual` | **not read** on GLM (HC ignores it) — see P4-1 | W `rms_norm_residual` (non-HC) / ignored (HC) | W | W |
| `norm_output` | W rms_norm → R as MLA `normed` → W MLA `v_extracted` | W rms_norm (all rows) → R QKV → W `ffn_out` staging (token IDs) | W | W |
| `qkv_output` | W MLA O-proj out; R GLM indexer `selected` | W KDA 3 planes + β/f/g; R/W MLA K/V cache entry | W | W |
| `attn_output` | W paged-decode out | W KDA `core_out`; W MTP shared-expert out | W | W |
| `gate_logits` | not read on the native EXL3 FFN (private scratch) | MoE router scores + sort metadata | MoE router | MoE router |
| `moe_output` | W EXL3 FFN out | W FFN out; R `hc_post` | W | W |
| `ssm_qkvz` | W KDA `packed`; W MLA indexer keys/gates **bytes 0..512 only**; W MLA `v_extracted` | W KDA `packed`; W MTP embeds | W | W; FLA workspace = `expert_gate_out` |
| `ssm_ba` | W MLA `q_latent` → W `k_rope_single` | W MLA `q_latent`/`k_rope_single` | W | W |
| `ssm_deinterleaved` | W MLA `q_full`; W indexer `query`; W KDA `g1`/`g2` | same, per batch row | W | W |
| `ssm_gates` | W GLM native index weights | — | — | — |
| `ssm_conv_out_f32` | W MLA `q_rope_direct`; W KDA `convolved` | W KDA `convolved` | W | W |
| `splitk_workspace` | W attention split-K partials | — | — | — |
| `expert_gate_out` | W MLA `kv_latent`; W split-K scratch; W FLA workspace | same | W | W |
| `expert_up_out` | W MLA `q_absorbed_buf` | W MLA `q_absorbed_buf` | W | W |
| `expert_down_out` | W GLM indexer `logits` | — | W | W |
| `logits` | W LM head; R argmax / D2H | W LM head per row; **MTP shared-expert gate scratch** | W per row; **GLM vocab-split leaves half stale (P3-1)** | W last-token row (+ co-dispatch row) |
| `scratch` | W decode metadata @+32768; W argmax out @0 | W MoE metadata @0; W MTP metadata @`MTP_META_OFFSET` | same | W prefill metadata |

Primary reader sites (file:line, single-seq decode):

- MLA absorbed chain — `layers/qwen3_attention/decode/attention_forward_mla.rs:90`
  (`q_latent = ssm_ba`), `:129` (`q_full = ssm_deinterleaved`), `:158`
  (`q_absorbed_buf = expert_up_out`), `:199` (`q_rope_direct = ssm_conv_out_f32`),
  `:240` (`kv_latent = expert_gate_out`), `:279` (`k_rope_single = ssm_ba`),
  `:384` (`attn_out = attn_output`), `:402` (fp8 index buffer = `expert_gate_out`);
  `decode/attention_forward_mla_output.rs:44` (`v_extracted = norm_output`), `:85`
  (`o_out = qkv_output`).
- GLM indexer — `layers/qwen3_attention/decode/glm_index.rs:48-49`
  (`keys`/`gates` = `ssm_qkvz`), `:205` (`query = ssm_deinterleaved`), `:218`
  (`weights = ssm_gates`), `:256` (`logits = expert_down_out`), `:279`
  (`selected = qkv_output`).
- KDA chain — `layers/glm5_kda/forward_attention.rs:82` (`normed = norm_output`),
  `:119` (`projected = qkv_output`), `:131-133` (β/f_a/g_a planes),
  `forward_recurrent.rs:22-25` (`packed = ssm_qkvz`, `convolved = ssm_conv_out_f32`,
  `core_out = attn_output`), `forward_aux.rs:181-182` (`g1`/`g2` in
  `ssm_deinterleaved`).
- Native EXL3 FFN — `layers/glm5_exl3_ffn/mod.rs:107-111` writes `moe_output`
  for `rows` rows; its routed workspace is a **private 11-buffer pool**
  (`layers/glm5_exl3_ffn/scratch.rs:54-82`), not the arena.
- mHC highway — `layers/glm5_exl3_mhc.rs` (`hc_streams`/`hc_post`/`hc_comb`),
  outside the fill.

---

## 2. Path 1 — multi-sequence decode

### 1a GLM-5.3 KDA layer, per-sequence loop

`layers/glm5_kda.rs` `decode_multi_seq` (else branch) loops
`self.decode(hidden.offset(i*h*2), residual.offset(i*h*2), states[i], …)`.
Every iteration re-runs the whole single-row chain over the **same row-0 scratch**
(`norm_output`, `qkv_output`, `ssm_*`, `attn_output`, `moe_output`). Because each
iteration fully rewrites every buffer it reads and the work is stream-ordered, the
reuse is sound (`trait_impl/multi_seq/mla.rs:33-38` states the same invariant for
the attention sibling). **None found.**

### 1b GLM-5.3 KDA layer, batched rows

`layers/glm5_kda/multi_seq.rs` — all projections write `m` rows
(`:127` norm_output, `:143` qkv_output planes, `:214`/`:237`/`:242`/`:243`
ssm_deinterleaved / ssm_qkvz / ssm_conv_out_f32 / attn_output). Row bounds are
enforced before launch by `model/glm_c4.rs:242-273` (`validate_scratch_rows`)
and `layers/ops/kda_indexed.rs:149-241` (`conv_args`/`recurrent_args` validate
`rows * CHANNELS * 2`, `rows * PLANE * 2`, `rows * 32 * 2`). **None found.**

### 1c Attention, batched MLA (`ms_mla_decode`)

`trait_impl/multi_seq/mla.rs:109` loops `for i in 0..c.n`; each iteration writes
`o_out_i = o_out.offset(i*h*bf16)` (`:129`), i.e. **all `n` rows that
`ms_phase_ffn` later consumes** (`multi_seq/ffn.rs:38-39`). Per-row buffer
assignments in `multi_seq/mla_serial.rs`: `:34` `q_latent = ssm_ba`, `:69`
`q_full = ssm_deinterleaved`, `:97` `q_absorbed_buf = expert_up_out`, `:101`
`q_rope_direct = ssm_conv_out_f32`, `:143` `kv_latent = expert_gate_out`, `:182`
`k_rope_single = ssm_ba`, `:233` `k_cache_entry = qkv_output`, `:281`
`attn_out = attn_output`, `:321` `v_extracted = ssm_qkvz` (`norm_output` is
deliberately *not* used here because it still holds the `n` per-sequence `normed`
inputs — documented at `:318-320`). **None found.**

### 1d Fused mixed prefill+decode

`model/trait_impl/decode_b.rs:71` bails to `decode_batch` for every MLA model, so
GLM-5.3 never takes the fused body. Out of the flag's blast radius.
`decode_b2.rs` "parks prefill logits at row `padded_n`" and uses `norm_output` as
transient token-ID staging (`decode_b.rs:147-148`) — both write-before-read.

### HAZARD P1-1 — the skip lands on the multi-seq hot path (exposure, no defect found)

`decode_a2.rs:155-158`:

```rust
let hc_perseq = self.config.hc_mult > 0
    && (qsa_active || std::env::var("ATLAS_HC_PERSEQ_DECODE").as_deref() == Ok("1"));
```

GLM-5.3 has `hc_mult = 4`, so the per-sequence `decode()` loop is the **default**
whenever `qsa_active` holds, i.e. when any `seq_len >= index_topk +
index_compress_ratio - 1`. For `glm5_next` `index_compress_ratio` is `0` (only the
qwen4_exp parser ever sets it — `config/parsers/qwen4_exp.rs:237`), giving a bound
of `2047`. The native EXL3 GLM indexer's dense-exact threshold is also `2047`
(`layers/qwen3_attention/glm_index_budget.rs:36-46`,
`dense_threshold = topk - 1` with `topk = 2048`, asserted at `:74-80`). The two
bounds therefore agree (one token of conservative slack at `seq_len == 2047`), so
the batched path never runs with an active indexer on this model.

**Consequence:** for `seq_len >= 2048` on GLM-5.3, "multi-sequence decode" *is*
the per-seq loop, and that loop calls the arena-gated `decode()`. Any reader in
§1 that was not row-local would become a cross-sequence leak. All readers checked
are row-local and fully rewritten — **no defect found** — but this is the exact
configuration the existing poison run (`docs/design/decode-vllm-parity-ledger-20261015.md:60-63`,
hash `80b1eb5f`, single-sequence only) did **not** cover.

**Minimal fix direction (if the poison matrix disagrees):** keep
`ATLAS_DECODE_ARENA_FILL=skip` off the per-seq fallback — i.e. resolve the fill
value once at model construction into a `bool` and have the fallback loop force
`Some(0)` for its iterations — rather than narrowing the gate.

### HAZARD P1-2 — candidacy note (informational)

`decode_a2.rs:147-154` `qsa_active` uses `index_compress_ratio` (a Qwen3.8 QSA
field) as the GLM per-seq trigger. It works today only because GLM leaves that
field `0`. A second, GLM-native predicate would be
`!(budget.dense_is_exact(seq_len))` (`glm_index_budget.rs:60-62`), which is the
condition that actually gates the indexer (`glm_index_budget.rs:60-62`). **Not a
defect; do not "fix" without a
product decision** (it changes when concurrent decode silently switches lanes).

---

## 3. Path 2 — prefix-cache hit / KDA-snapshot restore / `mla_prefix_skip` / radix

### 2a `ATLAS_MLA_PREFIX_SKIP=1`

`model/impl_b3_accessors.rs:145-152`: with the flag set,
`mla_prefill_needs_full_recompute()` returns `false`, so a prefix-cache hit makes
prefill process only the uncached suffix. The prefix-cache reuse itself is at
`trait_impl/prefill_a.rs:78-104`; the Marconi SSM-snapshot skip is at
`prefill_a.rs:145-190`.

### 2b KDA snapshot restore (Marconi)

`model/ssm_snapshot.rs:408-450` `restore()` D2D-copies h/conv state into the live
pool slot. Aux state is restored by `apply_aux_states` (`prefill_a.rs:163-165`).
The exact full-prompt leaf hit fixup is at
`trait_impl/prefill_b/finalize_last.rs:181-220`: re-`restore` the pristine SSM
state, then `restore_hidden(snap_id, normed, …)` (`:191-193`) overwrites
`norm_output` with the
stashed last-token post-norm hidden before `lm_head(normed)` (`:212-220`). Both
are **writes that precede the read**. **None found.**

### HAZARD P2-1 — arena row-count transition (no reader found; test required)

With a prefix hit, prefill writes arena rows for only the *suffix* length, and the
Marconi path can process a single row. Under `=skip`, decode step 1 then reads
arena lanes that hold the **previous request's** prefill data instead of zeros.
I enumerated every single-seq `decode()` reader of the 18 buffers (§1): every one
is row-local and written in the same call. **No defect found**, but this is the
second configuration the existing poison run did not cover (it was single-sequence
cold).

### HAZARD P2-2 — `ssm_qkvz` partial-write region on MLA layers (tightest coupling; none found)

`glm_index.rs:48-49` writes only bytes `[0, 512)` of `ssm_qkvz` (k key + gate, both
BF16) on a GLM MLA layer. The rest of `ssm_qkvz` is written on that layer only by
`ms_mla_decode_one`'s `v_extracted` (`mla_serial.rs:321`, `nq * mla_v_dim` BF16
elements) — and on the **single-seq** MLA path by nothing at all
(`attention_forward_mla_output.rs:44` uses `norm_output` for `v_extracted`).

The K3/DFlash O-staging plan reserves exactly that prefix:
`layers/qwen3_attention/glm_k3_mla_o.rs:14` `PREFIX = 2 * 128 * 2` and `:145`
`retained = (whole.0 + PREFIX, whole.1)`, with an explicit liveness check at
`:55-78`. Ordering inside `ms_mla_decode_one` is
indexer-maintenance/attention (`:281-305`) → `v_extracted` (`:321`), so the prefix
is consumed before it is overwritten.

**Verdict: none found**, but this is the one buffer where the *only* per-step write
is 512 bytes and the readers are the most numerous — it is where a poison test is
most likely to bite first.

### HAZARD P2-3 — radix / disk-id reuse (`reuse_prefix_match_disk_ids`)

`block_mgmt.rs` `reuse_prefix_match_disk_ids` and `apply_evicted_blocks` operate on
`seq.disk_block_ids` (host vectors) and on the KV pool, not on the arena. Not in
the fill's blast radius. **None found.**

---

## 4. Path 3 — MTP / DFlash verify

### 4a Reachability on the target model

`factory/build.rs:262-266` refuses the native EXL3 build for any speculative
configuration:

```rust
anyhow::ensure!(
    !use_speculative && !self_speculative && dflash_args.is_none() && lora_args.is_none(),
    "EXL3 validation requires speculative decoding and LoRA disabled"
);
```

`tests/exl3_validation/serve_perf.py` sets `default_num_drafts = 0`. Therefore
**MTP/DFlash verify is unreachable for GLM-5.3-Flash EXL3** and cannot be combined
with `ATLAS_DECODE_ARENA_FILL=skip` on the target. The rows below still apply to
non-EXL3 models, where the flag *is* reachable.

### 4b Generic verify paths (audited, non-EXL3)

`trait_impl/verify_a.rs` … `verify_e2.rs` write `K` rows of `hidden_states` /
`residual` and read them back per row; logits are written per row at
`verify_a.rs:247` / `verify_b.rs:422` (`logits().offset(t*vocab*bf16)`); the FFN
verify arms (`forward_k2/k3/k4/k5`, `forward_batched`, `forward_independent`,
`glm5_kda/multi_seq.rs:346-372`) write `moe_output` before consuming it. **None
found** for read-before-write.

### HAZARD P3-1 — GLM vocab-split verify head deliberately leaves half the logits row stale

`model/glm_vocab_split.rs:8-9` (module docstring):

> "The other half of the logits buffer is left stale, so this serves only
> argmax-consuming verify."

`glm_vocab_split.rs:22-23` (`PAIRS_OFFSET = 16384` inside the arena `scratch`
metadata region) and `:89` (`glm_split_head_argmax`). This is the **only
documented intentional stale-lane consumer** in the tree.

- Under `=0` the peer half reads as zeros.
- Under `=skip` it reads as the previous step's logits — *finite, plausible, and
  indistinguishable from a real logit vector*.
- Argmax is safe today because each rank scans only its own shard. Any future
  sampler, logit-bias, min-tokens ban, or logprob consumer on that path is a live
  correctness hazard, and `=skip` makes it silent rather than loud.

**Minimal fix direction:** have the split head write `-INFINITY` (BF16 `0xFF80`)
into the peer half before the peer exchange — one `memset_async` of
`vocab/2 * 2` bytes per row-batch, paid only on the verify path — or add an
assertion that the split head is only reachable from an argmax consumer.

### HAZARD P3-2 — `expert_gate_out` split-K scratch vs. `kv_latent` ordering (verified safe, fragile)

`decode/attention_forward_mla_sparse.rs:78` borrows `expert_gate_out` as the
split-K partials/LSE/merge scratch, and `attention_forward_mla.rs:402` reuses it as
the fp8-g128 causal-index buffer. Both happen **after** `kv_latent`
(`attention_forward_mla.rs:240`) has been consumed by the cache-assemble at
`:336-370`. For GLM-5.3 the TC decode head bails immediately
(`attention_forward_mla_sparse.rs:49`: `ctx.config.qk_rope_head_dim != 0` returns
`false`), so neither borrow is taken and the scalar
`paged_decode_attn_bf16` runs instead. **None found**; the ordering constraint is
undocumented and worth a comment, not a code change.

### HAZARD P3-3 — `q_latent` reuse before the index-Q projection (pre-existing, not arena-fill)

`attention_forward_mla.rs:388-393` calls
`glm_index_decode_update_and_select(normed, q_latent, …)` where `q_latent` is
`ssm_ba` — but `ssm_ba` was already reused as `k_rope_single` at `:279` whenever
`mla_rope > 0`. Dead on GLM-5.3 (`qk_rope_head_dim = 0` ⇒ the reuse never
happens, and the multi-seq path is already correct because it hoists the index-Q
projection into retained scratch — `glm_index.rs:159-174`,
`mla_serial.rs:28-33`).

**Minimal fix direction:** single-seq should take the same `precomputed_query`
seam the multi-seq path uses, instead of re-deriving the index-Q from `ssm_ba`.

### HAZARD P3-4 — MTP shared expert reuses `logits` as gate scratch (verified safe)

`layers/mtp_head/moe_forward.rs:184-185` (`shared_gate_scratch = logits()`,
`shared_up_scratch = ssm_qkvz()`) and `:203/244` (`shared_out = attn_output()`).
All writes precede reads within the layer; the verify LM head writes `logits`
afterwards (`forward_batch.rs:505`). **None found.**

---

## 5. Path 4 — the per-sequence MLA fallback

`model/trait_impl/decode_a2.rs:132-205`. Two triggers:
`mla_perseq_fallback` (escape hatch, `ATLAS_MLA_PERSEQ_FALLBACK=1`) and
`hc_perseq` (default for `hc_mult > 0` + QSA-active — see P1-1). The loop:

1. stages `n * vocab` bytes of logits to the host (`:159-172`),
2. calls `self.decode(tokens[i], seqs[i], stream)` per sequence (`:177`) — the
   arena-gated entry point,
3. `copy_d2h_on_stream(logits, &mut staged[i..])` (`:184-191`) before the next
   iteration,
4. uploads the assembled `[n, vocab]` batch back (`:191-193`).

Every arena reader enumerated in §1 is written inside the same `decode()` call, so
no cross-sequence leak exists today. **None found.** Two notes:

- The D2H staging is now the *only* thing protecting the assembled batch: the
  comment at `:178-181` assumes the next `decode()`'s `zero_all` wipes the shared
  buffer, which `=skip` no longer does. The ordering is still correct (the copy is
  synchronising on `copy_stream`, the stream the kernels actually use), but any
  future change that defers or reorders the staging would now silently read the
  *last* sequence's logits instead of detecting the wipe. Cheap hardening: assert
  `logits` row 0 equals the just-computed row before the loop advances, or stage
  into `logits.offset(i * vocab * bf16)` directly and skip the host round-trip.
- `decode_a2.rs:163-176` documents the exact failure signature of a broken
  staging ("a clean two-way row swap at every joint C=2 step, `#753 item B`",
  `decode_a2.rs:170-172`) —
  reuse that as the poison-matrix pass/fail signal.

### Finding P4-1 — `residual` is dead weight in GLM-5.3 decode

`layers/glm5_kda.rs:334/350/368` (`decode`, `prefill`, `decode_batched`) and
`layers/qwen3_attention/trait_impl/decode_inner.rs:445` (`decode_inner_hc`) all
take `_residual` and ignore it: HC models carry the multi-stream residual in
`hc_streams` instead. So for GLM-5.3 the per-step `fill_all` memsets an arena
buffer that no decode kernel reads (`sizes.rs:409-410`, `m * h * 2` per row —
~8 MB at `m = 2048`).

**This is a follow-up, not part of R-a:** dropping `residual` from the fill (and
eventually from the GLM decode signature) would remove ~8 MB/step of the very
bandwidth the `skip` flag was introduced to reclaim. Must be gated on an explicit
config predicate, not a model-type string.

---

## 6. Positive findings — no hazard, with evidence

These are the invariants that make `=skip` safe today. Each is a place where a
plausible partial-write was checked and found to be fully initialised.

| # | Site | Evidence |
|---|---|---|
| N1 | GLM indexer score padding | `kernels/gb10/common/glm_indexer_score.cuh:31-34` writes `-INFINITY` for every `pool_id >= pool_count` and returns before any read. `row8` variant: `kernels/gb10/common/glm_indexer.cu:318-321`. |
| N2 | GLM indexer top-k output row | `kernels/gb10/common/glm_indexer.cu:366-370` initialises **all** `output_width` entries to `-1` before selection; the tail is written at `:424-429`. No lane can hold a stale index. |
| N3 | MLA absorbed-Q gap | `layers/ops/kv_cache.rs:582-606` `mla_batched_gemv` takes `output_stride`; the GLM NoPE call passes `mla_cache_dim == kv_lora == n_out` (`attention_forward_mla.rs:158-176`), so there is no unwritten `[kv_lora, mla_cache_dim)` gap. |
| N4 | Batched-decode padding rows | `model/trait_impl/decode_a2.rs:447-452` explicitly `memset`s `hidden[n..dispatch_n)` to 0; padding rows also get dummy block tables and the dedicated `ssm_pool.dummy_slot()` (`:554-575`). |
| N5 | Indexed KDA row bounds | `layers/ops/kda_indexed.rs:149-241` (`conv_args`/`recurrent_args`) validates `rows * CHANNELS * 2`, `rows * PLANE * 2`, `rows * 32 * 2` before launch, and `KdaBuffer::validate` (`:96-116`) rejects any span that does not cover them. |
| N6 | EXL3 vocab head writes the whole row | `model/glm_exl3_head.rs:117-120` sets `active_vocab = config.vocab_size` with `active_vocab <= physical`; `write_bf16` (`:261-283`) converts exactly `rows * active_vocab` elements. The server reads exactly that width (`spark-server/src/scheduler/decode_logits_seq.rs:33`, `decode_logits_step.rs:243`). |
| N7 | `scratch` layout is partitioned | `spark-runtime/src/buffers/sizes.rs:186-246` documents and sizes the two overlays: fixed metadata `[0, 32768)`, decode metadata at `+32768`, MTP at `MTP_META_OFFSET`, GLM verify pairs at `+16384`, argmax out at `+0`. Every writer/reader pair in the GLM decode path stays inside its own region. |
| N8 | K3/DFlash O staging collision-checked | `layers/qwen3_attention/glm_k3_mla_o.rs:123-159` builds an explicit `live` span list and refuses the plan if the retained rows alias any live buffer, including the indexer prefix (`PREFIX`). |
| N9 | Fill precedes graph replay | `decode_a.rs:104-108` runs before `decode_a.rs:328-340`; capture and replay see identical fill state. |
| N10 | Native EXL3 MoE uses private scratch | `layers/glm5_exl3_ffn/scratch.rs:54-82` — 11 pool-allocated buffers, `MoeScratchBorrow::acquire()` leases them per call. Arena `gate_logits`/`expert_*`/`scratch` are untouched by the routed experts on the target model. |

---

## 7. Adjacent observations (out of R-a's scope; unverified; no edits made)

These surfaced during the audit. They are **not** arena-fill differentials and
**not** blocking. Recorded for the operator.

- **A1 — `moe/forward_atomic_c4.rs:75-92` accumulator may overlap the decode
  metadata region.** `accum_bytes = num_tokens * h * 4` (65 536 B at `n=4, h=4096`)
  at `scratch + accum_off`, i.e. it spans past the `+32768` fixed-metadata
  boundary that `decode_a.rs:130-159` owns. Reachable only for `MoeLayer` models
  (never GLM-5.3 EXL3, which uses the private-scratch native FFN — N10). Could not
  verify which model×path combinations reach it or whether the metadata is
  re-uploaded per layer on those paths. **Not asserted as a bug.** If the
  poison matrix on a non-EXL3 model shows corrupted attention, check this first.
- **A2 — `scratch_is_sufficient` is test-only.**
  `layers/glm5_kda/recurrent.rs:34-46` computes the `RegisterResident` borrow
  requirement (`expert_down_out >= 2 * plane_bytes`) but only `recurrent.rs:156`
  (the unit test) calls it; no production site validates the borrow. The
  regresident path is prefill-only (`KdaRecurrentPath::select`, `:15-19`), so
  decode is unaffected. Fix direction: call it from `Glm5Exl3KdaResources` load.
- **A3 — `layers/moe/gate_up_btile_arena.rs:103` and the C4/independent lanes
  assert `gate_logits` capacity** (`glm_c4.rs:256` "sort metadata",
  `rows * 96 + 1156`). Arena `gate_logits` is in the fill, so a stale
  `gate_logits` expert-index vector is precisely the documented CUDA-700
  hazard class (`accessors.rs:318-321`). On GLM-5.3 EXL3 no MoE reader touches
  `gate_logits`, so this is a cross-model note only.
- **A4 — `decode_meta` rows.** `sizes.rs:242` `logits_tokens =
  m.min(96.max(decode_meta.rows() + 1))`. The arena `logits` row cap is derived,
  and every LM-head writer I checked is row-indexed against the same
  `padded_n`/`rows`. No defect; flagging because it is the one buffer whose
  *capacity* is a moving derived value.

---

## 8. Proposed poison-test matrix (for the operator, under the bench lock)

**Pass criterion for every row:** `output_sha256` (and per-token logits where the
harness records them) must be **identical** to the eager, `ATLAS_DECODE_ARENA_FILL`
unset baseline for the same workload. A NaN, a CUDA 700/700-illegal-address, or a
hash change localises a read of arena memory that the step did not write.

**Prerequisite (one-line change, not owned by this audit):** parameterise the fill
in `tests/exl3_validation/serve_perf.py` — add `--arena-fill {zero,skip,poison}`
and emit the chosen `ATLAS_DECODE_ARENA_FILL` instead of the hard-coded
`ATLAS_DECODE_ARENA_FILL=skip` at `serve_perf.py:41`. Until then, start the server
with the env overridden in the container.

**Cheaper first gate (no poison, already implemented):**
`ATLAS_SSM_SAVE_DUMP=1` drives `debug_buffer_checksum`
(`spark-runtime/src/buffers/accessors.rs:224-300`) at decode step 0
(`decode_a.rs:69-88`) plus `ssm_pool.debug_state_checksum` and
`kv_cache.debug_kv_per_block`. Run it twice for the same workload and diff the
`ATLAS_BUF_CKSUM[decode_step0_pre]` lines: a buffer whose fingerprint differs
between two identical runs holds leftover-from-prior-occupant bytes — the exact
stale-scratch signature, without needing NaN.

| # | Workload (driver) | Flags | Path exercised | Expected failure signature if a hazard exists |
|---|---|---|---|---|
| M1 | single-seq long decode (`ab_prose_long`) — **already passed** (`80b1eb5f`) | `=poison` | single-seq `decode()`, all layers | NaN logits / hash change — anchor, must stay green |
| M2 | single-seq short decode, prompt < 2047 tokens | `=poison` | single-seq, dense MLA (indexer inert) | Hash change only in the last layers ⇒ `norm_output`/`qkv_output` partial write |
| M3 | single-seq decode crossing `seq_len = 2048` | `=poison` | indexer engages at the first full-attention layer | Divergence that *starts* at a specific step ⇒ indexer buffer (`ssm_gates` / `ssm_deinterleaved` query / `expert_down_out` logits / `qkv_output` selected). NaN in `ssm_gates` ⇒ P2-2 region |
| M4 | C=2 concurrent decode, both prompts < 2047 | `=poison` | batched `decode_multi_seq` (KDA batched + attention MS) | Row swap / two-sequence contamination ⇒ per-row scratch reuse |
| M5 | C=4 concurrent decode, prompts > 2048 | `=poison` | **per-seq fallback** (`hc_perseq`, P1-1) + indexer | "Clean two-way row swap at every joint C=2 step" (`decode_a2.rs:170-172`) ⇒ logits staging; per-sequence NaN ⇒ §1 reader |
| M6 | C=2, shared long system prompt, second turn (prefix cache warm) | `=poison` | `mla_prefix_skip=1` + Marconi restore (2a/2b) | First-token divergence only ⇒ `restore_hidden`/`norm_output`; later divergence ⇒ stale arena rows vs suffix-length prefill (P2-1) |
| M7 | Multi-turn agentic, same session hash (tail capture reuse) | `=poison` | `reserve_tail_slot` / mid-chunk tail | Divergence on turn ≥ 2 only ⇒ snapshot restore path |
| M8 | Exact full-prompt leaf hit (prompt repeated in one session) | `=poison` | `finalize_last.rs:176-220` exact-hit fixup | First token wrong, rest fine ⇒ double-advance or `restore_hidden` |
| M9 | Spill/fault-in (`bench/ssm_faultin.py`) | `=poison` | `model/ssm_fault_in.rs` + `reclaim_from_cache` | Divergence only after a spill boundary ⇒ faulted-back state vs arena |
| M10 | Chunked prefill (128-token chunks) then decode | `=poison` | `prefill_b`/`prefill_c` → decode transition | Divergence at chunk boundaries ⇒ arena row-count transition (P2-1) |
| M11 | Graph mode vs eager, same workload (M3 and M5) | `=poison` | capture/replay (N9) | Graph-only divergence ⇒ a fill-order dependence inside the captured region |
| M12 | `bench/prefix_cache_contamination.py` | `=skip` **and** `=poison` | prefix-cache hit under both fills | `=poison` clean but `=skip` divergent ⇒ **stale-but-finite read** (the class `=0` hides). This is the highest-value row: it is the only combination that separates "reads unwritten memory" from "reads the previous request's memory" |
| M13 | `bench/warm_cold_concurrent_diff.py` | `=poison` | concurrent decode, varying batch width | Batch-width-dependent divergence ⇒ a reader bounded by a fixed width rather than the live row count |
| M14 | Mixed prefill+decode co-dispatch | `=poison` | `decode_b.rs` fused body | **Expected N/A for GLM-5.3** (`is_mla_dispatch` bails to `decode_batch`, `decode_b.rs:71`). Keep the row only to prove the bail on a non-MLA model. |
| M15 | MTP/DFlash verify (qwen3.6-27B or any non-EXL3 target) | `=poison` | verify_a…e2 + `glm_vocab_split` | **Expected N/A for GLM-5.3 EXL3** (`factory/build.rs:262-266` refuses spec). On a generic model, watch specifically for P3-1 (stale peer half of the logits row) and P3-3 (`q_latent` reuse, only on rope>0 MLA + indexer) |
| M16 | `ATLAS_SSM_SAVE_DUMP=1`, run twice per M1/M5/M6 | (any fill) | `accessors.rs:224-300` fingerprints | Any `ATLAS_BUF_CKSUM[decode_step0_pre]` line that differs between the two identical runs ⇒ leftover-from-prior-occupant bytes. Cheapest signal in this table |

**Rows that are *known* to be outside the flag's blast radius** (do not spend GPU
time on them under `ARENA_FILL`): prefill-only runs (prefill still calls
`zero_all` — `prefill_a.rs:75`), any `o_lora_rank > 0` model, and any non-MLA
model.

---

## 9. What I could not verify (owed), and what would unblock it

1. **No GPU, no numerics.** Every claim above is a static source audit. The two
   properties that need hardware are: (a) that no reader of the 18 buffers reads
   an unwritten lane, and (b) that the unwritten lanes, if any, are numerically
   inert. Unblocking command (GPU host, under `flock /tmp/atlas-bench.lock`):
   ```bash
   cd <wt>   # or the operator's tree
   ATLAS_DECODE_ARENA_FILL=poison python3 tests/exl3_validation/serve_perf.py start --exact
   # then the workload driver, then compare output_sha256 against the eager reference
   ```
   with the fill parameterised per §8.
2. **Kernel-level write coverage.** Several conclusions rest on Rust launch
   arguments plus a reading of the CUDA sources (`kernels/gb10/common/glm_indexer*.cu`,
   `glm_indexer_score.cuh`). I did not read every kernel body that consumes an
   arena pointer (notably `paged_decode_attn_bf16`, `kda_pack_qkv`,
   `conv1d_update_prefill`, `kda_recurrent`, `mla_batched_gemv`,
   `exl3_moe*`, `exl3_linear_mma`). A hazard could hide in a kernel that writes
   fewer lanes than its ABI implies. The poison matrix is the right instrument;
   a manual pass over those kernels is the fallback.
3. **A1 (atomic-C4 accumulator overlap).** Cannot be resolved without knowing
   which shipped models route `MoeLayer::forward_atomic_c4_decode` and whether
   their decode metadata is re-uploaded per layer. Unblocking: `rg -n
   "forward_atomic_c4_decode" crates/spark-model/src` (already done — the only
   production caller is `layers/qwen3_ssm/trait_decode_multi_seq.rs:330`) plus a
   targeted poison run on that model family.
4. **`GLM-5.3` checkpoint `index_compress_ratio`.** I inferred `0` from
   `crates/atlas-core/src/config/tests.rs:100-135` (the GLM-5.3 config fixture
   never sets it) and from `config/parsers/qwen4_exp.rs:237` being the only
   writer. If the real checkpoint sets it to `4`, the `qsa_active` bound moves to
   `2051` and the batched path *would* run with an active indexer for
   `seq_len ∈ [2048, 2050]`. Unblocking: read `index_compress_ratio` out of the
   served `config.json`, or assert it at model load.

---

## 10. Summary

| Path | Verdict |
|---|---|
| 1 — multi-sequence decode | **none found** (1a/1b/1c/1d all write-before-read). Exposure note **P1-1**: for GLM-5.3 the per-seq loop — and therefore the arena-gated `decode()` — is the default for `seq_len >= 2048`, which is exactly the case the existing poison run did not cover. |
| 2 — prefix cache / KDA snapshot / `mla_prefix_skip` | **none found** (restore and `restore_hidden` both write before the read). **P2-1** (arena row-count transition) and **P2-2** (`ssm_qkvz` 512-byte prefix) are the tightest couplings and the first places to look if the matrix goes red. |
| 3 — MTP / DFlash verify | **Unreachable on GLM-5.3 EXL3** (`factory/build.rs:262-266`). **P3-1** is a real, documented, deliberate stale-lane consumer (`glm_vocab_split.rs:8-9`); P3-3 is a pre-existing rope>0 ordering bug that is dead on this model. |
| 4 — per-seq MLA fallback | **none found**. Hardening note: the D2H staging is now load-bearing in a new way; **P4-1** records that `residual` is memset but never read on GLM-5.3 (follow-up, not R-a). |

**Ten positive invariants (N1–N10)** were confirmed with kernel- or
bounds-check-level evidence; four adjacent observations (A1–A4) are recorded as
unverified and out of scope. The proposed matrix (§8) is 16 rows; **M12** is the
single highest-value one, because it is the only combination that distinguishes
"reads unwritten memory" from "reads the previous request's memory".
