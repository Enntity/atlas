# R-d — mHC split thresholds at verify sizes (design note)

Status: design only. No code changes. Branch `followup/rd-mhc-split-thresholds`
off `perf/decode-vllm-parity-20261015` @ `794e9400`.
Model geometry used throughout: GLM-5.3-Flash EXL3, `hidden = 4096`,
`hc_mult = 4`, `hc_sinkhorn_iters = 20`, `layers_total = 45`
(`kernels/gb10/glm-5.3-flash-exl3/MODEL.toml:8,13,36`), 2 mHC sites per layer
(`layers/qwen3_attention/types_weights.rs:235-236`) → 90 pre sites per forward.

Files read: `crates/spark-model/src/layers/glm5_exl3_mhc.rs`,
`kernels/gb10/common/exl3_mhc.cu` (byte-identical to
`kernels/gb10/glm-5.3-flash-exl3/exl3/exl3_mhc.cu`),
`kernels/gb10/common/exl3_mhc_tf32.cuh`,
`crates/spark-model/src/layers/{glm5_kda/hc.rs,qwen3_attention/exl3_mla.rs}`,
`tests/exl3_validation/microbench/mhc_finish/{bench,rbench}.cu`,
`docs/design/decode-vllm-parity-ledger-20261015.md`.

## TL;DR

1. The premise "the split paths engage for `rows < 8`" is **half wrong**. The
   *unrounded* split path engages for **every** `rows` in 1..16 (`precision ==
   UnroundedPost`, `glm5_exl3_mhc.rs:224`); only the *rounded* (layer-0
   attention) split path is gated on `rows < 8` (`:219-222`). At `r = 8` the
   89 non-layer-0 sites still run the split pair — but with `splits = 4`,
   which **no in-tree evidence has ever exercised**.
2. `r = 8` does not "fall back" for the unrounded sites: it falls back for one
   site (layer-0 attention) and changes split count for the other 89.
3. The split path is **per-row independent of `r` for a fixed split count**
   (verified structurally: all three kernels index only their own row). It is
   **not** `r`-independent in aggregate, because the split count is derived
   from `rows` in two places (`glm5_exl3_mhc.rs:240` and
   `exl3_mhc.cu:110`).
4. **Recommendation (see §6):** keep the `rows < 8 ? 8 : 4` derivation as the
   served default; widen only the *rounded* gate to `rows <= 16` (provably
   bit-identical, default-on); make the unrounded `S = 8` extension to rows
   8..15 a **default-off flag** on top of an explicit-`splits` kernel argument
   (SSOT), and gate F4 verify on the exact gate. Do **not** ship a bare
   threshold edit — it silently changes the default (flag-off) numerics.

## 1. The exact thresholds

| # | Threshold | Where | Value at `hidden = 4096` |
|---|---|---|---|
| T1 | residual precision | `glm5_exl3_mhc.rs:76-80` | `rows <= 16 → UnroundedPost`, else `Rounded` |
| T2 | site precision override | `glm5_kda/hc.rs:62-66`, `qwen3_attention/exl3_mla.rs:47-49` | layer-0 attention is always `Rounded` |
| T3 | split-kernel availability | `glm5_exl3_mhc.rs:146-153` | only if `ATLAS_EXL3_SPLIT_MHC=1` |
| T4 | rounded split path | `glm5_exl3_mhc.rs:219-222` | `Rounded && rows < 8 && hidden <= 4096 && 25*rs*4 <= hidden*2` |
| T5 | unrounded split count | `glm5_exl3_mhc.rs:240` | `rows < 8 && hidden <= 4096 ? 8 : 4` |
| T6 | unrounded split count (CUDA) | `exl3_mhc.cu:110` | `gridDim.x < 8 && hidden <= 4096 ? 8 : 4` |
| T7 | rounded split count | `glm5_exl3_mhc.rs:41-57` | `(48 / ceil(rows/64)).clamp(1, width/64/4)`; `rows` 1..64 → **48** |

`rounded_splits` for the reference profile: rows 1..64 → 48; 65..128 → 24;
129..192 → 16; 193..256 → 12; …; ≥ 2049 → 1.

**Why `r = 8` changes behaviour.** With `k = 7` drafts the verify forward is
`r = 8` rows, which is the first row count that fails every `rows < 8` test:

* Layer-0 attention (T2 forces `Rounded`): T4 fails → the fused
  `exl3_mhc_pre_norm` TF32 branch runs, not `exl3_mhc_pre_partials_rounded`.
* All other 89 sites: T1 gives `UnroundedPost`, so the split pair *does* run,
  but T5/T6 both yield `splits = 4` instead of `8`.

So at `r = 8` the split pair runs on 89 sites with a split count that decode
(`r = 1`) never uses, and the rounded split pair runs nowhere.

## 2. Bit-exactness contract of the split path (what it actually claims)

The split kernels are only a *scheduling* change: `exl3_mhc_pre_partials`
computes the 25 quantities (24 mixes + squared norm) × `splits` that
`exl3_mhc_pre_norm` computes inline, one (quantity, split, row) per block, and
`exl3_mhc_pre_norm_split` folds them. Bit-identity holds **conditionally on a
fixed split count**, and requires all of the following — these are the
constraints any `r = 8..16` change must preserve:

**R1 — one split-count SSOT.** The host selector (`:240`) and the CUDA
derivation (`exl3_mhc.cu:110`) must agree exactly. They are duplicated
derivations of the same quantity (an SSOT violation); editing only one makes
`ATLAS_EXL3_SPLIT_MHC=1` and the default path disagree at the bit level for
rows 8..15. `exl3_mhc_pre_norm` derives `splits` from `gridDim.x`, so *any*
change to the split count is a change to the **default flag-off kernel too**.

**R2 — feature partition.** `split_width = hidden / splits` must divide
exactly, and the per-thread loop must cover
`[split*split_width, (split+1)*split_width)` once. Fused inline loop
(`exl3_mhc.cu:117,140`: `feature = group*THREADS + lane`, stride
`groups*THREADS = 256`, `groups = 1`) and `exl3_mhc_pre_partials`
(`:253`: `feature = lane`, stride 256) are the same loop. A non-divisible
`hidden` would silently skip the tail features (host fail-fast at `:241-243`
is the only guard — keep it).

**R3 — per-thread accumulation order.** Streams 0..3 innermost inside a
feature, features ascending; `__fmaf_rn` throughout. Neither kernel may be
re-ordered, vectorized or reassociated.

**R4 — one reduction shape.** `projection_reduce` (`exl3_mhc.cu:26-45`): warp
butterfly offsets 16→1, then lane 0 serially sums the 8 warp leaders in
ascending warp index. Applied exactly once per (quantity, split). Partials
(`:266`) and the fused inline path (`:131,153`) share the helper.

**R5 — cross-split fold order.** Ascending from split 0, starting at `0.0f`,
one `__fadd_rn` per split, once per quantity — never a tree, pairwise or
parallel fold. Fused: `:131,153` `total = __fadd_rn(total, reduce(...))`;
finish: `:337-339` `total = __fadd_rn(total, partials[lane*splits + split])`.
Same operands, same order ⇒ identical bits.

**R6 — two different partial layouts, one logical read.** `pre_partials`
writes `partials[quantity*splits + split]` (`:269`);
`pre_partials_rounded` writes `output[split + quantity*splits]` (staged via
`split_partial(..., split_stride=1, quantity_stride=splits)`,
`exl3_mhc_tf32.cuh:11-52,61`). The finish kernel always reads
`partials[quantity*splits + split]` (`:339`). Do not "unify" the layouts.

**R7 — staging capacity and aliasing.** Partials live in the row's `normed`
output region (`:268`, `:336`), overwritten by the finish kernel only after
the fold (`:335-341` precedes the `normed` write at `:429-437`). Capacity:
`25*S` floats ≤ `hidden/2` floats, i.e. `25*S*4 <= hidden*2` — the host check
at `:242`. For `hidden = 4096` the binding limit is not the staging but the
finish kernel's compile-time `GROUPS = 4096/(4*256)` and
`collapsed_values[4096]` (`:322,382`), which is why `hidden <= 4096` is
hard-coded in T4/T5. A launch between the two kernels that writes `normed`
breaks the pair; both launches must stay in order on one stream.

**R8 — 16-lane Sinkhorn ≡ serial Sinkhorn.** `exl3_mhc_pre_norm_split:355-381`
runs lanes 32..47 (one matrix element each, `__syncwarp` only). The fused
serial version is lane 0 (`:159-201`). Identical by construction: row max
`fmaxf(fmaxf(r0,r1),fmaxf(r2,r3))`, `expf(v - max)`,
`__fadd_rn(__fdiv_rn(v, sum4(row)), hc_eps)`; then per iteration
(`iters = 20`) a row normalize `v / (sum4(row) + hc_eps)` and a column
normalize `v / (sum4(column) + hc_eps)`, with `sum4(a,b,c,d) =
__fadd_rn(__fadd_rn(a,c), __fadd_rn(b,d))`. Any pairwise reordering, or moving
`hc_eps` inside the division, breaks the bits. Note the fused kernel keeps the
whole matrix in registers while the split kernel round-trips through
`__shared__ float sink[16]` — that is safe only because of the
`__syncwarp` pairs at `:360,364,366,370,372,375,377`.

**R9 — no cross-row coupling anywhere.** `pre_partials` grid `[25, S, rows]`
puts the row in `z`; `pre_partials_rounded` grid `[S, rows]` puts it in `y`;
`pre_norm_split` grid `[rows]` puts it in `x`. Every load/store is
`row`-relative. No atomics, no two-pass reductions, no block-wide scalars
shared across rows. **This is what makes per-row results identical for
`r = 8` split-8 and `r = 16` split-8.**

**R10 — guards.** `exl3_mhc_pre_norm_split:332` returns silently if
`rounded_splits == 0 || > 64 || > width/64`. The host always supplies a valid
value, but a zero would no-op the entire site (no `comb`, no `normed`
update) without error — a fail-loud nit, not a bits issue.

## 3. Which split counts are `r`-independent, which are not

Per-row numerics depend only on `(hidden, splits, precision)` — never on `r`
(R9). Therefore:

| Split count | `r`-independent? | Notes |
|---|---|---|
| `S = 4` (rows 8..16, any `hidden`) | **yes** | constant over its whole range |
| `S = 8` (rows 1..7, `hidden <= 4096`) | **yes** | constant over its whole range |
| `S = 8` if T5 were widened to rows ≤ 16 | **yes** for rows 1..16 | `r=8` split-8 ≡ `r=16` split-8, bit for bit |
| rounded `rs` (48 / ceil(rows/64)) | **no** across 64-row blocks | constant only inside a 64-row block (48 for rows 1..64) |
| `rows < 8 ? 8 : 4` (the *selector*) | **no** | the sole source of `r`-dependence in the unrounded family |

So: **the split path itself is fully `r`-independent for any fixed `S`.** The
`r`-dependence lives entirely in the selector. Concretely, today
`r = 7` (S=8) and `r = 8` (S=4) produce different last bits in the 24 mixes,
because `split_width` is 512 vs 1024 and each thread accumulates 2 vs 4
features per split before the same `projection_reduce`. That difference then
flows through `sigmoid`/`expf`/Sinkhorn into `pre`, `post` and `comb`, and
into the residual writeback and `normed`.

## 4. Consequence for F4 (k = 7, r = 8)

Serial equivalence in F4 requires the verify forward at `r = 8` to be
numerically identical to `r = 1` decode. **It is not, today, and the split
flag is not the cause** — T5/T6 already select `S = 4` at `r = 8` in the
default flag-off path, so the divergence exists with
`ATLAS_EXL3_SPLIT_MHC` unset. Enabling the split path does not widen it
(the split pair mirrors the fused `S`), but it does put an *unevidenced*
configuration (`S = 4`, `rows > 1`) on the critical path for the first time.

Evidence status (this is the real gap, not the threshold):

* `bench.cu:22,42-46` compares fused-vs-split at `splits = 8`, **one row**
  (`dim3(25, splits, 1)`, `<<<1, 256>>>`).
* `rbench.cu:28-34` compares the rounded fused-vs-split at `rs = 48`,
  **one row** (`dim3(rs, 1)`).
* In-model exact records (`docs/design/decode-vllm-parity-ledger-20261015.md:37,53`)
  are decode/long workloads, i.e. `rows = 1`.

⇒ **No in-tree bit-exactness evidence exists for `rows > 1`, for
`splits = 4`, or for any verify shape.** The F4 gate would be the first.

## 5. Options considered

**Option A — threshold change only** (`:240` `rows < 8` → `rows <= 16`, plus
the same edit at `exl3_mhc.cu:110`). Rejected as a standalone change: it
alters the numerics of the *default* kernel for rows 8..15 (R1), the
`ATLAS_EXL3_SPLIT_MHC` flag cannot isolate it, and the exact gate
(`ab_full.sh` `output_sha256` vs `8081778d`/`80b1eb5f`, per the ledger) must be
re-run with no faster path to qualify. It is also the *only* change that makes
rows 1..16 mutually bit-identical, so it is still required — but as an
explicit, flagged, re-qualified change, not a threshold edit.

**Option B — default-off flag.** Keep the served default exactly as-is
(`rows < 8 ? 8 : 4`), and add a default-off switch that selects `S = 8` for
rows ≤ 16. Because `exl3_mhc_pre_norm` derives `S` from `gridDim.x`, a flag
requires first making `S` an **explicit kernel argument** with the host as
SSOT (removes the duplicated derivation, kills R1's hazard class, and makes
the dispatch table testable on the mock backend). Flag off ⇒ today's bits,
bit for bit. Flag on ⇒ `S = 8` over rows 1..16, `r`-independent over the
whole verify range.

**Option C — do nothing, keep the split path off for verify.** `r = 8` runs
the fused kernels everywhere. Zero bit risk, zero new qualification. Cost is
launch count and the loss of the split path's 25×S-way parallelism at verify
sizes (§7). Acceptable as the *initial* F4 configuration.

## 6. Recommendation

1. **Default-on, bit-neutral:** widen T4 (`glm5_exl3_mhc.rs:220`) from
   `rows < 8` to `rows <= 16` (or `rows < 17`). Provably bit-identical: for
   rows 1..64 `rounded_splits` is 48 either way (T7), and
   `exl3_mhc_pre_partials_rounded` + fold reproduces the fused rounded branch
   operand-for-operand (R4, R5, R6). This closes the coverage hole for the
   layer-0 attention site at verify sizes. Note the fit term
   `25*rs*4 <= hidden*2` (4800 ≤ 8192 for `hidden = 4096`) stays satisfied.
2. **Default-off, numerics-changing:** make `S = 8` for rows ≤ 16 reachable
   only behind a default-off flag, implemented as (a) `splits` becomes an
   explicit argument of `exl3_mhc_pre_norm` (host SSOT, delete the
   `gridDim.x` derivation), (b) host selector becomes
   `flag_on && rows <= 16 && hidden <= 4096 ? 8 : 4`, keeping today's value
   when the flag is off. PCND: the split count must be explicit config, not
   inferred from the launch grid.
3. **Do not ship the unrounded threshold edit by itself** (Option A) — it is a
   numerics change to the served default path with no isolation.
4. **F4 verify stays on the fused kernels (Option C) until both gates pass:**
   a rows-split microbench sweep (below) proving fused ≡ split at
   `splits ∈ {4, 8}` × `rows ∈ {1, 2, 7, 8, 9, 15, 16, 17}`, and an in-model
   exact run with `k = 7` verify showing unchanged `output_sha256`. Only then
   does the flag turn on for the verify path.
5. **Hard requirement restated:** bit-exactness here means "each configuration
   is bit-identical to the fused kernel at the same `(precision, rows,
   hidden, splits)`". It does **not** mean rows 1..7 and rows 8..16 agree with
   each other — that is what Option B is for, and it must be re-qualified
   rather than assumed.

## 7. Fallback cost, in kernel launches (no timing estimates)

Per mHC pre site, per forward:

| Path | Launches per site | Grid / block |
|---|---|---|
| Fused `exl3_mhc_pre_norm` (fallback) | **1** | `[rows]` × 256 |
| Unrounded split | **2** | partials `[25, S, rows]` × 256 (200 blocks at S=4/r=8; 400 at S=8) + finish `[rows]` × 256 |
| Rounded split | **2** | partials_rounded `[48, rows]` × 32 (384 blocks) + finish `[rows]` × 256 |

Per forward at `r = 8`, 45 layers × 2 sites = 90 pre sites:

| Configuration | Pre-site launches |
|---|---|
| `ATLAS_EXL3_SPLIT_MHC` unset (fallback) | **90** |
| Today's T4 (`rows < 8`), flag on | **179** (89 split sites × 2 + 1 fused) |
| T4 widened to `rows <= 16`, flag on | **180** (90 × 2) |

Marginal cost of the split path: **+1 launch per pre site**, i.e. +90 per
verify step at `r = 8` (90 → 180). Both launches are same-stream, in-order and
graph-capturable (F1), so there is no added host synchronization; the launch
that is "saved" by falling back is the launch that parallelizes the 25×S
projection reductions across blocks instead of serializing them inside one
block per row. `post` (1 launch) and `contract` (1 launch) are unchanged by
any of this.

**Can the split path be made `r`-independent?** Yes — it already is, for a
fixed `S` (R9). What cannot be done without a numerics change is making the
*current selector* `r`-independent over 1..16, because `S = 8` for rows 8..15
differs from today's `S = 4` in the last bits. The fallback is the fused
kernel at 1 launch per site.

## 8. Residual risks / open questions

* **Owed, GPU-only:** fused ≡ split at `rows > 1` and at `splits = 4`
  (bit-exact by construction R1–R9, unverified on hardware). Unblock:
  extend `tests/exl3_validation/microbench/mhc_finish/{bench,rbench}.cu` to a
  rows/sweeps loop and run under the bench lock on a Spark:
  `flock /tmp/atlas-bench.lock ~/.local/bin/spark1 'cd ~/atlas-dev/src/tests/exl3_validation/microbench/mhc_finish && docker exec atlas-dev-builder nvcc … -o /target/mb_mhc bench.cu && ~/atlas-dev/target/mb_mhc'`
  (per `microbench/README.md`), then confirm in-model with
  `tests/exl3_validation/ab_full.sh` `output_sha256` unchanged.
* **Owed, in-model:** `k = 7` verify exactness. Unblock: F4 verify shape on a
  Spark with `ATLAS_EXL3_SPLIT_MHC=1` vs the decode reference.
* **`exl3_mhc.cu:110` vs `glm5_exl3_mhc.rs:240`** duplicate the split-count
  derivation; the SSOT fix (Option B step a) is a prerequisite for any
  threshold change but is a kernel-signature change outside this note.
* **`exl3_mhc.cu:332`** silent no-op guard on `rounded_splits == 0` (PCND).
* **`hidden > 4096`** can never use `S = 8` (finish kernel `GROUPS` and
  `collapsed_values[4096]`); any threshold wording must keep the
  `hidden <= 4096` guard, and no other reference model may silently inherit a
  different `S`.
* **No runnable CPU check pins the dispatch table** today
  (`glm5_exl3_mhc.rs` tests never set `ATLAS_EXL3_SPLIT_MHC` and assert the
  non-split kernel set). Follow-up: a mock-backend test pinning
  `(rows, precision) → (kernel, grid, splits)` for rows 1..17, so a threshold
  edit fails CI rather than the exact gate.
* Partials staged in `normed` (R7): any future writer launched between the two
  split kernels, or a graph-capture ordering change, breaks the pair silently.
  Worth an assertion or comment at the implementation site.
