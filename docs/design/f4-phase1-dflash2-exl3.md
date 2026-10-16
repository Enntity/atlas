# F4 phase 1 — DFlash2 drafter on the native GLM-5.3 EXL3 target (draft-only)

Branch `perf/f4-dflash2-exl3`. Plan: `docs/plans/f4-dflash2-exl3-tp2-plan.md`.
Status: implemented and built; **not yet run on GPUs** (no acceptance number
exists yet). Everything here is inert unless `ATLAS_GLM_EXL3_DFLASH=1`.

## What phase 1 does

With `ATLAS_GLM_EXL3_DFLASH=1` each TP rank loads the whole
`incoai/GLM-5.3-Flash-DFlash2@dc77ff1c` drafter. Serial decode is unchanged
(same graphs, same tokens): before each serial decode of token `x` at position
`P`, the observer

1. scores earlier drafts against `x` (the serial token at `P`),
2. appends the context row of position `P-1` (captured during the previous
   decode forward) to the drafter context,
3. runs one DFlash2 block forward (bonus `x` at `P` + `k` mask rows, then the
   top-16 candidate selector walk) and records the `k` drafts for positions
   `P+1..P+k`.

Drafts are never verified and never change generation. Both ranks see the
same tokens and the same post-all-reduce hiddens, hold the same drafter and
compute the same drafts — nothing is exchanged between ranks.

Code: `crates/spark-model/src/model/glm_exl3_dflash.rs` (observer, hooks),
`glm_exl3_dflash/head.rs` (replicated packed vocabulary head),
`glm_exl3_dflash/scorer.rs` (acceptance scorer, unit-tested). The drafter
itself is the existing `BlockDiffusionDraftHead` + `Dflash2Head`
(`layers/dflash_head/`), Option-B paged context path, lane 0, BF16 weights.

## Capture contract

- vLLM: `get_eagle3_aux_layers_from_config` maps DFlash `target_layer_ids`
  `[5,14,24,33,42]` to aux layers `[6,15,25,34,43]`; the GLM EAGLE3 overlay
  (`spark-glm53/overlay/patch_glm_eagle3.py`) captures after layer `idx` when
  `idx + 1` is an aux layer, i.e. **after 0-indexed layers 5, 14, 24, 33, 42**,
  as `hc_contract(layer.hc_post(...))` = mean over the 4 mHC streams.
- Atlas: after `layer.decode/prefill` of layer `i ∈ target_layer_ids`
  returns, `hc_contract_strided` (`hyper_connection` module, resolved at
  attach) means the 4 streams of `buffers.hc_streams()` (FP32 highway,
  `[rows, 4, 4096]`) into BF16 slot `s` (= index of `i` in
  `target_layer_ids`) of a `[5 × 4096]` BF16 context row. Row layout =
  `concat(layer5, layer14, layer24, layer33, layer42)`, the `fc` input order.
- True post-layer value: on the native EXL3 path every layer ends with its
  FFN `hc_post` written into `hc_streams` (`glm5_kda/forward_ffn.rs`
  `hc_post`, MLA `decode_inner.rs` `hc_post_site`). `MhcResidualPrecision::
  UnroundedPost` concerns rounding of that FP32 value, not deferral, so the
  highway read right after the layer returns is the combined post state.
- Known numeric delta vs the reference: Atlas means FP32 streams then rounds
  once to BF16; the reference materializes the post streams (BF16 in vLLM)
  and then means. Expected well inside the phase-1 cosine ≥ 0.999 gate.
- Decode: row 0 of the highway → a fixed 40 KB save buffer, inside the
  decode body (captured into the slot-keyed decode CUDA graph; stable
  pointer). Appended to the context at the start of the next decode step.
- Prefill: every layer loop (`prefill_a`, `prefill_b/forward_layers`,
  `prefill_c` two-phase) captures rows `0..proc_count` at absolute positions
  `start..start+proc_count` straight into the per-sequence accumulator,
  stamping each row's absolute position (RoPE of the drafter context K).
- Context window: the drafter's `sliding_window` (2048) sizes the
  accumulator. Past it the window slides (keeps the newest ≤ 1024 rows, then
  refills to 2048), so beyond 2048 tokens the drafter sees 1024–2048 context
  rows rather than exactly 2048 (reference: exactly the last 2048).

## Head choice

**Packed EXL3, both TP2 halves replicated per rank — not a BF16 copy.** Each
rank uploads `lm_head` shard 0 and shard 1 (`TpShardPlan(Output, r, 2)`) and
projects the drafter's `k+1` final-normed rows through each with
`Glm5Exl3Projection::launch_local_bf16` and the pinned head numerics profile
(`glm53_tp2_packed_numerics(2, 4096, 77440, 6, 77440, 0)`), then row-copies
the halves into `[k+1, 154880]` BF16 logits for `dflash2_topk`. These are
exactly the BF16 values the target head produces for the same rows (its path
is the same `launch_local_bf16` → f32 → gather → bf16), and vLLM's drafter also
uses the target's (EXL3) head. Cost: ~0.48 GB/rank of packed weights read
per draft step, vs 1.27 GB for a dequantized BF16 copy. Hook:
`BlockDiffusionDraftHead.external_head` (`DraftLogitsHead`), checked first in
`project_head`; `lm_head_shared` is NULL. Embedding: the target's BF16
`embed_tokens` (replicated on both ranks), shared.

## Flags

| env | default | meaning |
|---|---|---|
| `ATLAS_GLM_EXL3_DFLASH` | off | `1` loads the observer (EXL3 GLM, TP=2/EP=1 only) |
| `ATLAS_GLM_EXL3_DFLASH_DRAFTER` | `/root/.cache/huggingface/hub/models--incoai--GLM-5.3-Flash-DFlash2-pin/snapshots/dc77ff1c99eeb2df044ee3d4f0094eb033fee410` | drafter dir (config.json + model.safetensors) |
| `ATLAS_GLM_EXL3_DFLASH_K` | 7 | drafts per step, 1..=7 (block = k+1 rows; trained block 8) |
| `ATLAS_GLM_EXL3_DFLASH_LOG_EVERY` | 64 | periodic `DFLASH_ACCEPT` cadence in steps (0 = request end only) |
| `ATLAS_GLM_EXL3_DFLASH_TRACE` | off | `1` logs every step's drafts (`DFLASH_DRAFT`) |

Forced in code for the observer: Option B paged context, one propose lane, no
draft cap, BF16 drafter weights (no FP8/MXFP8/NVFP4 twins whatever the env).
Still honoured from the generic DFlash env: `ATLAS_DFLASH_CTX_WINDOW`
(precompute chunk / scratch rows, default 4096) and
`ATLAS_DFLASH_PROPOSE_NO_GRAPH` (eager propose fallback if the piecewise
propose graphs misbehave).

Speculation guards (`serve_phases/weights.rs`, `factory/build.rs`,
`spark-model/src/preflight.rs`): deliberately **not** opened to
`--dflash/--speculative/--self-speculative`. Those flags make the server
build verify pools / an MTP proposer and route decode into verify lanes that
do not exist for EXL3 yet, which would alter generation. Phase 1 is
env-configured; with the flag set, a CLI speculative mode fails at startup
with the env recipe. Phase 2 opens the guards together with the EXL3 verify
lane.

## How to run (draft-only)

Same command/env as the serial EXL3 bench (`tests/exl3_validation/ab_full.sh`
stack + graphs), plus the observer:

```
bash tests/exl3_validation/ab_full.sh f4-draftonly /home/colum/atlas-dev2/bin/spark \
  --env ATLAS_NO_DECODE_GRAPHS=0 --env ATLAS_NO_DECODE_GRAPHS_MULTISEQ=0 \
  --env ATLAS_DEBUG_NO_GRAPH=0 --env ATLAS_EP_GRAPHS=1 --env ATLAS_KDA_FLA_GRAPH=1 \
  --env ATLAS_EXL3_COOP_MOE_V2=1 --env ATLAS_EXL3_COOP_MOE_V3=1 \
  --env ATLAS_GLM_EXL3_DFLASH=1 --env ATLAS_GLM_EXL3_DFLASH_K=7 \
  --env ATLAS_GLM_EXL3_DFLASH_LOG_EVERY=64
```

Run at C=1 with prefix caching off (the bench descriptors already use
`--max-batch-size 1`, no `--enable-prefix-caching`). Greedy is not required:
scoring uses whatever tokens the serial decode emits. Output must stay
byte-identical to the same run without the flag (the observer only reads
target state); TPOT will include the drafter.

Startup lines: `EXL3 DFlash: drafter … head = both packed EXL3 lm_head TP2
halves replicated (0.48 GB read/step)`, `DFlash drafter: BF16 weights only`,
`EXL3 DFlash draft-only observer ACTIVE on rank R`.

## Log lines (format; values illustrative, not measured)

```
DFLASH_ACCEPT rank=0 scope=periodic req=<id> slot=0 k=7 steps=57 tau=4.105 mean_acc=3.105 \
  prefix=[0.877,0.737,...] cond=[0.877,0.840,...] raw=[0.877,0.770,...] \
  hist=[7,8,5,...] truncated=0 proposed=64
DFLASH_ACCEPT rank=0 scope=final  req=<id> slot=0 ...   (at free_sequence)
DFLASH_ACCEPT rank=0 scope=global req=- slot=0 ...      (cumulative over finished requests)
DFLASH_DRAFT rank=0 req=<id> slot=0 pos=812 bonus=1234 ctx=812 drafts=[...]   (TRACE only)
```

- `steps`: draft steps whose k predicted positions all arrived; `truncated`:
  steps cut off by the end of the request (≤ k per request, excluded).
- `hist[a]`: steps whose longest matching draft prefix is `a` (0..=k);
  `mean_acc` its mean; `tau = 1 + mean_acc` (accepted tokens per verify step
  under greedy block verification, the reference's "acceptance length").
- `prefix[j]` = P(draft j+1 accepted) = P(acc ≥ j+1); `cond[j]` =
  P(acc ≥ j+1 | acc ≥ j); `raw[j]` = P(draft j+1 equals the serial token)
  regardless of earlier drafts.
- Both ranks log (rank 1's `final` appears when the worker reuses the slot);
  the two ranks' lines must be identical — a mismatch is a determinism bug.
- `DFLASH_ACCEPT … drafting stopped early: <reason>` names any sequence the
  observer gave up on (batched decode, a missing capture, a drafter error);
  the target decode is never failed by the observer.

## Measured results (2026-10-15, both ranks identical in every run)

Evidence root `$EV=/home/colum/Workspace/atlas-exl3-validation-20261010/perf-ab-20261015`.

| run | binary | workload | k | τ | prefix acceptance (positions 1–7) |
|---|---|---|---|---|---|
| `f4-draftonly-20261015T191446Z` | 53c6fc79 | ab_prose_long (128 in / 1024 out) | 7 | 2.689 | .725 .422 .233 .131 .076 .055 .046 |
| `f4ref-fp32-20261015T194917Z` | d5697cbf | bridges essay, 200 tok, think on | 7 | **3.109** | .776 .531 .323 .193 .141 .089 .057 |
| same | | bridges, 1024 tok, think on | 7 | 2.785 | .743 .471 .265 .140 .085 .052 .029 |
| same | | bridges, 1024 tok, think off | 7 | 2.261 | .655 .337 .134 .070 .040 .014 .010 |
| `f4ref-bf16-20261015T201816Z` | cc36da51 | the same three, `CAPTURE=bf16mean` | 7 | 3.109 / 2.782 / 2.255 | within ±0.005 of fp32mean |

Output hashes with the observer on equal the flag-off serial hashes in every
run, and all 3 repeats are identical.

**τ parity.** The mindreader reference workload (`tools/dt_workload.sh`
defaults: one user message "Write a detailed 400 word essay about the
history of bridges.", temperature 0, max_tokens 200, GLM template with its
open `<think>`) is reproduced by `tests/exl3_validation/perf_workloads_f4_ref.json`.
The reference's own greedy measurements on that workload are:
- `artifacts/dt-bands-5-2026-09-27-0411/bucket-1-rep2.log`, k=5: alpha
  [.720 .456 .297 .156 .089]; gen/drafts = 18000/6649 gives τ ≈ 2.71.
- `artifacts/dt-bands-2026-09-26/bands-report.txt`, k=7: alpha
  [.7265 .444 .274 .1755 .0885 .0445 .040], τ ≈ 2.79.

Atlas draft-only on the same prompt gives τ = 3.11 (200 tokens), so it is not
low against the reference engine. The quoted 3.75–4.09
(`glm-tp2-20260925/summary.json`) came from a one-off synthetic
long-context probe whose prompts were not retained (PROGRAM.md §3.1), so it
cannot be reproduced; the earlier stack-A table reports E[L] ≈ 2.1–2.6.

**Bisection.** The capture rounding knob (`bf16mean`) moves τ by at most
0.006. Layer offsets match vLLM by construction (`+1` aux ids, capture after
layer i). The context-window slide is inactive here (≤ 1050 tokens < 2048).
The selector walk is unchanged from the reference's algorithm. None of these
is indicated.

**Drafter cost** (`ATLAS_GLM_EXL3_DFLASH_PROFILE=1`, 63 samples, rank 0):
propose 14.7 ms per step. Of that:
- context precompute (fc + 5-layer K/V for the new row): 1.8 ms;
- block forward: 12.9 ms, made up of
  - packed EXL3 lm_head on 8 rows: 2.36 ms;
  - DFlash2 top-16 + selector: 0.31 ms;
  - the 5 layers + embedding + D2H readback: ≈ 10.2 ms.

The layers read ≈ 1.9 GB of BF16 weights per step (≈ 186 GB/s), so they are
bandwidth bound. Observer TPOT is 0.068 s against 0.0535 s serial.

## Phase 2, step 1: verify parity probe (measured)

`ATLAS_GLM_EXL3_DFLASH_VERIFY_PROBE=N` runs the production K=γ verify
(`decode_verify_graphed_kgamma`) every N-th draft step. It covers the
r = k+1 = 8 rows `[bonus, d1..d7]`, using:
- MLA layers on the multi-row prefill path;
- KDA layers on the snapshotting verify path (under
  `ATLAS_KDA_FLA_REFERENCE=1` each row takes the serial FLA decode kernel,
  with h/conv snapshots after every row);
- logits from the native EXL3 head.

After the verify the probe restores the SSM checkpoint, rewinds the aux
state and `seq_len`, and lets the serial decode continue. Each verify row
whose preceding drafts matched the serial tokens is then compared with the
serial token.

Run `f4ref-probe2-20261015T230451Z` (binary f6d71dca, N=1, so every step is
probed):
- Output hashes equal the serial hashes on all 9 requests (~1,460 probes,
  each a full verify followed by a restore). The restore is non-destructive.
- Ranks are identical.
- Verify-row argmax agrees with the serial token on 3826 of 3853 checked
  rows (99.30%). Six of the 27 misses are Atlas's thinking budget forcing
  `</think>` (154842) into the serial stream, so the serial token is not an
  argmax. Excluding those, 3826/3847 = **99.45%**.
- The remaining 21 misses are argmax flips between the M=1 serial forward
  and the M=8 verify forward. The EXL3 GEMM row profiles and the MoE path
  differ by M; see the per-request `mismatches` field.
- The accepted prefix the verify would choose equals the serial-scored
  accepted prefix on 1437 of 1452 probes.
- Cost: TPOT 0.234 s with a probe every step, against 0.068 s draft-only,
  so ≈ 166 ms per eager r=8 verify plus checkpoint and restore.

Consequence: the plan's phase-2 exit criterion (100% verify-row/serial
argmax) is not met as stated; ~0.55% of rows flip. Speculative output
would match serial for the first divergence-free stretch only, as in vLLM.
Byte-identity to serial (the phase-3 exit criterion) needs either
M-invariant verify numerics or a relaxed criterion (match the verify's
own greedy chain).

## Phase 2: the speculative lane (`ATLAS_GLM_EXL3_DFLASH_SPEC=1`)

Status: works end to end on both ranks. It is **flagged and
numerics-changing**: the verify rows run at M = k+1. The exit criteria,
relaxed by the coordinator (2026-10-16), are:
- (a) self-consistency;
- (b) determinism;
- (c) exact rollback;
- (d) serial divergence, reported but not gated.

### Step

Code: `glm_exl3_dflash/spec.rs`, `scheduler/decode_step.rs`. The scheduler
hands a greedy C=1 step with no sampling processors to
`Model::exl3_dflash_spec_step(last_token, seq)`. Any other step, or a model
`None`, decodes serially; the eligibility gate is listed below.

1. The head sends the worker command `0xFFFFFFE4` with the slot and the
   pending token. Both ranks then run the step themselves; every input is
   replicated.
2. The drafter appends the previous context row(s) and proposes
   `d1..dk` after the pending token `x` at `P`.
3. One `decode_verify_graphed_kgamma` runs over `[x, d1..dk]`, using the
   EXL3 arms below. It captures every row's context.
4. `a` = longest prefix with `d_j == argmax_{j-1}`. The step emits
   `d1..d_a, argmax_a`; the bonus becomes the next pending token.
5. Commit:
   - `seq_len = P + a + 1`;
   - `commit_accepted_prefix(a+1, k+1)` installs the KDA/conv snapshot after
     row `a` and rewinds the aux state;
   - MLA/indexer rows past `seq_len` are rewritten by the next verify;
   - the drafter context gains rows `0..=a` only.

The scheduler eligibility gate requires:
- temperature 0;
- no grammar, logprobs, penalties or logit bias;
- no pending `</think>` or tool-call forcing (the thinking budget still
  ends thinking through the serial path).

EXL3 verify arms (active only with the observer):
- MLA layers take the multi-row prefill path.
- Logits come from the native vocab-sharded head.
- KDA under `ATLAS_KDA_FLA_REFERENCE=1` runs the serial FLA decode kernel
  per row, with h/conv snapshots.
- The BF16 indexer projections take `dense_gemv_bf16_batchm` inside the
  verify scope (`ATLAS_GLM_EXL3_DFLASH_INDEXER_TC=1` reverts).

Knobs:
- `ATLAS_GLM_EXL3_DFLASH_FORCE_ACCEPT=j`: cap accepted drafts.
- `ATLAS_GLM_EXL3_DFLASH_ROLLBACK_CHECK=N`: exact rollback check.
- `ATLAS_GLM_EXL3_DFLASH_TOKEN_LOG=1`: `DFLASH_TOKENS` ids at free.
- `ATLAS_GLM_EXL3_DFLASH_EXPERT_STATS=1`: unique experts per multi-row MoE
  launch (syncs; diagnostic only).
- `ATLAS_GLM_EXL3_DFLASH_DRAFTER_WEIGHTS=bf16|mxfp8|nvfp4`.

`DFLASH_SPEC` (final line) gives spec steps, emitted tokens, tokens/step
and the rollback-check counts.

### Criteria (runs under `$EV`)

- (a) Holds by construction: each emitted token is the argmax of its verify
  row. The only exception is `emit_token`'s existing stray-token handling.
- (b) Holds in every run: 3 repeats give identical hashes, and both ranks'
  `DFLASH_*` lines are identical.
- (c) **Exact.** `f4spec-rbcheck-20261016T022424Z` runs a check on every
  partial-accept step:
  - restore the pre-verify state;
  - re-verify with every rejected row replaced by different tokens;
  - recommit and compare.

  The full KDA/conv state fingerprint and the accepted rows' argmax were
  identical on all checks: 281/281, 233/233, 354/354 and the rest per
  request. Its outputs equal the unchecked run's hashes. Forced accept:
  - `FORCE_ACCEPT=3` (`f4spec-force3-*`) reproduces the unforced token
    streams exactly on all 4 workloads.
  - `FORCE_ACCEPT=0` (`f4spec-force0-*`) diverges at token 315
    (bridges_1024) and 74 (prose). The exact check proves rejected rows leave
    no trace, so this is the row-position dependence of verify numerics: a
    token computed as row 0 rather than row j. It is not a rollback leak.
- (d) Divergence from serial (`f4spec-serialtok-*` vs `f4spec-k7-*`, first
  divergent generated-token index / positional match):

  | Workload | First divergence | Positional match |
  |---|---|---|
  | bridges_200 | none (identical) | 1.000 |
  | bridges_1024 | 356 | 0.478 |
  | nothink_1024 | 10 | 0.026 |
  | ab_prose_long | 1 | 0.018 |

### Where a step goes (nsys, `f4-trace-20261016/f4-spec-mr-trace`, rank 0, k=7, multirow + V6, BF16 drafter)

Wall 144.7 ms per step, of which GPU-busy (union) is 129.9 ms; ≈ 15 ms is
eager-launch and host idle.

| Item | ms/step |
|---|---:|
| Routed MoE (multirow kernel 48.3) | 52.6 |
| EXL3 coop dense projections | ~16 |
| Indexer BF16 projections on `dense_gemm_tc`, 2-CTA grid (now GEMV, b99b5cbd) | 13.9 |
| Drafter (BF16) | 15.6 |
| KDA (FLA per-row decode 6.8) | 7.8 |
| cuBLAS BF16 (router, shared expert) | ~7.2 |
| mHC | 5.6 |
| memcpy (KDA snapshots, ~840 copies) | 4.9 |
| MLA attention (grouped_gemm_mla 4.0) | 4.1 |

The routed MoE at r=8 reads 37.3 unique experts per layer (mean over 9,870
launches, `f4spec-mr-prof-*`), ≈ 4.7× the serial step's 8. Its bandwidth
floor is ≈ 36 ms per verify at 273 GB/s, so k=7 cannot verify in
1.2–1.4× serial. Smaller k lowers the unique experts.

### Fix to the port's multirow MoE (b030781f)

`exl3_coop_moe8_multirow` poisons expert groups of more than 16 rows. A
25-token prompt's prefill chunk under `ATLAS_EXL3_COOP_MOE_MULTIROW=1`
routed more than 16 rows to one expert. The result was NaN MoE outputs and
degenerate generations (`f4spec-mr-20261016T025900Z`: 8 tokens/step, a
constant output hash). Multirow is now limited to launches of ≤ 16 rows,
which covers every verify width.

## Phase 2 status at the 2026-10-16 stop (no GPU runs after this point)

**Works, measured**
- Speculative lane end to end on both ranks.
- Determinism and identical ranks.
- Exact rollback (`f4spec-rbcheck-*`).
- Multi-row MoE bit-exact against per-slot (`f4spec-mr2-*` hashes = `f4spec-k7-*`).
- Drafter weight selector: NVFP4 is the best τ/(T_verify + T_draft).

Throughput (preliminary, single-harness, 3 repeats, decode tok/s; serial
V6 baseline on the same kernels, `f4spec-base-v6-*`, is 20.8–21.4):

| Run | Config | bridges_200 | bridges_1024 | nothink_1024 | ab_prose_long |
|---|---|---:|---:|---:|---:|
| `f4spec-k7-*` | k=7, per-slot MoE, BF16 drafter | 16.84 | 15.55 | 13.13 | 15.42 |
| `f4spec-mr2-*` | + MULTIROW + V6 | 20.16 | 18.42 | 15.58 | 18.02 |
| `f4spec-mr-nv-*` | + NVFP4 drafter | 21.90 | 19.24 | 16.47 | 18.56 |
| `f4k7-idx-20261016T051530Z` | + indexer batch-M GEMV (b99b5cbd, binary 7f2dbe4f) | **24.17** | **21.25** | 18.21 | 20.43 |

Tokens per spec step: 2.9 / 2.6 / 2.2 / 2.6.

Drafter weights (draft-only, `f4dw-*`), τ for bridges 200 / 1024 and
drafter ms per step:

| Weights | τ (200 / 1024) | ms/step |
|---|---|---:|
| BF16 | 3.109 / 2.785 | 14.7 |
| MXFP8 | 3.099 / 2.778 | 10.7 |
| NVFP4 (TC) | 3.062 / 2.759 | 8.3 |

**Built, not GPU-verified** (binary f6/7f2d… lineage, last build
7f2dbe4f; later commits are unbuilt):
- b99b5cbd: indexer batch-M GEMV in verify. Measured (`f4k7-idx-*`, row
  above); output hashes are unchanged from `f4spec-mr-nv-*`.
- fe02685e: one-launch multi-row FLA KDA verify with in-kernel snapshots.
  Unbuilt. Must be checked bit-exact against
  `ATLAS_GLM_EXL3_DFLASH_KDA_PER_ROW=1` (identical hashes and
  `DFLASH_ROLLBACK_CHECK`) before timing.
- 977b5d4d: `DFLASH_SPEC_PROFILE` per-step timing. Unbuilt.
- The k sweep (k=5, k=3) and `ATLAS_EXL3_VERIFY_PROFILES=1` were queued and
  not run.

**Next steps (in order)**
1. Build HEAD. Run `f4k7-idx`-style (below), then KDA fused vs per-row
   (hashes must match), with `ATLAS_GLM_EXL3_DFLASH_PROFILE=1` for the
   draft/verify/commit split.
2. k sweep 3/5/7 on the best config. Smaller k cuts the unique experts per
   verify (37 at k=7) and is the largest remaining lever.
3. Verify CUDA graphs keyed by r: ~15 ms/step idle at k=7 eager.
4. `ATLAS_EXL3_VERIFY_PROFILES=1` A/B (numerics-changing, lane only).
5. Integrate τ-adaptive k.

**Env recipes** (on top of `tests/exl3_validation/ab_full.sh` plus the
graphs stack: `ATLAS_NO_DECODE_GRAPHS=0 ATLAS_NO_DECODE_GRAPHS_MULTISEQ=0
ATLAS_DEBUG_NO_GRAPH=0 ATLAS_EP_GRAPHS=1 ATLAS_KDA_FLA_GRAPH=1
ATLAS_EXL3_COOP_MOE_V2=1 ATLAS_EXL3_COOP_MOE_V3=1`)

- Draft-only τ:

  ```
  ATLAS_GLM_EXL3_DFLASH=1 ATLAS_GLM_EXL3_DFLASH_K=7 ATLAS_GLM_EXL3_DFLASH_LOG_EVERY=64
  ```

- Speculative, best measured config:

  ```
  ATLAS_GLM_EXL3_DFLASH=1 ATLAS_GLM_EXL3_DFLASH_SPEC=1 ATLAS_GLM_EXL3_DFLASH_K=7
  ATLAS_EXL3_COOP_MOE_MULTIROW=1 ATLAS_EXL3_COOP_MOE_V6=1
  ATLAS_GLM_EXL3_DFLASH_DRAFTER_WEIGHTS=nvfp4 ATLAS_DFLASH_NVFP4_TC=1 ATLAS_W4A16_TC=1
  ```

- Diagnostics (one at a time, never in a timed run):
  - `ATLAS_GLM_EXL3_DFLASH_ROLLBACK_CHECK=1`
  - `ATLAS_GLM_EXL3_DFLASH_EXPERT_STATS=1`
  - `ATLAS_GLM_VERIFY_PROFILE=1`
  - `ATLAS_GLM_EXL3_DFLASH_PROFILE=1`
  - `ATLAS_GLM_EXL3_DFLASH_TOKEN_LOG=1` (cheap) with
    `scratchpad/tokcmp.py`-style comparison for divergence.
- Workloads: `tests/exl3_validation/perf_workloads_f4_ref.json`.

**Open issues**
- The k=7 MoE floor (37 unique experts at r=8, ≈ 36 ms per verify at
  bandwidth) rules out 1.2–1.4× serial verify at k=7.
- Divergence from serial starts within 1–356 tokens (labelled, not
  gated).
- `FORCE_ACCEPT=0` differs from unforced because of row-position numerics.
- The thinking budget is enforced only on serial steps (the eligibility
  gate defers to serial when `force_end_thinking` is set).
- C=1 only.

## Known gaps toward phase 2 and risks

- **Unverified on GPU.** Builds and unit tests only. First run should check:
  startup lines, no new kernel-resolution errors, output hash identical to
  the serial baseline, identical `DFLASH_ACCEPT` lines on both ranks, and a
  plausible τ (reference 3.75–4.09 at k=7). Then the parity exit criterion
  (cosine ≥ 0.999 of the captured context rows / drafter logits vs reference
  dumps): `ATLAS_DFLASH_CTX_PARITY_DUMP=1` and `ATLAS_DFLASH_BLOCK_DUMP=1`
  still work on this drafter (they write `/tmp/atlas_ctx_parity.*` and
  `/tmp/atlas_block_*` on each rank).
- C=1 only: batched decode (n > 1) stops drafting for those sequences.
  Prefix-cache hits leave holes (full hits stop drafting; Marconi partial
  hits give a partial-context drafter); measure with prefix caching off.
- `ATLAS_GLM_PREFILL_SP=1` is refused (each rank would hold half the rows).
- Context window slide (above) differs from the reference past 2048 tokens.
- The block conv uses block size k+1; for k ≠ 7 this departs from the
  drafter's training block of 8 (warned at startup).
- Memory per rank: drafter 2.34 GB + packed head halves 0.48 GB + drafter
  scratch (~0.5 GB at `ATLAS_DFLASH_CTX_WINDOW=4096`, hidden 4096,
  intermediate 12288) + drafter paged KV (~42 MB per region, `max_batch_size + 1` regions) are taken before the KV
  budget; the per-request context accumulator (84 MB) is allocated at
  `alloc_sequence`.
- Teardown: the observer's allocations are not registered as model
  resources; a model teardown reclaims them through `sweep_unreleased`
  (warning line). Server runs are unaffected.
- Draft latency is on the serial critical path in phase 1 (drafter + host
  readback before each serial step); TPOT with the flag is not a speed
  number.
- Phase 2 (verify) still needs: opening the guards with an EXL3 verify lane
  (r = k+1 rows through KDA batched snapshots, MLA EXL3 chunk path with FP8
  KV + kpool, coop MoE thin up to r·8 slots, EXL3 head on r rows), capture of
  all r verify rows (`try_dflash_capture_all` analogue reading the highway),
  context commit of the accepted prefix, and rollback (phase 3).
