# Plan: ClickHouse-style Column Codecs for KV Cache Compression (Experiment)

Status: **PLAN ONLY — not implemented.** No code lands outside the staged
gates in §5. Scope decisions are logged in [`decisions.md`](../../decisions.md);
assumptions and constraints in [`assumptions.md`](../../assumptions.md);
future work in [`todo.md`](../../todo.md).

Idea under test: KV cache streams are treated as *columns* (one channel
across token positions) and compressed with ClickHouse-style **composable
column codecs** — data-preparation transforms (Delta, DoubleDelta, Gorilla,
T64) piped into general compressors (LZ4/ZSTD), with **per-block adaptive
codec selection**. Reference spec:
[ClickHouse CODEC clause](https://clickhouse.com/docs/reference/statements/create/table/codec).

---

## 1. Clarification of Scope (per AGENTS.md / ponytail)

| Dimension | Decision | Notes |
|---|---|---|
| Deliverable in this pass | **Plan only** | Experiment implementation is staged (§5) |
| Target tiers | **All tiers, incl. GPU-resident** (operator choice) | GPU-resident is contraindicated vs the TQ+ kernel line — staged **last** (Stage 3) with kill criteria (§2, §5) |
| Data domain | **All serialized state**: BF16 KV, post-quant streams (codebook indices + scales), SSM state | Codecs compose **after** the existing lossy quantizers; the codec layer itself is lossless |
| Codec set | **Gorilla-TSDB core + T64 + adaptive**: `Delta`, `DoubleDelta`, `Gorilla`, `T64` + `LZ4`/`ZSTD` backends, composable pipelines, per-block smallest-of-N selection | Full catalog (ALP, FPC, ZXC, GCD, SZ3, AES-GCM-SIV) is deferred (§8) |
| Success gate | **End-to-end decode tok/s under spill pressure** must hold or improve vs baseline | Bit-exact roundtrip is mandatory regardless of gate. All numbers per measurement-discipline (§3) |
| Plan methodology | Six Sigma DMAIC; staged implementation + test plan | Each stage: entry criteria, implementation steps, test plan, exit/kill criteria, rollback |

---

## 2. Define

### Problem

KV cache bytes dominate the memory hierarchy during long-context serving.
Atlas already applies *lossy* quantization to the GPU-resident cache
(`KvCacheDtype`: FP8/NVFP4/Turbo2–8 and the TQ+ asymmetric variants). The
host-side tiers — `kv_spill.rs` swap files (incl. SSM state overflow), the
`--high-speed-swap` host tier, prefix-cache persistence — store KV
**uncompressed**. No tier applies general-purpose or structure-exploiting
*lossless* compression to KV streams.

### Hypothesis

**H1 (ratio).** KV streams have exploitable regularity along the token axis
(correlated neighbouring positions, narrow integer codebooks, smooth
scales), so ClickHouse-style pipelines
(`CODEC(Delta, ZSTD)`, `CODEC(DoubleDelta, LZ4)`, `CODEC(Gorilla, ZSTD)`,
`CODEC(T64, LZ4)`) compress KV blocks materially better per unit CPU than
plain per-block ZSTD/LZ4.

**H2 (throughput).** Codec encode/decode cost is small relative to the
bytes saved on the spill/reload and (Stage 3) attention-bandwidth paths, so
end-to-end decode tok/s under spill pressure holds or improves.

**H3 (GPU-resident, Stage 3 only).** For bit-oriented codecs (T64
crop+bitpack, Delta+bitpack, Gorilla XOR), in-kernel decode is cheaper than
the HBM bandwidth it saves.

H1 is the cheap falsifiable core. H3 is the risky tail — it is tested last
and may be killed without failing the experiment.

### Primary metric (Y) and secondary metrics

- **Y:** end-to-end decode tok/s under spill pressure (GB10 flagship,
  `tests/single_gpu_suite.py` spill slice + `atlas-spark-bench`).
- Secondary: compressed bytes per KV block vs baselines; encode/decode
  GB/s per Grace core; spill/reload latency; compression ratio per stream
  type.
- **Mandatory invariant:** decoded KV is bit-exact vs the uncompressed
  path at every tier (these are lossless codecs; the existing quantizers
  own all lossiness).

### In scope

Codec library + block format; integration with `kv_spill` serialization
(incl. SSM state), the high-speed-swap host tier and prefix-cache
persistence; (Stage 3) codec-compressed HBM storage read by attention
kernels. Offline benchmark harness and captured KV corpora.

### Out of scope

Replacing or re-tuning the `KvCacheDtype` quantizers; encryption codecs;
multi-node KV coordination (`docs/plans/atlas-kv-coord-plan.md`); SZ3-style
lossy modes; adaptive-codec ML selection (keep smallest-of-N brute force).

### Contraindication register (AGENTS.md duty)

| # | Contraindication | Mitigation |
|---|---|---|
| C1 | **GPU-resident codecs collide with the TQ+ kernel work** (`docs/turboquant-plus.md`): decode-path kernels were just reworked; adding decompression to attention reads regresses decode latency and duplicates `KvCacheDtype` dispatch | Stage 3 is **last**, restricted to cheap bit-oriented codecs, gated by inline roofline arithmetic (measurement rule 7) and kernel parity tests. Killable without failing the experiment. Hot/cold page split is the fallback design |
| C2 | **Lossless codecs vs existing lossy quant line**: risk of two overlapping "KV compression" stories | Codecs compose strictly **after** quantizers; the plan measures both BF16 and post-quant streams (§4) and documents the layering: quantizer = lossy rate–distortion, codec = lossless byte squeeze |
| C3 | **New deps (`zstd`, `lz4_flex`) vs `deny.toml` AGPL policy** | Verify licenses at Stage 0 entry (both BSD-family upstream; expected compatible). No FFI to system libs |
| C4 | **KISS**: columnar re-layout + 5 codecs is a lot of machinery for an experiment | Stage 0 is offline-only and buys the go/no-go with no runtime integration; smallest-of-N adaptive selection replaces codec-prediction heuristics |

---

## 3. Measure — baselines and measurement discipline

All perf claims in this experiment follow
[`measurement-discipline`](../../.agents/skills/measurement-discipline/SKILL.md):

- **Fingerprint every run** (rule 1): build commit, serve binary + full
  flags (max-seq-len, batch, kv dtype, spill/swap flags), checkpoint id +
  revision, harness params (ISL/OSL, prompt class, temp/seed, thinking
  on/off), box, date. Recorded as a sidecar file next to each result.
- **n≥3 repeats** for anything that headlines; single runs reported as
  "preliminary, single-shot" only.
- **One-variable A/B** (rule 2): codec on/off differs in exactly one
  config field; everything else pinned.
- **Engagement check** (rule 4): a negative result needs a positive
  control. A run where every block header records `codec=NONE` has *not*
  exercised the codec layer. Block-header codec histograms are the
  engagement proof and ship with every result.
- **Graduated claim language** (rule 5): Stage results say "observed X
  under fingerprint F" until cross-validated.

### Baselines (captured once, reused by all stages)

| Baseline | What |
|---|---|
| B0 | Raw spill (current `kv_spill.rs` bytes) |
| B1 | Per-block ZSTD(3) and LZ4 on raw blocks (the "obvious" alternative) |
| B2 | Existing quant dtypes alone (`fp8`, `nvfp4`, `turbo4`, …) without codecs |
| B3 | B2 + best Stage-0 codec pipeline (the layered hypothesis) |

---

## 4. Analyze

### What is a "column" in a KV block

The paged block layout (see `crates/spark-runtime/src/kv_cache/paged_impl.rs`)
stores per-page slabs of `[K|V] × kv_heads × head_dim` for a run of token
positions. The **columnar view** slices one `(side, kv_head, head_dim
channel)` stream across the token positions of a block — i.e. the analogue
of a ClickHouse column where rows = positions. Exact axis order and the
block/window size (one page vs a multi-page window) are **captured and
measured in Stage 0, not assumed**; the codec works on byte streams either
way.

### Codec × stream fit matrix (hypotheses to test in Stage 0)

| Stream | Domain | Candidate pipelines | Rationale |
|---|---|---|---|
| K/V values along token axis (BF16) | float16 | `Gorilla→ZSTD`, `Delta→ZSTD` | neighbouring positions correlated → small XOR/deltas (unproven: the central empirical question) |
| Codebook indices (turbo/nvfp4/fp8 packs) | 2–4 bit ints in byte/nibble containers | `T64→LZ4`, `Delta→LZ4` | tiny value range → T64 crops bits; locality → small deltas |
| Group scales (FP8 scale streams) | 8-bit | `Delta→LZ4`, `Gorilla→ZSTD` | smooth along positions |
| Position/page-id metadata | int64 | `DoubleDelta→LZ4` | constant stride — the Gorilla-TSDB classic |
| SSM state | float | `Gorilla→ZSTD`, plain `ZSTD` | no token axis; per-channel streams |

### Prior art (external references in §7)

- Gorilla TSDB (VLDB 2015) — `DoubleDelta` timestamps + `Gorilla` values
  is exactly the ClickHouse composition being ported.
- ClickHouse codec docs — pipeline composition, T64 bit-transpose trick,
  adaptive codec selection (per-block smallest-of-N, store raw if larger,
  record codec id per block).
- SplitZip (arXiv 2605.01708) — lossless KV compression with GPU-friendly
  encoding; closest published work to H1/H3.
- ZSTD lossless KV/weights study (arXiv 2503.18869) — ~47% KV reduction
  reported; anchors B1 expectations.
- delta-compress-llm — delta encoding over KV (F16 at Q4 size), supports
  the token-axis locality premise.
- `docs/plans/atlas-kv-cache-server-plan.md` — `murr` columnar cache
  reviewed as reference-only (layout mismatch with Atlas blocks).

---

## 5. Improve — staged implementation + test plan

Each stage is a standalone PR-sized unit. **No stage starts before the
previous stage's exit criteria pass.** Any stage may be killed per its kill
criteria with the experiment still reporting a result.

### Stage 0 — codec library + offline benchmark (no runtime integration)

**Implement.** New small crate `crates/atlas-codecs/` (Atlas small-primitive
idiom): codec pipeline enum (`Delta`, `DoubleDelta`, `Gorilla`, `T64` prep
stages; `Lz4`/`Zstd(n)` backends; composable pipelines), per-block header
(codec id, uncompressed/compressed len), and a bench binary that consumes a
captured KV-block corpus. Corpus: dump real blocks from a served model via
the spill path (plus synthetic control streams with known structure).
Verify `zstd`/`lz4_flex` licenses against `deny.toml` (C3) at entry.

**Test plan.**
- Roundtrip property tests: arbitrary block bytes → encode → decode ==
  input, for every pipeline and edge size (0, 1, 63, 64, unaligned).
- Golden vectors for each specialized codec (hand-checked bit patterns).
- Ratio oracle: run `clickhouse-local` with matching `CODEC(...)` on the
  same column data; ratio agreement within tolerance validates the idea,
  byte-identical output is **not** required (implementations may differ).
- Bench output = ratio + encode/decode GB/s per stream type, with full
  run fingerprints.

**Exit criteria.** Codec × stream winners chosen from data; H1 supported
(≥1.5× smaller than B1 on at least one major stream) or falsified with
numbers. **Kill criteria.** H1 falsified on all streams → experiment ends
here; report is the deliverable. **Rollback.** Delete the crate.

### Stage 1 — spill tier (host): `kv_spill` serialization + SSM state

**Implement.** Route `kv_spill.rs` serialization through
`atlas-codecs`: columnar re-layout within each swap file block, per-block
adaptive smallest-of-N selection (never larger than raw), codec id recorded
in the block header (PCND: explicit, no implicit default). SSM state
serialized with its own pipeline from the Stage-0 matrix. Byte budget in
`KvSpillManager` counts **compressed** bytes.

**Test plan.**
- Unit: spill → reload → dequant (`kv_dequant.rs`) == original BF16,
  bit-exact, for every `KvCacheDtype` × pipeline combination
  (`kv_cache/tests.rs` idiom).
- Integration: multi-page spill/reload under `--high-speed-swap` with
  SSM overflow; swap-file lifecycle tests (stale-file cleanup, budget
  enforcement on compressed sizes).
- Perf gate (Y): e2e decode tok/s under spill pressure, n≥3, one-variable
  A/B vs B0/B1, engagement histogram non-trivial. Positive control: a
  deliberately compressible synthetic workload must show high
  `codec≠NONE` share.

**Exit criteria.** Bit-exact roundtrip everywhere; tok/s ≥ baseline with
measured ratio ≥ Stage-0 projection × 0.8 (integration overhead allowed).
**Kill criteria.** tok/s regression > 2% at n≥3 despite ratio win →
default to `codec=NONE` config and stop at Stage 1 (ratio-only win
recorded). **Rollback.** One config flag reverts serialization to raw.

### Stage 2 — host offload tier + prefix-cache persistence

**Implement.** Apply the same pipeline to the `--high-speed-swap`
host-tier blocks and prefix-cache persisted blocks. Decode path:
codec-decode → existing `kv_dequant` (unchanged). No GPU changes.

**Test plan.**
- Parity: reload from codec-compressed host tier == reload from raw tier,
  bit-exact (extend `metal_backend/tests/parity_*` helpers where a GPU
  compare is the cheapest oracle).
- Reload latency A/B (fingerprinted, n≥3): compressed reload must not
  regress TTFT on cache-hit requests; decompression is CPU-side on Grace
  cores and should be bandwidth-cheap.
- Prefix-cache correctness: radix-tree hit/miss behaviour unchanged;
  persisted block format versioned (old files read as raw/NONE).

**Exit criteria.** Bit-exact parity, TTFT neutral-or-better on hits,
capacity win measured (how much more context fits in the host tier).
**Kill criteria.** TTFT regression > 2% on cache hits → keep Stage 1 only.
**Rollback.** Format version flag; old readers tolerate both.

### Stage 3 — GPU-resident (contraindicated; last; killable)

**Implement (only if Stages 1–2 pass and the entry review approves).**
Restricted design: **bit-oriented codecs only** (T64 crop+bitpack,
Delta+bitpack, Gorilla XOR) on **cold pages** in HBM; hot pages and the
decode critical path stay uncompressed; LZ4/ZSTD never in a kernel. First
design question at entry review: uniform codec-compressed cache vs hot/cold
page split (default: hot/cold). Dispatch extends the existing kernel
dispatch tables; a compressed-page flag rides the page metadata.

**Entry gate (measurement rule 7 — roofline is a hypothesis).** Inline
arithmetic first: bytes/token read × GB10 bandwidth vs in-kernel decode
cycles, then empirical confirmation. The arithmetic must predict a win
before any kernel is written.

**Test plan.**
- Kernel parity: attention outputs with codec-compressed cold pages ==
  uncompressed, per `parity_turbo*` idiom; every new dispatch arm covered.
- tok/s gate (Y) under spill + long-context pressure, n≥3, one-variable.
- Numerical edge cases: empty/full pages, eviction races, partial pages.

**Exit criteria.** Parity + tok/s win attributable to reduced KV bandwidth.
**Kill criteria.** Anything short of it — including "parity works but no
tok/s win" — ends Stage 3 with a written result; **the experiment succeeds
on Stages 1–2 alone.** **Rollback.** Page flag keeps all reads on the
uncompressed path.

---

## 6. Control

- **CI:** `atlas-codecs` roundtrip/golden tests in the standard pipeline
  (fmt, clippy `ATLAS_SKIP_BUILD=1`, license headers, typos; file-size cap
  ≤500 LoC/file).
- **Perf regression detection:** the Stage-1/3 tok/s A/B procedure is
  captured as a Justfile recipe (measurement commands per core AGENTS.md),
  so any later codec change re-runs the gate with fingerprints.
- **Docs:** ADR recording the codec layering decision (quantizer vs codec)
  and the Stage-3 outcome; `docs/GB10_DEPLOYMENT_GUIDE.md` gains the spill
  compression flags when they ship; release notes per `docs/releases/`.
- **Config:** codec pipeline selection is explicit per tier (PCND); no
  implicit defaults in production paths.

---

## 7. External references

- ClickHouse CODEC clause (spec for pipelines, T64, adaptive selection):
  <https://clickhouse.com/docs/reference/statements/create/table/codec>
- ClickHouse codecs blog: <https://clickhouse.com/blog/optimize-clickhouse-codecs-compression-schema>
- Gorilla TSDB paper (DoubleDelta + Gorilla):
  <https://www.vldb.org/pvldb/vol8/p1816-teller.pdf>
- FPC (float predictor + XOR): <https://userweb.cs.txstate.edu/~burtscher/papers/dcc07a.pdf>
- ALP (adaptive lossless float): <https://ir.cwi.nl/pub/33334>
- Chimp (float compression): <https://www.vldb.org/pvldb/vol15/p3058-liakos.pdf>
- SplitZip — lossless KV compression, GPU-friendly:
  <https://arxiv.org/html/2605.01708v1>
- Reimagining Memory Access for LLM Inference (ZSTD KV/weights):
  <https://arxiv.org/html/2503.18869v1>
- KV cache compression review: <https://arxiv.org/html/2508.06297v1>
- delta-compress-llm (delta encoding over KV):
  <https://github.com/cenconq25/delta-compress-llm>
- XOR float compression explainer: <https://clemenswinter.com/2024/04/07/the-simple-beauty-of-xor-floating-point-compression/>
- In-tree: [`docs/turboquant-plus.md`](../turboquant-plus.md),
  [`crates/spark-runtime/src/kv_spill.rs`](../../crates/spark-runtime/src/kv_spill.rs),
  [`crates/spark-runtime/src/kv_dequant.rs`](../../crates/spark-runtime/src/kv_dequant.rs),
  [`crates/spark-runtime/src/kv_cache.rs`](../../crates/spark-runtime/src/kv_cache.rs),
  [`docs/plans/atlas-kv-cache-server-plan.md`](atlas-kv-cache-server-plan.md),
  [TurboQuant paper](https://arxiv.org/abs/2504.19874)

---

## 8. Follow-ups (recorded in todo.md)

- Full ClickHouse catalog sweep (ALP, FPC, ZXC on Grace ARM, GCD) if H1
  holds.
- Adaptive codec *prediction* (learned per-stream selection) if
  smallest-of-N encode cost ever matters.
- ZXC asymmetric codec for compress-once/decompress-many reload paths.
- Cross-replica compressed KV transfer format (link `atlas-kv-coord-plan.md`).
