# Phase 2: graph-safe sparse concurrency, measured 2026-09-06

The device-length graph path works, but the measured throughput change is
small and mixed. **Do not promote it as a meaningful speedup.** The explicit
`GLM_MULTI_SEQ_SPARSE_GRAPHS` opt-in remains default-off. Work proceeds to
[bounded C4 batching](c4-implementation-plan.md), not a claim that CUDA graphs
close the speculative-decoding gap in [peer results](phase2-comparisons.md).

## Same-binary graph-off/on comparison

Both ranks used `atlas-glm53-flash:kernel-20260906-v5`, binary SHA256
`e1e98467a6a838953ec2895fde85bbfdc40d3901d592c49840373fe54d1aeb03`.
Runtime source is captured by `ca6b2578`; the binary was built from those
source contents before that commit. These are binary-identified experiment
observations, **not clean-tip CI gate records**.

Fixed settings: TP2/EP2, BF16 KV/index, FP32 recurrent state, native NVFP4
head, 16384 context, 4096 prefill budget, three active/admitted slots, no
speculation, WMMA scorer, batched MLA, C3 grouped MoE, FP4 prefill and
independent sparse indexing enabled. Batch logging was enabled on both arms;
profilers were disabled. The only changed launch setting was the graph opt-in.
No competing workload or build ran during throughput measurement.

Each cell is the median of two measured batches after one warm-up. Requests
started at a client barrier, used identical token prompts, temperature0 and
64 output tokens. Every measured request in this table reached64 tokens.

| Workload | Off e2e tok/s | On e2e tok/s | Off post-first-text tok/s | On post-first-text tok/s | Post-window change |
| --- | ---: | ---: | ---: | ---: | ---: |
| C2, 1000 prompt tokens each | 15.012 | 15.149 | 16.680 | 16.854 | +1.04% |
| C3, 1000 prompt tokens each | 23.728 | 23.350 | 26.584 | 26.117 | -1.76% |
| C3, 3000 prompt tokens each | 11.605 | 11.742 | 13.358 | 13.540 | +1.36% |

E2e is all completion tokens divided by the full batch wall. The post-window
rate subtracts one token per stream and divides by last completion minus
earliest first text. It still contains staggered prefill and batch drains;
it is not a steady full-width GPU rate. The historical harness metric, which
does not subtract first tokens, remains in the raw JSON for compatibility.

Exact compact-JSON token-array SHA256 fingerprints:

- 1000: `d5d5ef3912ad4ca0305e6b5a7c0895ff78b24fdd4fa30713742529848f42a52f`
- 3000: `374d9fab23e9c29c07e85f26c650b51a71f32d6f1011f73c086ffe89096fc266`

The first exploratory96-token run stopped some streams at90 through the
separate fuzzy-repetition watchdog. Its partial-output receipts are excluded
from this fixed-output comparison. `--allow-repetition` raises the per-request
content-loop threshold only: it does not disable fuzzy detection, EOS, other
stop conditions, or the server's rollback safeguards. No global watchdog was
disabled to obtain these rates.

## Correctness and safety gates

- 667 model CPU tests, seven GLM server tests and eight metric tests passed.
  Default-feature model check, formatting, kernel-shadow and license checks
  passed. Clippy still reports the three pre-existing findings documented in
  the first campaign; the complete multi-model serve matrix was not run.
- The standalone GPU harness first passed18 fixed-C3 graph replays, then20
  production-geometry replays at16400 capacity,4100 scores and32 heads. It
  changes device lengths, shuffled tables and physical slots at fixed
  addresses, covers dense/sparse and pool/page boundaries, and poisons future
  pages and allocation guards. Invalid lengths include0, capacity+1 and
  UINT_MAX. The production geometry exercises the partial513th scorer CTA.
- Production-geometry exact-tie tests also passed20 replays. NVIDIA Compute
  Sanitizer memcheck reported **zero errors**. Device allocations were
  53,263,524 bytes. The harness compares wrappers with current shared old
  exports; it is not a numerical comparison against a historical binary.
- Both ranks captured C3 and C2 slot-keyed graphs in the initial three-needle
  short smoke test; all three requests retrieved only their own needles.
- The eager control passed six mixed2047/2048/2049 retrieval checks and three
  mixed10000/10001/10003 checks, with no foreign needles. Logs confirm real
  C3 `[0,1,2]` and noncontiguous C2 `[0,2]` batches, not just offered clients.

- Graph-on also passed all six mixed-threshold and three mixed10K checks,
  without foreign needles. Including its initial short smoke, all12 graph-on
  retrieval checks passed. Both rank logs show exactly one capture for each
  key: C3 `[0,1,2]`, C2 `[0,1]`, `[0,2]`, `[1,2]`. Subsequent requests reused
  those keys across short, threshold and long histories. Head traces contain
  430 C3 steps and323 C2 steps, including68 noncontiguous `[0,2]` steps.

Retrieval tests deliberately retain EOS and watchdogs; their output caps are
unequal and they are not throughput tests. This is a limited behavioral
qualification, not a complete model-quality evaluation.

The114GiB container ceiling and4GiB load guard were retained. Observed host
available memory stayed around10–11GiB during model tests; the bounded build
used two CPU cores, at most4GiB and no GPU. No reset, reboot, clock adjustment,
swap-policy change or sudo operation was used. A CUDA shared-symbol linkage
conflict failed the first offline PTX build and was corrected before deployment.

## Reproduction and artifacts

Use the same v5 image and fixed settings above on both ranks, changing only
`GLM_MULTI_SEQ_SPARSE_GRAPHS=0/1` between restarts. Run on the development host:

```bash
/usr/bin/python3 scripts/benchmark_glm53_concurrency.py \
  --base-url http://192.168.8.181:8890 --prompt-tokens 1000 \
  --output-tokens 64 --min-concurrency 2 --max-concurrency 3 \
  --repetitions 2 --allow-repetition
```

For the sparse row use `--prompt-tokens 3000 --min-concurrency 3`.
Raw client JSON and rank logs are retained outside CI gate storage under
`/tmp/atlas-glm53-phase2-20260906/` on the development host. The original v65,
v4 long-concurrency, and v5 eager-control containers are preserved stopped
under distinct rollback names. Do not start one beside a resident model.
