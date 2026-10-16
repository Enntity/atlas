# Prefill A/B at ~3k tokens — grouped fat-MoE multirow (2026-10-16)

Branch `followup/p13-fat-moe-multirow` (`b52aef80` kernel, `fa02f778` workload).
Binary `spark.p13fat`, sha256 `22875833…c32e378`, identical on both Sparks.
It was built by `p13fat_build.sh` from an rsync of this worktree; the build log
names no commit, so the binding to `b52aef80` rests on the prior session's
record. The A/B itself is binary-internal: both arms use the same binary.
Image `src-4ee96c12b44c`, cpuset 15-19, TP=2/EP=1, `ab_full.sh` base flags +
the exact decode stack (V2/V3/V6/V7, SHARED_OVERLAP, arena skip, graphs).
Descriptor profile: `--max-seq-len 4096`, `fp8_g128` KV, one sequence.
Every run held `/tmp/atlas-bench.lock`. 3 repeats per cell; the medians below
use the client first-output-token time, and the server-side
`usage.time_to_first_token_ms` agrees to within 0.2 s in every cell.
Run dirs are under `$EV/perf-ab-20261015/`.

## One-variable A/B: `ATLAS_EXL3_COOP_MOE_FAT_MULTIROW` (chunk 128)

| Prompt tokens | Flag off (`p14L-off-20261016T151027Z`) | Flag on (`p14L-on-20261016T152550Z`) | Speedup | `output_sha256` (off = on) |
|---:|---|---|---:|---|
| 761 | 41.75 / 42.20 / 42.72 s, median 42.20 (18.0 tok/s) | 3.46 / 3.42 / 3.46 s, median 3.46 (220 tok/s) | 12.2× | `1c4328172e81` |
| 2,943 | 165.73 / 165.88 / 165.80 s, median 165.80 (17.8 tok/s) | 13.74 / 13.81 / 13.79 s, median 13.79 (213 tok/s) | 12.0× | `671ef63c142a` |

- **Bit-exact:** all 12 outputs in each workload share one hash across both arms.
- **Engagement:** the flag-off log has the `fat (prefill) routed launch`
  fallback line; the flag-on log has none.
- **Reproduces the earlier session:** the `p13fat-off`/`p13fat-on` runs gave
  42.72 → 3.42 s on the 761-token prompt.
- The speedup holds near the context limit: per-token prefill cost is flat
  from 761 to 2,943 tokens in both arms.

## Chunk size (flag on, 2,943-token prompt) — numerics-changing

| `--max-prefill-tokens` | Run | Median | tok/s | `output_sha256` |
|---:|---|---:|---:|---|
| 128 | `p14L-on-20261016T152550Z` | 13.79 s | 213 | `671ef63c142a` |
| 256 | `p14Lchunk256-20261016T153143Z` | 11.76 s | 250 | `4c433997b96d` |
| 512 | `p14Lchunk512-20261016T153710Z` | 10.68 s | 276 | `e779a96788f1` |

Each chunk size is deterministic over its own repeats, but **the outputs
differ between chunk sizes**, and the earlier 761-token chunk-512 run also
changed the hash (`07355bd7f0ef` vs `1c4328172e81`). Chunk size changes the
per-expert row counts in each launch, and with them the thin/fat profile
split. So a larger chunk is a numerics-changing serve setting, not a free
speedup: it needs its own label and qualification.

- Chunk 1024 was not run. Chunk 4096 previously wedged spark1 with `EP peer
  lost` and a docker-inspect timeout, after only 6 GB was left free under
  `--gpu-memory-utilization 0.84`.

## Not covered

- The serving stack (`SHARED_PACKED`, `GLM_SPARSE_DECODE_FP8_SPLIT`, prefix
  caching, 16k context) was not A/B'd; `serve_perf.py` still serves the
  older `spark.5e1674ba` binary without this flag.
- No prompts beyond 4,096 tokens were tested.
- There is no reference-engine (vLLM) prefill on the same hardware yet.
- Default-flip neutrality on a fresh image is still owed.
- Remaining flag-on attribution is in `$EV/p13trace3-20261016` (761-token
  prompt). It splits as follows: thin multirow MoE
  (`exl3_coop_moe8_multirow_b4_k32_n256`) 32%, `grouped_gemm_mla` 8.6%,
  `exl3_mhc_pre_norm` 8.1%, coop GEMMs ~15%.
