# Phase 4: cache contracts and state-indexed KDA

Status: cache contract committed and built; indexed KDA standalone gates passed,
production integration underway. No indexed-KDA serving speedup is claimed yet.
The architectural basis is the [pinned vLLM roadmap](vllm-infrastructure-roadmap.md).

## Existing-layout cache contract

Commit `240d6f6e` makes main allocation, MTP allocation, and the exact512 BF16
kernel requirement share a validated GLM geometry. It adds checked byte and
physical-capacity accounting while retaining separate K/V ownership, full-block
index tails, all existing dtype choices, and MTP's existing block-count policy.
691 model CPU tests passed; independent review and native build passed.
Image v10 binary SHA256 on both nodes:
`316cfa33bb72bb08709ee699880f909d631dc137453e5a91e396bed065d039ce`.
This infrastructure change does not itself reduce memory or claim more tokens/s.

## Guarded long C4: quality passes, performance limitation retained

Image v9, source `7b9dac41`, binary SHA256:
`39706bb596a9f3cd0fb7c312018f00ba3a566ee23f2dece6efeb3dd5b7a1e1c7`.
TP2/EP2, non-speculative, native NVFP4 weights, BF16 KV/index, FP32 KDA state,
context16384, prefill chunk1024, four active/admitted slots, no KV overcommit,
eager sparse decode, grouped C3/C4 MoE, exact M4 MLA, WMMA index and FP4 prefill.
Both ranks retain114 GiB container ceilings and GPU-utilization fraction0.90.

Physical cache capacity was6822/8175 blocks (head/worker), above4097 required
for four full16K sessions including the dummy. Sampled host `MemAvailable`
remained above10 GiB after load and during these tests; this is not a CUDA-free
memory measurement. No host reset, OOM kill, privileged host change or GPU fault.

Quality gates all passed without foreign needles:

- Four short prompts768/800/832/896, caps32/16/48/64.
- Two batches around2048: prompts2047/2048/2049/2051, caps32/64/48/96.
- Unequal prompts8192/12288/15000/15360, caps32/64/48/96, needle position0.05.
- Boundary prompts16288/16320/16336/16352, caps96/64/48/32, position0.9.
  **All four boundary requests actually reached16384 total tokens.**
- Fresh four-request budgeted chat afterward; a16385-token prompt was rejected
  with HTTP400 before model execution.

The boundary batch includes an actual N4 trace followed by nonprefix N3 drains,
but only a short N4 interval. It is not evidence of sustained C4 at16K. The
matched3072/64 benchmark below supplies sustained N4 sparse-decode coverage.
Needle matching is a narrow behavioral check, not unrestricted model quality.

| Same v9 profile, 3072 prompt / 64 output | Full-wall aggregate tokens/s | Post-first aggregate tokens/s |
| --- | ---: | ---: |
| C3 requests | 6.992 | 9.692 |
| C4 requests | 7.306 | 9.654 |

One warmup and two measured batches per width, temperature0/seed1, per-request
repetition allowance, every output reaches64. Token hash:
`8e2036a4364a25fe8e2c95821d6ff6022e9116d9522bc9f3fb4f49a64da8f199`.
C4 gains4.5% full-wall; post-first throughput is essentially flat. This guarded
1K-chunk lane is **not promoted as an overall performance improvement**.

The unequal long batch takes134.5 seconds overall; its last-first-token receipt
is120.8 seconds. The boundary batch takes170.3 seconds. Logs show serial1K
prefill chunks interleaved with decode and exhausted prefix-checkpoint capacity
falling back to SSM recomputation. These are observed scheduling/cache costs,
not evidence that increasing state allocation or changing a flag is safe.
EP mixed scheduling remains a separate infrastructure milestone.

Raw receipts/logs: `v9-long-*`, `v9-over-context-rejection.json`, and
`v9-post-long-chat-16k.json` under the phase3 receipt directory. Preserve the
earlier `v9-post-long-chat.json`: its answers passed, but its harness retained
the short2048 context label. The explicit16384 rerun corrects that metadata;
payloads and validators are unchanged. v9 containers are preserved, stopped,
as `atlas-glm53-v9-long-c4-ep0/1`.

## State-indexed KDA: independent numerical gate

The first standalone candidate failed exact BF16 output comparison despite
matching complete H and convolution state. Production `--fmad=false` removed
the first-case mismatch but still left one BF16 ULP at step2. Removing the
dimension-specializing equality guard, while retaining runtime arithmetic and
host geometry validation, made the initial39 cases pass. No tolerance changed.

The expanded prototype passes91 cases natively and under compute-sanitizer
memcheck: complete FP32 H and convolution history, all BF16 outputs, nonprefix
slots, drains, reset/reuse, invalid-slot masking, changed metadata on graph
replay, zero-Q/K and stronger-state inputs, inactive slots and canaries.
Explicit device buffers:38,894,224 bytes, below64 MiB.

| Rows | Per-row conv + recurrent pair, us | Indexed pair, us | Ratio |
| --- | ---: | ---: | ---: |
| 2 | 45.452 | 20.702 | 2.196x |
| 3 | 68.174 | 26.559 | 2.567x |
| 4 | 113.724 | 39.288 | 2.895x |

CUDA-event intervals around eager submissions; five interleaved100-step trials,
alternating order, median reported, state resets excluded. This measures the
stateful core only, not graph throughput or full-model tokens/s.

A frozen legacy reference separately passes60 temporal cases (tokens1/2/3/5/17,
standard/zero/strong inputs, padded convolution strides and optional bias,
eager/graph) with both GLM no-FMA and default-FMA compilation, natively and under
memcheck. Explicit buffers7,266,944 bytes. These pre-extraction receipts establish
the oracle; **shared-helper extraction must repeat these gates**.
Raw log: `kda-expanded-and-temporal-gpu.log`.

Commit `a61814d1` adds the CPU-tested state/launch foundation and these oracles:
actual SSM pool IDs, distinct from LoRA/KV IDs; checked FP32 pool ranges and
state pointers; disjoint live pools/metadata; existing metadata-gap reuse;
exact13/14-argument kernel ABIs; paired prevalidation before state mutation.
706 model CPU tests and six decode-layout tests passed. It contains no runtime
dispatch wiring. Subsequent integration preserves exact-slot graph keys and
refreshes each rank's real slot IDs before every lookup/replay; no new tuning
flag or new GPU allocation is the objective.
