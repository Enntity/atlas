# Phase 4: cache contracts and state-indexed KDA

Status: cache contract and indexed KDA committed, built, and validated on both
Sparks. v11 short-C4 graphs are deployed; the matched coding workload improves
3.6% full-wall aggregate throughput over v10. This is a small-sample result,
not a general model-quality or full serve-matrix claim.
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

## Post-extraction and integration gates

Production scalar and indexed exports now share private arithmetic helpers;
the original scalar ABIs remain unchanged. The indexed harness compares the
frozen scalar oracle and unchanged TP convolution, rather than comparing only
two users of the same extracted helper. All91 indexed cases and all60 temporal
cases (both compiler FMA modes) pass natively and under memcheck again.
The repeated eager timing measures N2:45.444→18.654us, N3:68.370→28.803us,
N4:93.580→38.100us. Compare paired arms within each invocation; the change from
the earlier microbenchmark's baseline is not a separate serving improvement.
Receipt: `kda-shared-gpu.log`.

Runtime integration passes all712 model CPU tests. Root separately reran that
suite; kernel-shadow, license, formatting and whitespace checks passed.
Native build and the bounded full-model serving gates subsequently passed;
see the v11 results below.

The preceding v10 short2048/chunk1024 graph profile passed budgeted chat and
four near-limit needles, then established the baseline for this integration:
1024/64 C4 full-wall28.162/post-first31.310 tokens/s; coding148/256 C3
34.071/34.628 and C4 45.669/46.425. Same workload hashes, one warmup plus two
measured batches, every output reaches its cap. Containers are preserved,
stopped, as `atlas-glm53-v10-short-control-ep0/1`.

## v11 indexed KDA: bounded serving validation and matched throughput

Source `e8771f30`, image `atlas-glm53-flash:kernel-20260906-v11`, identical binary
SHA256 on both ranks:
`949ec12d3aac6024927cbd070c16b0a1658c1129343bedb222fb48539a5e3f29`.
The subsequent `d3b0ac9f` is equivalent range/alignment style cleanup, not the
source revision of this deployed binary.

The production path now submits one indexed convolution and one indexed
recurrence for eligible independent N2–N4 GLM decode. It uses validated actual
SSM pool slots, refreshes their device metadata before graph lookup/replay,
and preserves exact ordered-slot graph keys. No new GPU allocation or new
tuning flag was introduced. C1, prefill, verification and unsupported profiles
retain their existing paths; a malformed present state view fails before mutation.
The typed cache contract retains existing ownership/allocation sizes; it is
not the proposed single-latent or bounded-tail storage redesign.

Eager full-model gates passed: fresh C1/C3 needles, two four-request near-limit
batches, and four budgeted chat answers. Both ranks logged indexed-path
selection. Preserved stopped containers: `atlas-glm53-v11-eager-ep0/1`.

Graph-enabled gates also passed:

- Four budgeted chat answers, four short independent needles, and eight
  near-limit needles over two batches (1900/1920/1950/1984 prompt tokens,
  caps96/64/48/64, needle position0.9).
- Both ranks actually captured N4 `[0,1,2,3]`, nonprefix N3 `[0,1,3]`, and
  N2 `[0,3]` graphs. Subsequent matching decode traces support replay through
  the inspected graph-cache path; logs do not emit a separate replay event.
- Fresh C1, C3 and four chat answers passed again after all throughput runs.

No foreign needles were found. Needle checks are narrow behavioral gates;
some responses repeat or finish before their caps through the existing content
watchdog. This is not hidden as exact-output or general generation-quality
parity. Fixed-output throughput below separately verifies every requested cap.

Same settings for v10 and v11: context2048/chunk1024, C4 admission, TP2/EP2,
non-speculative, BF16 KV/index, FP32 KDA state, exact M4 MLA, grouped C3/C4 MoE,
FP4 prefill, short decode graphs, sparse switches off, KV overcommit off.
One warmup plus two measured batches per width; medians, temperature0/seed1.
No concurrent CPU build or other GPU workload during timing.

| Workload | v10 full-wall / post-first tokens/s | v11 full-wall / post-first tokens/s | Full-wall change |
| --- | ---: | ---: | ---: |
| C4, 1024 prompt / 64 output | 28.162 / 31.310 | 28.886 / 31.989 | +2.6% |
| C3, coding148 prompt / 256 output | 34.071 / 34.628 | 34.890 / 35.472 | +2.4% |
| C4, coding148 prompt / 256 output | 45.669 / 46.425 | 47.309 / 48.123 | +3.6% |

Full-wall includes prefill and drain. Post-first excludes the first token of
each stream in its numerator and spans first text to last completion; neither
is the sum of per-session rates. The C4 coding median session decode rate is
12.445 tokens/s (v10:11.991). Both measured v11 C4 coding batches deliver
47.308–47.311 full-wall tokens/s. There is no statistical confidence claim or
direct comparison to speculative-decoding public results.

All outputs reached64/256 respectively. The generated1024-token workload alone
uses per-request repetition allowance; the literal coding workload does not.
Its generated code is not executed or graded. Exact prompt-token hashes:

- 1024 synthetic: `e74375dce562b280752663f16202b17f37f388a64155a15e2db604c43fda23ec`.
- Coding148: `8b104308b377752ad9803d298be34e01cedf9f8ac577cacfea653c3359e7aebf`.
- Coding literal UTF-8: `940f3a003a1ac66b5195b27a78e31b92f5622bd29d221a35949056bdd87477d1`.

Receipts: `v11-eager-*`, `v11-graphs-*`, `v11-image-build.log`; matched baseline
`v10-short-1k-64.json` and `v10-short-coding-256.json`. Local root:
`/tmp/atlas-glm53-phase3-20260906/`; persistent head copy:
`/home/mangokid/atlas-glm53-deploy-20260906/phase3/`.

### Safety, known warning, and remaining work

After graph-profile loading, sampled host `MemAvailable` was10410/9655 MiB;
after the final gates it was10540/10607 MiB (head/worker). Final GPU temperatures
were52/57°C. Both containers remained running with `OOMKilled=false`; no GPU
fault, node reset, host-policy change, or sudo action was observed/performed.
Memory limits and guards remained unchanged. These samples are not continuous
minimum-memory telemetry or a guarantee against every hardware failure.

The worker logged a preexisting false slow-broadcast classification after
83.6 seconds idle in eager mode and46.0 seconds idle in graph mode. Both calls
returned successfully, and subsequent quality checks passed. The timer includes
normal waiting for the next command; the latched health bit currently has no
production consumer. No independent CUDA/NCCL fault was found in the saved logs.
Do not clear it through unilateral reconnect. The exact evidence and a typed
intent fix are in [the health follow-up plan](ep-command-idle-health-plan.md).
This issue remains unfixed in v11; output-loop warnings remain separate.

The bounded full-model gates pass, along with all712 model CPU tests and
eight GLM server preflight tests. Formatting, kernel shadows and license gates
passed. Full clippy still has three preexisting model warnings; the full
multi-model serve matrix and full-model MTP regression suite were not run.

[Current deployment and v10 rollback](deployment-current.md) supersede the old
phase-1 recipes. Next infrastructure work is the
[typed EP execution-plan slice](ep-mixed-execution-plan.md): first validated
intent, legacy wire compatibility and two-rank CPU agreement; then bounded
mixed scheduling. Do not lift the current EP/MLA exclusions or confuse the new
independent-row KDA kernel with temporal/ragged prefill support.
