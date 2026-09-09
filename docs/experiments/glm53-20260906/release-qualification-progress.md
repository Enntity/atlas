# Release qualification: September 9–10

Deadline: September 10, 02:42 UTC. Native engine source is `a6cfeec0`, merged
with upstream `6c5f17dab9c27ee2396aef1ac2501a17b201c715`. Portable C8 harness
and its followup-envelope correction are `708db27c` and `440526f2`.

## Immutable native build

Build completed with exit0 and OOM=false (6m29s server, 6.58s helpers).
All212 GLM CUDA kernels were compiled; no skipped CUDA build.

| Artifact | SHA256 |
|---|---|
| Source build archive | `77c785741e67b838312fa824a91cf99ebd576ab809077984ae4226585cc41166` |
| Server ELF | `9892e1a88932dea0ba4d0097bdf2b05ebef95f3599b9098f7f8019ed943523e9` |
| Head image | `7c3d3c18923d84a1cd95366cc64122e1dd5ef93703e5488ea5cb2f180fe0e60a` |
| Worker image | `f1725e96a928072bee3e38e9bc5d1f68fc79e2c3e68a25d9c203ab620e9ce872` |
| Guard | `9622492cbc51448f26591f56399f0b8c3f3d981bfd2cb49978ccc46cb759d0ff` |
| Node supervisor | `b0613a45f96de1caccbf1fa843cf1fa2276f30355de7e74102dd964cf561a9a0` |
| Node relay | `2ea3f845cfb240f854e83e92f743e405017ce09faef1480f557af7953c34300a` |

Evidence below is retained outside Git in
`atlas-campaigns/20260909/glm-native-controller`. Failed runs remain failures.

## Findings before final qualification

1. Initial C8 startup: worker budget818 KV blocks was below8×128 demand.
   The selected startup refused; the controller stopped both containers
   (exit137, OOM=false, swap0). This is **not** a clean quiescent serving exit.
   The unused16-slot Marconi prefix snapshot reservation was then explicitly
   disabled with `--ssm-cache-slots=0 --ssm-checkpoint-interval=0`. Selected
   prefix reuse is unsupported and context2044 cannot reach its4096-token
   checkpoint. Live state/MTP snapshots, CUDA reserve and watchdogs remain.
   The new profile allocated4333/4180 target blocks and admitted eight owners;
   both ranks logged actual E8 commits at widths8/7/6/5.
2. New C8 tool-result roundtrips: all8 initial answers and8 tool calls passed,
   but4 followups hit their64-token cap during reasoning. The shared thinking
   policy has a soft32-token budget with sentence-safe deferral up to roughly96,
   including on selected MTP. Correct own results appeared in reasoning; this
   alone is not a visible-answer PASS or evidence of a hidden-state failure.
   The new followup allowance is128; exact visible results and normal stops
   are still required. Coding requests remain148 input/256 output, unchanged.
3. First long-context launcher attempt refused Docker's canonical `CAP_...`
   capability names before either container started. The exact checker was
   corrected against both retained real create observations; missing/extra
   capabilities remain rejected.
4. First actual4K ordinary run: early needle was exact and stopped; middle
   needle was correct but repeated until the content-loop watchdog stopped it.
   The bare-completion probe **failed**. Both model processes subsequently
   exited0, OOM=false, swap0. A fresh probe now uses actual `/tokenize(messages)`
   no-thinking chat-template tokens, not guessed framing. Exact answer/stop
   checks and watchdogs are unchanged; new quality remains to be measured.

## Performance interpretation

The4K raw run's first1028-token chunk took1.064s (warm0.949s), while later1024
chunks took2.46–2.57s. Overall3968-token TTFT was8.34s, not a comparable1K
benchmark. Later paged GLM attention still uses the scalar BF16 GEMM for four
projections; an opt-in existing-cuBLAS experiment is being prepared separately.

Pre-merge C1 coding used65 verification rounds per256 output tokens; initial
merged traces use80. The merged full argmax changes equal-BF16-logit tie order.
This can alter target/draft trajectories and acceptance. The extra rounds are
consistent with much of the observed rate reduction, but proving the cause
requires actual shared-prefix tied logits, not timing alone. Distributed shard
argmax is OFF in both compared recipes, so its host merge is not the active
cause here. Do not present old C1–C4 rates as current merged-image results.

Final C1–C8 timing, repeated portable launch, chat-templated full-context quality,
and any prefill candidate are still pending. No reference-performance goal or
maximum-context qualification is claimed by this progress record.
