# Phase 7: shared-cache deployment continuation (2026-09-08)

The acceptance target remains a reproducible full-wall median of at least30
tok/s at C1 or60 aggregate tok/s at C4 on the unchanged LRU148/256 fixture,
one warmup plus three measured waves and a confirming repeat. Latest bounded
C1 medians are28.699 initially and28.618 on a fresh v26 restart; qualified C4
remains47.319 on its separate profile. Neither establishes the target. The
preceding v25 medians28.632/28.726 and v20 medians28.545/28.529 are historical
comparisons, not evidence of a v26 speedup. The
earlier27.947/25.161 C1 receipts below are historical, not current best results.
Do not substitute post-first-token rates or discard low-acceptance
samples. The four strict answer checks are a smoke gate, not comprehensive
model quality qualification.

See [deployment-current.md](deployment-current.md) for profile safeguards and
rollback boundaries. The latest [v23 no-overlap control](glm-mtp-nooverlap-control-results.md)
changes the diagnostic divergence distribution without curing it; its separate
trace-OFF full-wall median28.181 is a negative timing result, not a promotion.
The [first-KV probe2da770dc](glm-mtp-first-kv-v24-results.md) passed root's v24
native/health gates and found a difference in the existing private prefix,
before the current body. No throughput promotion or checked B-tile serving
activation is approved here.

A subsequent source audit found a reachable unordered target-prefill/eager-
drafter handoff on the head, already present in v24. The
[bounded stream fix plan](../../../scripts/dev/glm_eager_prefill_stream_fix_plan.md)
records the exact producer/consumer mismatch and real-entry regression tests.
The reviewed fix is now committed `d3f989c7` with917 passing model CPU tests.
The [v25 native result](glm-mtp-stream-fix-v25-results.md) has complete sampled
cross-rank and repeated-request agreement; its automated full quality/health
gate passes. Two earlier attempts retain idle-command health errors explicitly;
the separate idle-receive fix `e40a9066` now passes two native35-second C1/v1
idle gates. Its [v26 results](glm-ep-idle-v26-results.md) record trace-OFF
initial/fresh full-wall medians28.699/28.618 and clean quality/shutdown.
Native v2 coverage remains separate. The small historical differences are
not causal significance or30 C1.
Existing28.53 C1 rates are retained historical measurements, not
evidence that the old handoff is safe or a speedup prediction.
The checked B-tile family is now independently approved and committedad367a70;
its897-test CPU gate does not activate the resident layout.
The next ownership prerequisite is committed `6a486364`: exact GLM checkpoint
retirement at the actual MLA/KDA/down free sites, with935 passing CPU tests and
independent frozen-source review. It remains unselected by the whole-model
legacy loader. Its source/CPU archive `btile-retirement-6a486364-receipts.tar`
is verified on controller/head with SHA256
`e7b3647adbbb0bbe4a44a7db071092463190b2ee072a1d9e7b6711fb06db235c`.
Actual resident readers and loader activation remain separate gates. The
[next idle-receive native plan](../../../scripts/dev/glm_ep_idle_native_plan.md)
does not enable them or presume a throughput gain.

## Recovery and artifact identity

This recovery checkpoint used `1c186acf`, the independently reviewed deferred
shared-cache/provenance fix. The prior native builder finished successfully
on2026-09-07 at05:09:34UTC in2m13s, exit0/OOMKilled=false. On resumption both
model services were stopped and both hosts had about116GiB MemAvailable.

Temporary source directories and local temporary receipts had been cleaned
up during the interruption. Docker initially could not extract the retained
binary because its missing Cargo file bind mounts became empty directory
placeholders. Only those two empty placeholders were removed, then the exact
committed source archive restored; no user files or model weights were removed.
The recovered binary was extracted without restarting the builder. The source
archive and full Docker build log are now retained in the head's persistent
`atlas-glm53-deploy-20260906/phase7` directory. Earlier archived phase6 receipts
remain present; the latest unarchived temporary logs are not claimed recovered.

Both nodes' new `atlas-glm53-flash:kernel-20260908-v19` image binaries have SHA256
`9b851f1d2ada98434357dbb4fe10728a5189be61b172438b8254202a901e3583`.
This differs from v18; a recovered-binary string check additionally confirms
the new deferred-cache path. Images use the retained v18 runtime base and the
new binary only. CUDA source is unchanged from189db87e. Full-image IDs differ
between nodes; the executable checksum is the cross-node identity check.

New local receipts are persistent under
`/home/abc/storage/models/atlas-campaigns/20260908/`, not `/tmp`.

## First cache oracle run

The v19 diagnostic enables the shared cache and its byte/output comparisons,
keeps actual verifier graphs OFF and shared/EP-reduce overlap OFF. Other C1
settings retain the accepted-history repair, BF16 EH, NVFP4 WO, one BF16
vocabulary draft, routed M16 and router BN4. Context2044, chunk1024, batch1,
container114GiB, OOM guard4096MiB, no request spill/overcommit remain unchanged.
No native build, packaging or standalone GPU work overlaps model execution.
Both-rank load memory is sampled every5seconds. Hardware results follow below;
the presence of this plan is not evidence of a successful cache load or TPS.

The first v19 attempt passes all126 byte oracles and installs42 caches on EACH
rank. The head then fails closed at ordinary KV sizing: its90% total-memory
budget is107.6GiB, below100.5GiB pre-KV usage plus7.3GiB inference reserve.
The worker reaches ready. Head exits1/OOMKilled=false; worker is stopped
gracefully, exit0/OOMKilled=false. Both are preserved as
`atlas-glm53-v19-cache-budget-ep0/1`. No inference request was sent.

After cache installation head free memory is16,907,632,640 bytes; the exact
future arena+inference reservation is8,490,639,100 bytes. Worker free is
18,159,030,272 bytes at that boundary. The inference reserve retains its4GiB
CUDA headroom plus recurrent-state/prefill obligations. This is a total-budget
rejection, not exhaustion of physical memory or a reason to remove a reserve.

The next profile raises only GPU_MEM_UTIL from.90 to.91, about1.2GiB additional
total budget against a1,056,964,608-byte cache. Keep the same full inference
reserve,114GiB container cap and4096MiB watchdog, and require measured4GiB
host MemAvailable after load/warmup. Both subsequent cache-OFF and cache-ON
controls use.91; do not attribute an unmatched budget change to a kernel gain.

The.91 diagnostic loads successfully with about11.5GiB host MemAvailable on
each rank. Both ranks pass all126 byte comparisons and all126 K5 projection
output comparisons against retained originals. All four strict answers pass.

The1984/48 near-context request then rejects cleanly: the solo first-chunk
policy uses the full1025-row arena, not the configured1024-row scheduling
budget. The cache's validated1024-row limit catches this at layer3. Both model
containers subsequently stop gracefully with exit0/OOMKilled=false and are
preserved as `atlas-glm53-v19-cache91-oracle-ep0/1`. No clean TPS run occurred.

Next fix: keep the scheduler's existing chunk policy and use retained original
T-layout kernels for installed-cache inputs beyond the tested1024-row FP8
range. Preserve the cached K5 fast path and explicit buffer/geometry/lifetime
checks. Do not raise the FP8 admission limit without native evidence, shrink
the context, or call the failed near-context gate a pass. The final image must
repeat this gate before performance qualification.

The independently reviewed correction is committed as `7bfa3bae`. Its actual
dispatch test first failed at1025 rows, then passed the exact old-T ABI for
all three projections. The final824-test model CPU suite passes, including
arena-bound2048-row fallback, invalid spans/owners/capture refusal and a missing
FP8-handle check before any eager oracle mutation. These are CPU dispatch tests,
not native numerical evidence. The frozen three-file Rust slice was copied to
the retained native builder; no CUDA production source changed. Both hosts are
idle before starting the build, which is bounded to8GiB/two CPU cores because
restored source timestamps may cause CUDA recompilation. Native rollout remains
pending until its new executable is built and verified on both nodes.

### v20 native correctness gate

The rebuilt executable completes in6m03s (189 CUDA compilations plus Rust),
exit0/OOMKilled=false. Both `kernel-20260908-v20` images contain SHA256
`6d774b5a4bd69d121a0822af227a504bc818a3b194ca308cf8731c9421a9e447`.
The committed source's824-test CPU suite is repeated successfully. No native
builder, packaging or standalone GPU job overlaps this model execution.

At.91 utilization the eager/no-overlap cache diagnostic passes all126 resident
byte and126 K5 output comparisons on each rank. Four strict answers pass. The
1984-token needle request now passes through its1025-row first chunk and
retrieves `NEBULA-2847` (16 output tokens,3.609s full wall). This is a boundary
retrieval check, not a capped throughput benchmark or a new prefill claim.

A128-token streaming request receives5,642 bytes before its intentional2s
client cancellation (curl exit28); all four fresh answer checks then pass.
Available host memory remains above4GiB, including about11.5GiB after load and
12.3GiB on the head after the long-context request. Both oracle containers stop
gracefully, exit0/OOMKilled=false, and are preserved as
`atlas-glm53-v20-cache91-oracle-ep0/1`.

Next: fresh cache-OFF and cache-ON graph-enabled runs at the same.91 budget,
same image and LRU148/256 fixture, one warmup plus three measurements. The
diagnostic gate above is not evidence of a throughput improvement.

### Matched v20 cache-OFF control

Fresh graph-enabled cache-OFF C1 at.91 completes one warmup and three148/256
measurements:28.049 /28.011 /28.116 full-wall tok/s, median **28.049**.
All outputs reach256. The strict offline correlator matches the four complete
requests and retains the client metrics unchanged; measured mean accepted
drafts is2.583 in all three requests. Excluding each request's first timing
window gives diagnostic medians108.83ms verifier and11.455ms proposal.

Post-run host MemAvailable is12,935,316kB head /12,182,092kB worker. Both
containers stop gracefully and are preserved as
`atlas-glm53-v20-control91-ep0/1`. This is a fresh control, not a cache gain or
proof that older25.161 repeat variability is fixed. The next cache-ON run
differs only in the cache selection, not the image or memory budget.

### Cache-ON initial and clean repeat

Same v20/.91 recipe, diagnostics OFF and graphs/overlap ON:

| Run | Three full-wall C1 rates (tok/s) | Full-wall median |
| --- | --- | --- |
| Cache-OFF control | 28.049 / 28.011 / 28.116 | 28.049 |
| Cache-ON initial | 28.546 / 28.545 / 28.526 | 28.545 |
| Cache-ON fresh restart | 28.599 / 28.215 / 28.529 | 28.529 |

All measured outputs reach256; each run has its own warmup. Initial improvement
is1.77%; the confirming median is1.71% above the control. Acceptance is2.583
for all initial measured requests and2.583 /2.534 /2.583 on the repeat. Initial
diagnostic verifier/proposal medians are106.76 /11.59ms; repeat106.35 /11.52ms.
These exclude each request's first timing window and are not client metrics.

The initial cache-ON decode-window median is30.074 tok/s, but **full-wall is
28.545**, so neither run meets the30 C1 target. The small gain is reproducible
in this matched short profile; it does not prove broader acceptance variability
fixed, a C4 benefit, or a general900-prefill improvement. C4 remains47.319 on
the previous measured profile. No concurrent native compilation/packaging or
standalone GPU workload runs during these measurements.

After recording the repeat's clean logs, the same graph-enabled process passes
the four answer checks, the1984-token needle request (3.592s,16 tokens), and
another intentional2s cancellation followed by four fresh passing answers.
The cancelled client receives5,642 bytes and exits28. Both model containers
then stop gracefully, exit0/OOMKilled=false, preserved as
`atlas-glm53-v20-cache91-repeat-ep0/1`. The executable and restart recipe are
retained while root runs standalone tests with both models stopped.

## Routed B-tile M64 standalone extension

Committed prototype `40c4bf5c` completes four native fixture passes: ordinary
and Compute Sanitizer execution for each of default FMA and `--fmad=false`.
Both sanitizer summaries report zero errors; the bounded GPU container exits
0/OOMKilled=false. Model services and native builders were stopped throughout.
The fixture explicitly allocates 56,015,240 device bytes, below its64MiB cap.
Across eager and fixed-pointer graph execution it checks dense, separate
compact, and fused compact readers, scalar/vector scales, gathered/route-major
activation ownership, both output arrays, CPU reference columns, immutable
inputs and allocation guards, including empty and remote-only routes.

This is correctness evidence through1024 rows, not an engine integration or
TPS result. The newly observed solo1025-row arena remains outside this
prototype's envelope. Before replacing physical routed-weight storage, extend
the readers' native coverage through actual admitted arena rows (including
concurrency padding); unlike the shared-cache experiment, overwritten original
T storage cannot provide a fallback. Typed layout ownership, all other readers,
and the BF16-versus-prequantized activation contract remain promotion gates.

Native executable SHA256 values:

- default: `080468e50c0facdb3b15bc76e9af62e859f809ec9422510921ad31e8567680c3`
- no-FMA: `08dc5f760ca0445caf48d10c40e4246d4db261ddab0356ea901bbfe00c48043e`

Raw receipts: persistent campaign `btile-m64-full/native-gates.log` plus
the corresponding CPU bounds/fixture logs. No timing claim is made here.

### Padded-arena and native-source follow-up gates

The1088-row extension `ba37e742` and native-source byte repack `6246882b` now
pass all eight native fixture executions: ordinary plus Compute Sanitizer for
both compiler FMA policies, for each fixture. Four sanitizer summaries report
zero errors; the2GiB/two-core standalone container exits0/OOMKilled=false.
Both model services and all builders/packaging are stopped during GPU work.

The reader checks27 cases with all12 graph/ABI/scale/gather variants and
57,211,784 explicit guarded device bytes, including1025/1028/1087/1088 rows.
The native-source fixture checks six complementary-poison transactions per
execution: every packed/scale byte against independent native-to-T-to-tile and
direct references, inverse/bijection, scratch reuse, immutable references,
owner/scalar stability and guards. Explicit device13,632,768 bytes; host tensor
payload18,415,616 bytes. This does not repack any model weights.

Raw `btile-arena-repack-native-{build,gates}.log` records source-slice identity,
executable checksums and completion. The v20 serving receipts and restart
recipes are archived under the head's persistent `phase7/receipts-20260908`;
`phase7/v20-receipt-manifest.sha256` verifies the v20 subset.

## Original-T direct-register M16: correctness passes, no promotion

Standalone source `1f07c65e` removes the intermediate shared B transpose while
retaining original transposed resident weights. Native C4/K5 fixtures pass
ordinary and Compute Sanitizer execution under both compiler FMA policies:
eight complete passes, four zero-error summaries, exit0/OOMKilled=false.
Current and candidate kernels both use56 registers with zero spills; static
shared memory drops18,112 to11,968 bytes. Device fixture peaks are38,566,408
and38,766,376 bytes. These are standalone gates, not resident-model oracles.

Three clean paired C4/K5 timing invocations complete all360 timing rows and
post-timing numerical/input/guard checks. Across eight active cases, geometric
mean candidate/current-M16 latency ratios are0.99468 launch-only /0.99347
builder-inclusive at C4, and0.99715 /0.98984 at K5. The small gains are mixed:
C4 population1 launch median regresses4.09%; K5 boundaries regress on all
three launch-only repeats. Empty and remote-only cases are retained in the
raw record, not counted as active-weight speedups. No production promotion.

Raw receipts: `m16-direct-register/native-{build,gates,timing}.log`. Both model
services and all native builders/packaging were stopped during these runs.

## Bounded hidden trace: first native gate rejected

Diagnostic source `5bc9a6ef`, launcher `a85b1a7b`, and v21 executable
`48f0afc568d89b197a478846390817e2a8e2ee2d19bb787a243af07e6823f6fb`
were packaged and verified on both nodes. Actual container environments
confirm hidden trace and K5 ledger enabled on both ranks. The cache-ON/.91
profile loaded with over11GiB host MemAvailable per node.

The first148/64 request fails the diagnostic's exact-profile guard on both
ranks before any hidden trace record. The worker exits its command loop;
root stops both services gracefully, exit0/OOMKilled=false, preserving
`atlas-glm53-v21-hidden-rejected-ep0/1`. The benchmark exits1 and supplies no
valid capped comparison. Raw `v21-hidden-rejected-rank0/1.log`, container
environment, memory and stop receipts are retained. The no-pool LoRA route
uses inert `Fold` in production while the diagnostic fixture assumes `Skip`;
this mismatch is under investigation, not permission to relax adapter safety.
The proven v20 image and measured28.53 full-wall profile remain unchanged.

## Cache-enabled v20 K5 phase diagnostic

Using the unchanged v20 image/cache-ON/.91 profile, enable verifier profiling
and disable target graphs/shared-overlap. One148/32 warmup plus one148/32
measurement both complete. This is not the148/256 throughput qualification.
Both ranks have16 complete K5 intervals, eight per request, with exactly42
instances of every MoE phase and34 KDA/11 MLA layers per interval. Excluding
the first interval of each request leaves14 per rank, with these median summed
phase times in milliseconds:

| Phase | Rank0 | Rank1 |
| --- | ---: | ---: |
| Routed gate/up, including setup/activation work in that phase |26.6945|27.9190|
| Routed SiLU/down |13.3425|14.4605|
| Shared expert |15.9980|16.5640|
| Router projection |4.9045|4.9310|
| All45 layers, excluding final norm/vocabulary/argmax |123.4900|123.3650|

Do not sum ranks or independently computed medians. Synchronized eager
profiling perturbs timing and removes production overlap; these are priorities,
not graph-on cost shares or causal deltas against the older v16 diagnostic.
Shared work remains material, but the stronger routed-layout experiment also
still addresses a substantial phase. No new kernel is promoted from this data.

Raw `v20-cache-profile-*` receipts include both complete logs, client result,
memory samples and condensed interval/count records. The recipe differs from
clean v20 only by these diagnostics and graph/overlap settings. Both services
stop gracefully, exit0/OOMKilled=false, preserved as
`atlas-glm53-v20-cache-profile-ep0/1`. No native builder or standalone GPU
workload overlapped the model run.

## v22 hidden trace: native and recovery gates pass; divergence localized

Source `b4092742` corrects the diagnostic guard to inspect actual adapter
ownership, including sticky failed-install history, while accepting the inert
no-pool Fold route. It does not change proposer math. Both nodes verify v22
executable SHA256
`dfaf9ebfa5262b8f3f5f8fc7b54564c3a37b52b97d36dd4d336b77c6ef97a5b1`.
Six sequential148/64 requests cap successfully. Complete-file analysis accounts
for all384 records, with no missing/foreign/duplicate/order/chain errors.

Across repeats, rank0 step0 has identical conditioning inputs but differing
post-body/norm hashes; rank1 is mostly stable. Both ranks agree on all gathered
argmax pairs and draft tokens. This narrows the unmeasured region to EH/body/
normalization/private state, not a proven reset or kernel bug. See
[the complete trace evidence](glm-mtp-hidden-trace-v22-results.md).

Four answer checks, the1984-token needle, cancellation, and four fresh recovery
answers pass. Root stops both services gracefully, exit0/OOMKilled=false,
preserving `atlas-glm53-v22-hidden-clean-ep0/1`. These instrumented64-output
requests are not throughput qualification; v20's28.529 confirmed C1 full-wall
median and the separate older47.319 C4 profile remain the measured baselines.

The subsequent read-only audit rules out the simple full-head configuration
hypothesis: explicit MLA dimension overrides reach Q, attention, V and O, and
comm=None suppresses TP/EP reductions. No scratch/stream ownership violation
is established. Captured prompt-prefix rows/private KV remain unmeasured.
The proposed next probe hashes only the post-EH step0 row within the existing
request-owned trace budget; no reset, precision or serving-layout change.

Separately, `d5f9cde9` seals native B-tile provenance and a4MiB repack workspace.
Its856 CPU tests cover real production validation, typed dispatch and ownership,
including exhaustive scalar/copy/launch/sync fault injection. There is no loader
caller, Ready publication or enabled serving reader, and therefore no claimed
model speedup from this infrastructure commit.

## v23 post-EH probe and production-reader integration

The default-off post-EH extension1f675c8b passes859 CPU tests,16 analyzer tests,
and exact reanalysis of every prior v22 evidence field. Native v23 completes
six148/64 diagnostic requests with all384 records accounted for. All48 sampled
cross-rank post-EH hashes agree, while every corresponding final hash differs.
The unexplained interval is therefore after the measured EH boundary; equal
private KV or a particular body bug is not established. Generation3 changes
draft/acceptance trajectories, so three cross-request comparisons match24
steps rather than32. See [complete v23 evidence](glm-mtp-hidden-trace-v23-results.md).

Answer,1984-token needle, cancellation and fresh recovery checks pass. Both
containers stop cleanly and are preserved; the v20 qualified baseline remains
unchanged. No diagnostic decode-window timing qualifies the30/60 target.

Kernel source3332c36e promotes the six unchanged B-tile helper bodies into
production, with standalone fixtures including that same source. All15 new
CUDA exports pass the actual strict compiled-PTX ABI gate after Python-only
parser correctionaa88a9b0. The [native promotion results](glm-btile-cuda-promotion-results.md)
record20 completed standalone executions across both FMA policies and ten
zero-error memchecks, with exact artifact hashes and bounded device budgets.
No serving selection or speed gain is established. Checked Rust family review
and owning publication remain incomplete. The literal
shared/down loader extractiond6f4e554 passes866 model CPU tests, including80
configuration combinations and888 injected I/O failures. It preserves the
legacy allocation/stream/free order and does not activate B-tile storage.
