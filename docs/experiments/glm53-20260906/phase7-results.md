# Phase 7: shared-cache deployment continuation (2026-09-08)

The acceptance target remains a reproducible full-wall median of at least30
tok/s at C1 or60 aggregate tok/s at C4 on the unchanged LRU148/256 fixture,
one warmup plus three measured waves and a confirming repeat. Current C1
receipts are27.947 initially and25.161 on a clean repeat; neither establishes
the target. Do not substitute post-first-token rates or discard low-acceptance
samples. The four strict answer checks are a smoke gate, not comprehensive
model quality qualification.

## Recovery and artifact identity

Production source remains `1c186acf`, the independently reviewed deferred
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
