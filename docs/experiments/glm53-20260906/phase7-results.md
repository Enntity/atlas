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
