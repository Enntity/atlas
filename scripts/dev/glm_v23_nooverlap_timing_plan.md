# v23 no-overlap trace-off timing probe

After the six-request diagnostic and its quality/recovery gates completed,
both ranks stopped with exit0/OOMfalse. The no-overlap trace reduced cross-rank
final-hidden divergence from192/192 to32/192 sampled steps but did not remove
it. This is not a correctness fix or permission to promote a serving profile.

While controller-only implementation continues, root may time the unchanged
v23 image using the existing frozen no-overlap recipe with explicit CACHE=1,
VERIFY=0, TRACE=0. No build, weight conversion or standalone GPU workload may
overlap it. Keep114GiB containers,4096MiB guard and all existing memory/context
limits; require both standard names absent and verify both executable hashes,
effective trace/cache/overlap values and post-load host memory.

Use the unchanged literal LRU prompt,148 prompt tokens,256 output cap,
temperature0,seed1,normal repetition policy,C1,one warmup plus three measured
requests. Full-wall median is the primary result. Every measured output must
reach the cap; record content hashes and retain full logs. A result at or above
30 requires a fresh confirming restart plus quality/recovery gates before any
throughput target claim, and the unresolved hidden-state discrepancy must still
be disclosed rather than describing this flag as a source-level fix.

No acceptance threshold, prompt or benchmark implementation changes. Stop and
preserve the containers cleanly after the bounded run. An observed speedup is
separate from the ongoing B-tile kernel integration and KV/root-cause work.
