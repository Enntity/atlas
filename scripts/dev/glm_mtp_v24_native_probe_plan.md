# v24 native first-KV diagnostic

Root builds only after the first-KV Rust slice has completed behavioral RED,
focused/full CPU tests, non-test check, formatting/license checks and independent
exact-source review. Commit only that slice, then archive committed spark-model
sources; never copy the parallel B-tile worktree into the native builder.

Use the existing stopped8GiB/two-CPU offline native Rust builder. Preserve its
unchanged CUDA189db87e baseline and v23 rollback image. The v24 overlay image
contains only the newly built executable on v23, with exact committed Rust and
CUDA identity labels. Verify SHA256 inside both rank images. This is a diagnostic
image, not the promoted B-tile CUDA image or a performance candidate.

Both model services and standalone GPU work must be stopped during build and
packaging. After compilation completes, launch the existing v23 overlap-on
diagnostic recipe with only image changed to v24: explicit cache1,verify0,trace1.
Keep114GiB containers,4096MiB guard, KV overcommit0, swapspace0, maxcontext2044,
prefill1024, C1/four drafts, repair, target verification graphs and all numerical
policies unchanged. Refuse existing standard-name containers rather than remove
them. Verify actual both-rank settings and >=4GiB host available memory.

Run the same literal148/64 prompt with one warmup and five sequential repeats.
Freeze complete logs from both ranks before any other requests; require all384
records and the exact generation1..6 manifest. Use the committed strict v3
analyzer. Report first-attempt post-EH/prefix/appended/final equality separately
from physical block-map equality and chosen drafts. The probe adds305,152bytes
of D2H per request/rank at148rows, not every attempt. No throughput inference.

Then perform the existing four answer checks,1984/16 needle case, two-second
client cancellation and four recovery answers, sequentially. Preserve logs and
stop/rename both containers cleanly. If any owner/budget/health guard fails,
retain the failure and stop; do not relax it or silently omit failed requests.

Equal private prefix and appended row would narrow the next boundary; different
prefix would direct investigation upstream. Neither alone proves a root cause,
and this plan does not authorize broad cache zeroing, synchronization changes,
precision changes or a serving promotion. Any follow-up discriminator gets a
separate bounded plan based on these actual observations.
