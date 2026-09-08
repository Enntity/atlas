# v23 shared-overlap diagnostic control

Root-only native control after the promoted-reader standalone gates finish.
Keep the exact v23 executable, all model shapes, cache policy, four drafts,
accepted-pair repair, target verification graphs and hidden tracing. Change
only `MOE_SHARED_REDUCE_OVERLAP=1` to0 in the frozen C1 recipe. No kernel,
weight precision, repair, scheduler or source change. This is a diagnostic
control, not a proposed serving optimization or throughput qualification.

Reason: v23 shows identical post-EH rows but different final rows. A remaining
possibility is initial target-prefill auxiliary shared work affecting the
subsequent no-communicator proposer body or the state it consumes. The actual
`forward_prefill.rs` predicate requires EP shared work, `num_tokens > 64`,
no graph capture and no profiling. Thus this control changes the initial
148-token target prefill; the five-token K5 verifier is already sequential
with either flag value. It is not a K5-verifier overlap experiment.

The source makes the auxiliary stream wait on `event_a` and joins `event_b`
before the shared blend. No missing join or race has been established.
Disabling this overlap tests the initial-prefill execution/history candidate;
it cannot exclude every other asynchronous or private-state difference.

Require both model services and native builders/standalone GPU tests stopped
before launch. Preserve old containers, verify the one-flag recipe diff and
actual both-rank environments, keep114GiB container limits and4096MiB guard,
and check host memory after loading. Run the same148/64 literal request with
one warmup plus five repeats. Snapshot both complete logs before other requests
and analyze all generation1..6 records with the existing strict v2 analyzer.

Compare exact initial input/post-EH/final hashes, complete record coverage,
cross-rank and matched cross-request differences, not just chosen tokens.
Repeat outcomes with different causal metadata remain separate. Persistence
of divergence means disabling this overlap alone did not remove it; disappearance
only implicates the configuration/execution change and still needs a source
cause, not an immediate synchronization workaround.

Run answer/needle/cancel/recovery gates afterward, then stop and preserve both
containers cleanly. Traced timings never count toward30 C1/60 C4. No compiler,
packaging or other GPU workload overlaps the model run. The new private-KV
probe and checked B-tile dispatch may progress on the controller independently.
