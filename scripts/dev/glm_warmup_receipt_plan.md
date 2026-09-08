# Preserve warmup evidence for the next GLM concurrency gate

2026-09-08. The qualified v26 matrix discarded its warmup results. Those
historical receipts remain unchanged; do not retroactively claim warmup caps.

For future runs, retain the one existing warmup batch under `warmup_runs` in
each concurrency result. Keep `runs`, all measured medians and the measured
`all_outputs_reached_cap` flag unchanged. Add a separately named
`warmup_all_outputs_reached_cap` flag, including natural early stops rather
than overriding EOS. Preserve per-request text hashes, counts and timing
offsets already produced by `batch`. No additional requests, new warmup
policy, new dependencies or engine changes.

TDD: drive the actual CLI main loop with only the network batch/prompt
boundaries replaced. Cover C1 through C4, one warmup followed by three
measurements, deliberately different warmup rates, warmup early stops and
measured early stops. Confirm measured medians never include warmup, separate
cap flags are truthful, request identity survives JSON serialization, and
stderr/stdout both retain the evidence. Record the missing-field RED before
the implementation, then run the complete concurrency metric tests GREEN.

Independent review and an exact-manifest commit precede future native use.
No node execution is needed for this harness-only change.
