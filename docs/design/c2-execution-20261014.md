# C2 execution, 2026-10-14: treatment blocked by worker request binding

C2 remains **unresolved**. The first treatment repeat stopped on its first request; the second never ran. Treatment was not collected, and the native local o_proj first measurement remains **undischarged**. No downstream diagnostic score or new numerical qualification verdict is claimed.

The durable preparation and attempt root is `/home/colum/Workspace/atlas-exl3-validation-20261010/q2-c2-execution-20261014`. The compact [summary](evidence/c2-execution-20261014/summary.json) binds the source files, failure logs, baseline artifacts, and teardown records by SHA256. The [analyzer](evidence/c2-execution-20261014/analyze.py) retains its complete-campaign guards; these incomplete attempts must not be scored with it.

## What completed

Both baseline repeats in `c2-refexact-02` completed all eight registered requests: arithmetic steps 0–3, followed by longer steps 0–3. Their final logits archives and every full-vocabulary row are byte identical across repeats. Each repeat also matches the same eight rows of the archived same-image `gate-v2-qualification-20261013-luna1/attempts/v2cand01/attempt-files/short16` artifact.

For both repeats, all 16 rank/invocation gated-norm payloads match the archived candidate `q2-norm-replay-20261011/fixtures/rankN-rowM/expected.npy` tensors. The native archive's `[tokens, heads, dim]` layout is compared with the explicitly bound `[tokens, heads × dim]` capture layout. No archived weight file or retracted replay was read.

This establishes observer/capture neutrality **only for these eight rows**, under runtime source `2fbb787cd49f2b524746b04707b90aadd1f150fc`, image `sha256:011423015fc1da354e07fbe5ee1b49751efb4e3a886159f2a23f458744cbcff0`, and actual KV format `fp8_g128`. Historical image neutrality remains separately bound to its existing proof; this result does not establish neutrality for other cases or the newer production v24 path.

## Why treatment stopped

`c2-refexact-01` failed before creating either container because the adapter's container-absence guard did not recognize Docker's lowercase diagnostic. Its failure and empty lifecycle records remain intact. The separately bound `473aa8bc` adapter correction allowed a fresh `c2-refexact-02` attempt.

In the first treatment request, rank 0 consumed its `reference-exact` fixture for `c2-20261014-req-00` and emitted one valid post-substitution receipt. Rank 1 terminated with: “norm substitution requires the ForwardContext request-id binding”. The worker path lacked that request binding. Rank 0 then reported its bootstrap peer connection closing, and the collector received `RemoteDisconnected` without a response.

There is one rank-0 substitution receipt and 15 unconsumed fixture entries. Treatment has no completed request, `manifest.json`, or `logits.npz`; the second treatment repeat never ran. Partial engagement cannot discharge the first measurement or establish causality.

## Teardown and next step

The completed baseline runs and failed first treatment run each preserved both rank IDs, container logs, stop/wait receipts, and captures. All three lifecycle records report zero teardown/retrieval errors. Both failed attempts remain preserved.

Post-failure host inspections at 16:15:20–16:15:23 UTC show **no active GPU compute processes on either Spark**. All C2 containers are exited: both baseline repeats exited 0 on both ranks, and the failed treatment exited 74 on both ranks. The compact summary records both inspection paths and hashes.

The runtime worker request-binding path must be fixed and verified, followed by a newly bound image, neutrality evidence, and a fresh registered C2 attempt. The current campaign must not be topped up or relabeled as successful.

Summary SHA256: `547d3369c657a9020141613cdec53b22d3e245a19369e3115ff34f7d0f902981`.
