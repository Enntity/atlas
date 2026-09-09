# Paired routed-FFN serving comparison

2026-09-09; native source `5b4662a9`, default-off candidate. This is the
internal warmed 148-input/256-output coding workload, not a reproduced MiaAI
workload. First fresh process for each mode; another process remains required.

| Mode | C1 full-wall tok/s | C2 aggregate full-wall tok/s | C2 decode-window tok/s |
| --- | ---: | ---: | ---: |
| Two existing K5 FFNs | 27.295 | 27.314 | 28.012 |
| Joint routed M10 FFN | 27.333 | 34.333 | 35.420 |

C2 improves **25.7%**; C1 is unchanged within 0.2%. Each width has one warmup
and three measured waves. Joint C2 measured waves are34.333/34.340/34.303;
control27.367/27.314/27.314. Every output reaches256 tokens. Both modes produce
identical full coding text in all warmup/measured C1/C2 requests. The four
concurrent answer messages, two structured auto-tool messages and two retrieval
outputs also match between modes and pass their existing validators. This does
not certify generated programs, broad model quality or all rollback combinations.

Both campaigns fully pass the corrected native controller: real paired
quiescence/release, controller exit0 and independently inspected Docker exit0
on both ranks. No OOM, restart or observed swap use. Minimum host available
memory is9,715,604/9,451,012KiB for control and9,823,108/9,447,548KiB for joint
(head/worker). Both nodes return to approximately116GiB available after exit.

The same binary/images, two active slots, eager MTP4, BF16 KV, FP32 SSM,
2044-token context,1024-token prefill cap and114GiB no-swap container limits
are retained. Only `ATLAS_GLM_C2_PAIR_FFN=two-k5|joint` differs. Both preserve
the existing generic-T shared projections and M5 vocabulary-head arithmetic.
The joint path combines routed expert work and its reduction. Native logs at
this source do not count E6 transactions or accepted drafts; an explicit
committed-transaction summary is being added for the follow-up qualification.

## Provenance

Evidence root:
`/home/abc/storage/models/atlas-campaigns/20260909/glm-native-controller/`.
Prepared directories and retained summaries/logs:
`native-prepared-5b4662a9-{two-k5,joint}-routefix`,
`native-summary-5b4662a9-{two-k5,joint}-routefix.json`, and corresponding
`native-{head,worker,run}-5b4662a9-{two-k5,joint}-routefix.log`.

Server ELF SHA256:
`df4105448138d84e8f78a6580a76bd47087cd9fad141dce2290698f94a022728`.
Head image:
`a51dceefc6fbc755c5b0d9ca9758a89b3a15130e3eb36f756ed9b89998078c99`.
Worker image:
`276b8008a4c5eb9a1425bdb8e65a36436ccd0fbaf7ff298d9f94ea25f992152d`.
Supervisor helper is the unchanged digest-pinned `9aca8d8f` installation.
Raw controller evidence retains exact recipes, workload identity and observations.

C2 now falls within MiaAI's published31–37 range on our internal workload,
but remains below the37 upper target. C1 remains below30. Concurrent MTP at
C3..C8 and comparable broader serving capabilities remain outstanding; the
non-speculative C1..C8 baseline is not replaced by these two-slot results.
