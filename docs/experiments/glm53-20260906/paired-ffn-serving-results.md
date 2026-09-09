# Paired routed-FFN serving comparison

2026-09-09; native source `5b4662a9`, default-off candidate. This is the
internal warmed 148-input/256-output coding workload, not a reproduced MiaAI
workload. Two fresh processes per mode now pass, including a reversed-order
comparison using instrumentation-only source `816ddaad`.

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

## Fresh-process repeat and actual acceptance

`816ddaad` adds only committed-transaction counters and bounded summary logs;
model arithmetic is unchanged. Run order is Joint then TwoK5, reversing the
first comparison. Both modes again pass all quality checks, retain identical
answer/tool/retrieval/coding outputs, and complete actual paired release and
both Docker/controller exit0. Zero observed swap and no OOM/restarts.

| Mode | C1 full-wall tok/s | C2 aggregate full-wall tok/s | C2 decode-window tok/s |
| --- | ---: | ---: | ---: |
| Two existing K5 FFNs | 27.365 | 27.320 | 28.020 |
| Joint routed M10 FFN | 27.345 | 34.418 | 35.510 |

The repeat gain is26.0%. Joint C2 differs by0.25% between fresh processes.
Actual logs on both ranks confirm E6 commits with the corresponding mode.
Each coding C2 wave completes65 paired transactions, with the same physical
owner histogram in both modes: accepted-draft pairs(0,0)/(1,1)/(2,2)/(3,3)/(4,4)
occur4/5/12/13/31 times. C1 similarly records65 serial transactions per wave
with counts4/5/12/13/31. Thus the coding speedup is not explained by changed
draft acceptance. Counts describe committed prefixes, not emitted tokens:
the output cap can suppress the last committed tokens. Distinct quality prompts
also exercise mixed acceptance pairs; this is not all25 native rollback proof.
The old generic `Done` acceptance fields are not populated by the selected
caller; use the new `GLM C2 committed summary`, not its zero-valued legacy p1.

Minimum host available memory is9,978,920/9,721,416KiB for Joint and
9,961,652/9,974,428KiB for TwoK5 (head/worker). Evidence names follow the first
comparison with `816ddaad-{joint,two-k5}-repeat`. Server ELF:
`a34a2b86de0f79c567bb8e7a57219a1d5bbbbb92ce9c7981b2f7cbed3debf276`.
Head image `a9611043610d98406ca3337c27cb094caccde9768ef3026454c4c5d757e15db3`;
worker image `6bd8956fec14f6cf79eb85e772f7f5d083d50af73aee6cd9d633fed372c7dac7`.

C2 now falls within MiaAI's published31–37 range on our internal workload,
but remains below the37 upper target. C1 remains below30. Concurrent MTP at
C3..C8 and comparable broader serving capabilities remain outstanding; the
non-speculative C1..C8 baseline is not replaced by these two-slot results.
