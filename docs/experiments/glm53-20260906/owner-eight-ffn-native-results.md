# Five-through-eight-owner FFN foundation

2026-09-09. This is a bounded single-GPU arithmetic microbenchmark, **not**
TP2 serving throughput, model quality, or C6/C8 admission.

## Immutable source and scope

Harness commit `045a25b4`, archive SHA256
`ed9df9e090dba51f711263b4ec2cd004398f79a9d8e852084c2809f5c0fbda75`;
native executable SHA256
`fb44d55188bb8cd4e7ac670b2f954c90b4fb8b511103cb581c0851ef723e0b0b`.
Built for `sm_121a` with the existing production shared generic-T and routed
FFN kernels. No new kernel arithmetic was introduced by this harness change.

Control executes M10 chunks, with an established K5 tail for odd owner counts.
Candidate executes the complete M25/M30/M35/M40 batch. Both use dense routed
down projection. Each process exercises scalar/vector shared loads, both
measurement orders, three EP masks and normal/reversed routing. Two rank
arithmetic contributions are serialized on one GPU; no NCCL is measured.

Rows beyond31 have six explicit row-identity input bits to avoid the previous
fixture's period31 alias. The old two/three/four-owner literal inputs are
unchanged. All shared intermediates are compared bit-exactly, alongside
full router/post oracles and sampled full-K shared/routed projection oracles.
Poisoning and restored-input repetitions cover every owner, including rows32–39.

## Results

Both initial timing and a fresh-process repeat use20 iterations, with D2H
outside the timed region. Ratios below cover both scalar/vector loads and
both execution orders; they are control time divided by candidate time.

| Owners / rows | First speed ratio | Fresh-process speed ratio |
| --- | --- | --- |
| 5 / M25 | 2.950–3.025× | 2.982–3.028× |
| 6 / M30 | 2.890–2.951× | 2.899–2.973× |
| 7 / M35 | 3.868–3.957× | 3.919–3.974× |
| 8 / M40 | 3.809–3.894× | 3.778–3.901× |

All four widths pass zero-tolerance arithmetic comparisons. Separate Compute
Sanitizer memcheck processes for each width report zero errors and zero leaked
allocations. The largest actual guarded device allocation is156,625,800 bytes,
below the unchanged192MiB harness limit. Timing containers have2GiB memory,
memcheck containers4GiB; memory-swap equals memory, with fixed timeouts and no
model loaded. These checks do not replace full-model coherence, tool calls,
needle retrieval, or supervised distributed memory/exit evidence.

All wider memcheck/repeat containers and the old C2/C3/C4 arithmetic controls
exit0 without Docker OOM indications. Post-campaign both nodes have no running
containers or GPU compute applications, zero used swap, and about116GiB
reported available memory in `free -m` (118,744/118,742MiB respectively).

## Receipts

Local campaign directory:
`/home/abc/storage/models/atlas-campaigns/20260909/glm-native-controller`.

- Initial: `owner-eight-{c5,c6,m35,m40}-native-timing.log`.
- Repeat: `owner-eight-repeat-c{5,6,7,8}.log` and `.log.inspect`.
- Memcheck: `owner-eight-memcheck-c{5,6}.log`,
  `owner-eight-m{35,40}-memcheck.log`, and matching inspection receipts.
- Old arithmetic controls: `owner-eight-control-c{2,3,4}.log`.
- Immutable source: `owner-eight-harness-source.tgz`; build log
  `owner-eight-native-build.log`.

Next gate is real model compute/producer/transport integration through eight
owners, followed by a committed native image and supervised warmed C1–C8
quality and timing campaigns. The currently qualified serving image remains
`a069efc3`; do not apply the FFN ratios to its end-to-end rates.
