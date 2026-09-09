# Connected paired controller: real Docker/SSH CPU qualification

2026-09-09, based on `d1be1216`. No new inference throughput is claimed.

The runnable supervisor now connects immutable `prepare`/one-shot `run` inputs,
fixed SSH node verbs, actual Docker creation and inspection, pinned host process
observations, persistent guard transports, HTTP readiness, a bounded workload,
paired lease renewals, rank0 drain, both quiescent receipts and both observed
zero exits. Finite status commands do not block the lease loop. Failure cleanup
retains direct child owners until reaped or the explicit bound expires, records
uncertainty and preserves remote resources. Local SSH exit is never remote
completion. The workload interpreter runs through its retained ELF descriptor.

## Actual connected run

The controller host ran two real private-PID/private-IPC Docker containers,
through a dedicated loopback SSH listener with isolated pinned credentials.
Each container had memory=swap 2GiB, CPU0/1, no GPU requests/devices and the real
guard plus the existing registered CPU Model fixture. Fixture HTTP readiness
required both actual registrations. Rank1 waited for rank0's real SIGINT drain
witness before consuming the existing local shutdown replay. This does not
simulate or certify NCCL, GPU kernels, model loading or generated text.

Candidate evidence is retained under:
`/home/abc/storage/models/atlas-campaigns/20260909/glm-native-controller/`.
`controller-cpu-green.log` completed with supervisor exit0;
`cpu-green-evidence.tgz` retains the exact prepared bundle and event journal;
`cpu-green-docker-final.json` retains independent final Docker observations.

- Both actual Model registrations and startup reports were observed.
- A pinned Python interpreter executed the bundled workload: thirteen real
  HTTP health checks spanning twelve seconds, exit0.
- Both lease ordinals1 and2 were queued as paired renewals during that workload.
- Both genuine quiescent receipts and full local release writes were observed.
- Full container IDs `68749b02c9d470cbee7a07307507e2beedf0c2fce6019dfd3f08af8c7b97f0ba`
  and `916b647f2d739bbdeca9bc42bfe76a3d887c4c93a7d6b5fb1627b54ad72a5f9e`
  independently exited0, OOMKilled=false and RestartCount=0.
- The controller host had no swap configured. Both Sparks remained idle with
  zero observed swap use; no native model was started for this qualification.

The candidate supervisor SHA256 was
`9255c716dc6878a825804d947e8925a0bf6268a9bc5a37ed47ab08c38d132bd1`;
prepared bundle digest
`b882c9cfc78a41bd51778d0560ab70eac7098cc2e0c09e7c91deb6f12a00b8e5`.
These identify the actual candidate, not an assertion that a later commit
produced it. Post-commit source/binary checks and fresh receipts belong in the
external campaign closure; do not relabel this candidate run as exact-tip.

## Integration findings and limits

The first actual Engine run refused Docker's omitted empty `Tmpfs` map before
starting either container. The second refused `OomKillDisable:null` after start;
recorded-ID cleanup stopped both gated containers. Handling now admits only
those narrow representations and still rejects extra tmpfs mounts, missing
required fields and true OOM-killer disable. These match Moby's
[omitempty field](https://github.com/moby/moby/blob/master/api/types/container/hostconfig.go)
and [unsupported OOM-disable handling](https://github.com/moby/moby/blob/master/daemon/daemon_unix.go).

The intentional CPU wait stub was then exercised after rank0's actual
`registered` witness; the controller observed terminal EOF and refused success.
The specific inner error is inferred from that fixture source path, not printed
by the production terminal ingress. Rank1 registration was not observed in
that negative run. The GREEN fixture above observed both registrations.

Focused supervisor checks currently pass46 cases, including prepared-file
tampering, actual finite/persistent process I/O, state ordering and the two real
Engine normalization regressions. These checks are safety/adapter evidence,
not progress toward the tokens/s target.

Next: rebuild spark and all three helpers for ARM64 from the frozen source,
retain the v29 rollback, qualify one healthy native C2 MTP run with coherence,
tools and retrieval, then measure warmed concurrency. The selected MTP scheduler
is still serialized across its two slots; batched verification and wider MTP
admission remain performance work, not capabilities demonstrated here.
