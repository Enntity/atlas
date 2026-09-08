# v24 first-attempt private-KV evidence

2026-09-08. All six requests already have different canonical private-KV prefix
hashes across ranks **before the first proposer body executes**. The input and
post-EH rows agree, and the newly appended K/V row agrees afterward. This
localizes an observed difference upstream of the current body; it does not
identify which prefix producer differs or prove a KV corruption/reset bug.
Status: CPU-reviewed and bounded native diagnostic/health gates passed;
**not a serving performance promotion**.

## Executable, profile and complete-log validation

Diagnostic source `2da770dc`, unchanged CUDA `189db87e`, image
`atlas-glm53-flash:kernel-20260908-v24`; root verified both executables as SHA256
`818d1ad78b2f0cfcffff354dd045c2437572bed4f2860d5bb5fdf99b2d515efb`.
The root-owned offline Rust build used the bounded8GiB/two-CPU builder,
finished in2m11s and exited0/OOMfalse; no model or standalone GPU work overlapped
build/packaging, and no other GPU workload overlapped diagnostic serving.
The retained build log contains existing closure warnings; success is not a
warning-free-build claim. This executable does not include the separately
promoted B-tile CUDA or the unapproved checked-family worktree.
The actual profile is C1 TP2/EP2, four drafts with accepted-pair repair,
context2044, configured prefill1024, BF16 KV, target shared cache ON/VERIFY OFF,
target verification graphs ON and shared overlap ON. This is not the earlier
no-overlap control. Both ranks enable the existing hidden-trace diagnostic.
Containers retain114GiB limits and4096MiB host guards; post-load host available
memory is12,315/12,172MiB, with zero swap use.

One warmup plus five sequential literal148/64 requests produced complete logs
with192 strict version3 records per rank: generation1..6, slot0, eight attempts
and four steps per request. The manifest explicitly owns every trace record.
Independent analyzer replay exactly reproduced `v24-hidden-analysis.json`;
there are no missing/duplicate/foreign-owner/schema/chain failures. Logs were
frozen before subsequent quality requests.

The [probe contract](../../../scripts/dev/glm_mtp_first_kv_probe_plan.md) reads
only valid logical prefix rows, excludes unused tails/future blocks, and hashes
raw BF16 K/V rows in canonical logical order. Physical block-map hashes are
separate. Only attempt1/step0 receives KV observations: **12 prefix, 12 appended
and12 map digests total**, not one probe for every trace row. At prefix148 the
implementation contract adds303,104 prefix bytes plus2,048 appended bytes per
request/rank, using one32KiB reusable host payload buffer. This byte budget is
derived from the checked implementation, not an independent native copy trace.

## First-attempt measured boundary

Every request starts at position149/seed1304/step0, hidden_row0, cache148→149.
The prefix includes all148 logical rows, including bootstrap repair; the
appended observation is logical row148.

| Observed boundary | Cross-rank result | Across six requests |
|---|---|---|
| Input BF16[4096] | 6/6 equal | All12 hashes identical |
| Post-EH BF16[4096] | 6/6 equal | All12 hashes identical |
| Existing canonical prefix | 6/6 different | Rank0 six unique; rank1 one stable hash |
| Newly appended K/V row | 6/6 equal | All12 hashes identical |
| Physical-map digest | 6/6 equal | Alternates between two values |
| Final BF16[4096] | 6/6 different | Rank0 six unique; rank1 one stable hash |

Shared hashes:

- Input: `6c03a73705a818c124de98781562cbaf9fa239cdf3ba860ce4f1f1694ad75965`.
- Post-EH: `1970b738b46101eed56d8b0989801578c713a42d2fd312d8399da94e03c6ea27`.
- Rank1 prefix: `492a374877bb35504750b89914c150d097d753e510d2f88ce1e077063daa838b`.
- Appended row: `e91f0ed403dabd3dbf62192c6b793cb44a1be5a42b66797cc823658922fe4261`.
- Rank1 final: `b2cdc71699edf14687cef52097bb437c42923e7b058d57a380e05a26f9231ec3`.

Rank0 canonical prefix hashes, in request order:

| Generation | Request | SHA256 |
|---|---|---|
| 1 | warmup | `bf35b464ccc3303edec34f30d9eca8d4aa1e10c9aba2e1469f8d7dea3821dd62` |
| 2 | repeat1 | `264c219ac2c2a88c186b52c2d18debe0b7cd7bc8bf8e1a1e7e1594f160314b6f` |
| 3 | repeat2 | `7cb6d30562dbc0c1c30bd35f37572f097e112f1e83fd072298728ead2485b6a0` |
| 4 | repeat3 | `842b4adb2883c7dff795e228b947a6e49e716f9e576df0bbb1be3f2e5a113b16` |
| 5 | repeat4 | `0de9c6fc87c6029058fae28a5af2448695ef8ba17f45196c368de998715a2424` |
| 6 | repeat5 | `77a9979986f83b96204b303b15d8faa74a7ffd227a3e2905ab9ef3a20b2648a7` |

The first rank0 final is
`3311392faf4a1416531319929b19c89f79de9514cc58e5a30f513468bc40814c`.
Both ranks choose49449 from the same gathered pairs: shard0(26.625,49449),
shard1(16.625,1307). Equal chosen tokens do not establish equal body state.

Odd generations have physical-map digest
`af3cef2d5f930f17afac3f58daca3aa410d11e6a5d86501cb39ae0476a5b1485`;
even generations have
`2a818ae3d51e2156a6e85d2d4ccdced684e668e6c202596259fd70d746af77cd`.
Rank0 prefix changes even between requests sharing a map digest; rank1 prefix
remains stable while map digests alternate. Physical-map variation is not
being mistaken for logical-prefix variation.

## Remaining sampled steps and repeated comparisons

Across all192 cross-rank step pairs, every final-hidden hash differs. All48
sampled step0 post-EH pairs agree. The144 later input hashes differ consistently
with the preceding final-hidden chain. Drafts, gathered pair payloads and
reported causal metadata agree across ranks on all192 pairs. Classification
counts are six `kv_prefix_difference`,42 `final_hidden_difference`, and144
`input_difference`; classification precedence does not erase the later final
differences. The remaining186 step pairs have no KV probe and no inferred KV
equality.

All nine declared warmup/adjacent-request comparisons match32 steps per rank,
with no unmatched position/seed keys or changed reported metadata. Unlike
the exceptional v23 trajectories, there is no reduced matching coverage here.
On every edge rank0 differs in24 input and32 final hashes, while rank1 differs
in neither. All matched sampled post-EH pairs agree and no chosen draft changes.
Gathered-pair payload difference counts per edge are19/14/15/16/16/11/10/12/12,
identical on both ranks because each observes the gathered payload; this does
not mean both shard-local computations changed.

Each edge has only one available first-attempt KV comparison per rank: rank0
prefix differs, rank1 prefix agrees, and appended rows agree. These18 pairwise
comparisons reuse12 captured probes; they are not18 new probe executions.
Neither matching metadata nor digest stability proves static weight equality,
unobserved scratch equality, or complete target causal-prefix equality.

## Interpretation and health boundary

The current body's input and measured post-EH output do not explain away the
already different prefix it consumes. Equal appended bytes show no observed
difference in that one newly written K/V row; they do not prove every internal
WKV intermediate or the whole body is correct. A prior prompt-hidden capture,
primer/repair computation, initialization or later mutation could account for
the prefix difference. The present aggregate hash cannot distinguish those
possibilities or locate a differing row/byte. Audit those upstream producers
before prescribing a broad reset, precision change or synchronization fix.

All five measured client outputs reached64 tokens with completion text SHA256
`1bdda4caa55ae2df2226c4c3c080d7ab23a6ca80ba56488a1de6b75e0ee2f669`.
Four answer checks and the1,984-token needle case passed in the subsequent
quality receipts. A deliberate two-second cancellation returned curl28 after
8,642 response bytes; four subsequent recovery answer checks passed. Both
containers then stopped with exit0/OOMfalse and were preserved as
`atlas-glm53-v24-hidden-clean-ep0/1`. Those separate root-owned health
receipts do not alter the frozen six-request trace window or establish broad
model-quality certification.
All trace timings are diagnostic and cannot qualify30 C1/60 C4. No numerical
fix or performance promotion is claimed by these observations.

## Persistent evidence

Directory: `/home/abc/storage/models/atlas-campaigns/20260908/`.
Build receipts are `v24-native-build.log` and `v24-native-build-identity.log`.
The following suffixes use prefix `v24-hidden-`:

| Suffix | Bytes | SHA256 |
|---|---:|---|
| `clean-rank0.log` | 347510 | `240b71e8145179fcaaa08131af5d52092ba92e655f13cb27119499c7da16748a` |
| `clean-rank1.log` | 321813 | `83e2a8019093c002cbef9f11f88d313eb58dfe1ffc49494a90a96bfdcea4f909` |
| `analysis-manifest.json` | 1676 | `717c29d691a469651a69255426a4d7340e786ceba2c619a569de1d083faf9dd0` |
| `analysis.json` | 2144050 | `33cf876c7a799e0a1dc4e2df2acd0d90c72cf91ba6c6d44bf722769828d587ae` |
| `c1-64.json` | 5863 | `9f63271cddb511c2b459c0490a5ae03cb35881643e6d4c763f0410a5980c1fc5` |

See also `load-memory-both.log`, `answers.json`, `niah.json`, `cancel.stderr`,
`recovery.json` and `stop-head.log`/`stop-worker.log`, plus the
preceding [v23 post-EH](glm-mtp-hidden-trace-v23-results.md) and
[no-overlap control](glm-mtp-nooverlap-control-results.md) results.
