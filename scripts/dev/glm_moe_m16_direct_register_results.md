# Original-T M16 direct-register result: no promotion

Date: 2026-09-08. Source commit: `1f07c65e`. The four source files remain
unchanged after independent review. Root alone built and ran native tests with
model services stopped. This analysis reads their persistent receipts.

## Decision

**Do not promote the direct-register candidate.** Correctness and memory gates
pass, but the current-M16 comparison is too small and inconsistent. Removing
the intermediate transpose is not an established material latency win. Keep
this negative result and pursue the separately validated resident B-tile path.

All 360 timing rows were paired by shape, repeat, case and path: three repeats,
two widths, ten cases, six paths. The main active-case summary excludes only
empty and remote-only controls; their complete comparisons remain below.
Ratios are candidate/current M16 latency, not M64 speedups. Negative percent
means faster. Each cell is the reported median of five interleaved 100-launch
trials; this log does not retain individual trial samples or confidence bounds.

| Shape | Launch old/new mean us | Launch geometric mean change | Builder-inclusive old/new mean us | Builder geometric mean change |
| --- | --- | --- | --- | --- |
| c4 | 165.423 / 164.562 | -0.532% | 198.835 / 197.474 | -0.653% |
| c5 | 169.911 / 169.425 | -0.285% | 206.210 / 204.125 | -1.016% |

These means weight all eight active synthetic cases equally, not actual model
routing frequencies. Active launch wins are 17/24 C4 and 14/24 K5; inclusive
wins are 18/24 and 20/24. Mean absolute savings are only 0.861/0.486 us launch
and 1.362/2.085 us inclusive (C4/K5). C4 aggregate launch change by repeat is
+0.212%, +0.514%, -2.297%; inclusive -1.620%, -0.836%, +0.507%.
K5 launch is -0.700%, -0.105%, -0.049%; inclusive -1.455%, -0.693%, -0.899%.
Including controls in the geometric means gives C4 launch/inclusive
-0.512%/-0.528%, K5 -0.283%/-0.815%; this does not change the decision.

Important counterexamples: C4 population1 regresses in both first two launch
repeats (median +4.09%) and every inclusive repeat (median +1.09%). K5 boundary
launches regress all three times (+3.12%, +1.52%, +1.18%). K5 partial-all-local
launch changes sign (-4.44%, +1.48%, +3.41%). Small inclusive improvements do
not erase these matched launch regressions.

## Complete matched timing table

Cells are **old M16 -> direct M16 microseconds (latency percent change)**.
L = launch-only; B = worklist builder plus launch. C5 is the five-row/K5
fixture, not measured five-client aggregate serving. M64 remains an independent
numerical reference and is retained in the raw log, not substituted for M16.

### C4

| Case | Path | Repeat 1 | Repeat 2 | Repeat 3 |
| --- | --- | --- | --- | --- |
| four-local-four-remote | L | 174.150 -> 173.419 (-0.42%) | 160.391 -> 165.111 (+2.94%) | 159.887 -> 158.468 (-0.89%) |
| four-local-four-remote | B | 208.151 -> 207.366 (-0.38%) | 201.512 -> 200.229 (-0.64%) | 183.755 -> 189.333 (+3.04%) |
| boundaries | L | 173.440 -> 172.934 (-0.29%) | 169.306 -> 162.888 (-3.79%) | 158.904 -> 158.318 (-0.37%) |
| boundaries | B | 209.974 -> 208.891 (-0.52%) | 194.587 -> 193.959 (-0.32%) | 186.197 -> 183.503 (-1.45%) |
| varied-gather | L | 165.812 -> 172.455 (+4.01%) | 165.956 -> 164.004 (-1.18%) | 165.238 -> 158.852 (-3.86%) |
| varied-gather | B | 210.294 -> 201.045 (-4.40%) | 201.247 -> 200.229 (-0.51%) | 190.689 -> 189.844 (-0.44%) |
| empty | L | 4.104 -> 4.063 (-1.00%) | 4.092 -> 4.087 (-0.12%) | 4.073 -> 4.076 (+0.07%) |
| empty | B | 26.642 -> 26.631 (-0.04%) | 26.633 -> 26.630 (-0.01%) | 26.636 -> 26.638 (+0.01%) |
| remote-only | L | 4.093 -> 4.042 (-1.25%) | 4.104 -> 4.087 (-0.41%) | 4.099 -> 4.104 (+0.12%) |
| remote-only | B | 26.652 -> 26.632 (-0.08%) | 26.641 -> 26.632 (-0.03%) | 26.641 -> 26.641 (+0.00%) |
| partial-all-local | L | 166.264 -> 173.222 (+4.18%) | 165.657 -> 164.116 (-0.93%) | 163.420 -> 158.543 (-2.98%) |
| partial-all-local | B | 208.622 -> 199.227 (-4.50%) | 201.448 -> 200.372 (-0.53%) | 190.712 -> 188.812 (-1.00%) |
| population1 | L | 165.495 -> 172.828 (+4.43%) | 159.706 -> 166.233 (+4.09%) | 166.002 -> 161.879 (-2.48%) |
| population1 | B | 201.361 -> 208.890 (+3.74%) | 196.993 -> 197.263 (+0.14%) | 183.538 -> 185.544 (+1.09%) |
| population2 | L | 174.626 -> 169.861 (-2.73%) | 160.722 -> 165.806 (+3.16%) | 162.160 -> 157.972 (-2.58%) |
| population2 | B | 209.544 -> 201.000 (-4.08%) | 201.673 -> 200.584 (-0.54%) | 191.693 -> 190.141 (-0.81%) |
| population3 | L | 170.975 -> 165.107 (-3.43%) | 163.949 -> 164.577 (+0.38%) | 166.109 -> 158.212 (-4.75%) |
| population3 | B | 210.454 -> 206.323 (-1.96%) | 201.504 -> 193.664 (-3.89%) | 186.748 -> 190.639 (+2.08%) |
| population4 | L | 171.491 -> 165.271 (-3.63%) | 161.452 -> 160.935 (-0.32%) | 159.036 -> 158.472 (-0.35%) |
| population4 | B | 211.422 -> 210.181 (-0.59%) | 201.978 -> 201.297 (-0.34%) | 187.953 -> 191.031 (+1.64%) |

### C5

| Case | Path | Repeat 1 | Repeat 2 | Repeat 3 |
| --- | --- | --- | --- | --- |
| four-local-four-remote | L | 166.012 -> 172.331 (+3.81%) | 169.517 -> 168.034 (-0.87%) | 174.468 -> 175.310 (+0.48%) |
| four-local-four-remote | B | 207.024 -> 199.216 (-3.77%) | 201.719 -> 198.037 (-1.83%) | 210.343 -> 209.282 (-0.50%) |
| boundaries | L | 166.054 -> 171.238 (+3.12%) | 166.035 -> 168.566 (+1.52%) | 171.684 -> 173.710 (+1.18%) |
| boundaries | B | 208.588 -> 208.253 (-0.16%) | 200.571 -> 203.657 (+1.54%) | 211.479 -> 207.703 (-1.79%) |
| varied-gather | L | 170.957 -> 168.394 (-1.50%) | 166.631 -> 164.760 (-1.12%) | 176.057 -> 173.532 (-1.43%) |
| varied-gather | B | 208.033 -> 207.882 (-0.07%) | 203.559 -> 203.137 (-0.21%) | 203.457 -> 209.331 (+2.89%) |
| empty | L | 4.100 -> 4.098 (-0.05%) | 4.100 -> 4.105 (+0.12%) | 4.111 -> 4.086 (-0.61%) |
| empty | B | 26.641 -> 26.641 (+0.00%) | 26.644 -> 26.637 (-0.03%) | 26.643 -> 26.639 (-0.02%) |
| remote-only | L | 4.105 -> 4.083 (-0.54%) | 4.106 -> 4.084 (-0.54%) | 4.097 -> 4.095 (-0.05%) |
| remote-only | B | 26.641 -> 26.642 (+0.00%) | 26.639 -> 26.640 (+0.00%) | 26.642 -> 26.639 (-0.01%) |
| partial-all-local | L | 173.919 -> 166.193 (-4.44%) | 165.774 -> 168.232 (+1.48%) | 170.198 -> 176.009 (+3.41%) |
| partial-all-local | B | 206.138 -> 206.021 (-0.06%) | 201.743 -> 200.821 (-0.46%) | 210.283 -> 203.323 (-3.31%) |
| population1 | L | 173.146 -> 168.985 (-2.40%) | 161.773 -> 160.977 (-0.49%) | 174.287 -> 167.958 (-3.63%) |
| population1 | B | 207.550 -> 199.794 (-3.74%) | 199.900 -> 197.147 (-1.38%) | 210.538 -> 209.926 (-0.29%) |
| population2 | L | 170.720 -> 168.241 (-1.45%) | 168.997 -> 160.070 (-5.28%) | 174.936 -> 173.435 (-0.86%) |
| population2 | B | 207.925 -> 200.281 (-3.68%) | 201.254 -> 202.818 (+0.78%) | 211.188 -> 209.957 (-0.58%) |
| population3 | L | 169.773 -> 169.727 (-0.03%) | 161.177 -> 165.131 (+2.45%) | 177.115 -> 176.841 (-0.15%) |
| population3 | B | 200.665 -> 207.748 (+3.53%) | 204.297 -> 203.518 (-0.38%) | 208.562 -> 202.777 (-2.77%) |
| population4 | L | 172.339 -> 168.168 (-2.42%) | 161.306 -> 164.038 (+1.69%) | 174.988 -> 176.322 (+0.76%) |
| population4 | B | 208.295 -> 201.123 (-3.44%) | 203.814 -> 196.626 (-3.53%) | 212.107 -> 210.611 (-0.71%) |

## Gates, mechanism and limitations

Eight numerical executions pass (C4/K5, default FMA/no-FMA, ordinary/memcheck),
with four zero-error memcheck summaries and clean exit0/OOM=false. Every run
checks complete production-M64/current-M16/candidate output pairs, scalar/vector
loaders, graph metadata refresh, CPU columns, populations1..4/5, poisoned
remote/padding outputs, immutable packed/scales and allocation guards. Explicit
guarded device budgets are 38,566,408 and 38,766,376 bytes. Timing runs also
recheck all complete outputs, original weights/scales, metadata and guards.

Ptxas reports old and new at 56 registers and zero spills; shared storage drops
18,112 -> 11,968 bytes. Those resource reductions do not establish occupancy
or throughput improvement. The removed intermediate transpose/barrier is not
shown to dominate; direct fragments retain strided shared-byte reads/packing.

The source still streams the same 32MiB packed B plus 4MiB scales for every
active fixture call through the same 64 K64 stages, independent of expert
population1..5. Similar timings across those populations point toward that
unchanged B movement/load pipeline as a likely major cost. This is an inference,
not measured DRAM saturation: no hardware counters distinguish cache/DRAM,
shared-bank conflicts, or instruction stalls here.

A second concrete infrastructure cost is the unchanged worklist builder:
`kernels/gb10/common/moe_permute.cu:306` runs its entire 288-expert scan and
worklist emission on thread0. Empty/remote launch-only calls cost about4.1us,
while builder-inclusive controls cost about26.64us; active inclusive-minus-
launch means are about33.4us C4 and36.3us K5 for old M16. The differences include
launch/order/cache effects and are not isolated builder profiling. A bounded
parallel-builder experiment is independently plausible, but these eager timings
are not CUDA-graph production cycle timings and cannot promise that saving.

This four-local-pair hot fixture is not a model working set or a topology/C4
serving test. It cannot establish a full-wall TPS improvement, acceptance
stability, a 30 C1 / 60 aggregate C4 result, or the exact model-wide bottleneck.
The retained B-tile integration prerequisites have a stronger standalone
signal, but still require typed ownership and complete reader compatibility.

## Exact source, binaries and raw receipts

Source SHA256:

```text
48dc1c28d6d11b2ec9d045aefaa9040c08deadabdf98bc3f1f97d2798ae3f0c1  glm_moe_m16_direct_register_plan.md
c68a10c9a76f2e2c9b9be3436ebddf164b2a87a9bffae4aeac478a3dacf1c6ad  glm_moe_m16_direct_fragment.h
bbbb1a7da28344ed560a0c4f6bd29681c2eea578868c9443b3a2801ebc12b224  glm_moe_m16_direct_register.cuh
14827a3cb6e5e3f7b1e61edf5e36fbd7583b091a6d88a2d2725e503ffe6ccb2d  bench_glm_moe_m16_direct_register.cu
```

Timing executable SHA256, unchanged across all three invocations per shape:

```text
398db02e2646de2b82bcdaf0aef67f564a8a68e257add24f62327de3ada02b1f  /out/bench-glm-m16-direct-c4
de157ac208996014e4b2fcd26b3428b248d6859afc700750a8d128f3f662558f  /out/bench-glm-m16-direct-c5
```

Persistent receipt directory:
`/home/abc/storage/models/atlas-campaigns/20260908/m16-direct-register/`.
Raw receipt SHA256:

```text
81f4d05d1706e459d2645f6f220e9141dc7650222a445a32357c82124b731a83  native-build.log
3569512da5d95eb7bea2cda4a50335d9922f8737fa79f7016b7ed44ef65a3bcd  native-gates.log
cb40a75d7b354649f7838989421bce943c7da62a228d587ac5da6a81fa67f5eb  native-timing.log
1289e7a6718be45719e6b5ffa7e4656bbbf2d77a73e00bbf1b2956dae8996c68  fragment-red.log
2e07059e6bb14f7ea39ad83f6b67953de0311ac2026657182568b3bb39830b0f  fragment-green.log
8c2461ec06d06073fd9785ede956027511091773947a919e317a04779d988f4e  c4-host.log
4f0ee800ef4f2540e037bbd2cea768257b5f92cc10c9cec88048bbb18cd127ad  k5-host.log
```

