# Promoted B-tile CUDA: bounded native gate results

2026-09-08. The promoted CUDA implementations pass the actual compiled-PTX
signature gate and all ten standalone numerical/memory-check combinations.
This is **not layout activation, model validation or a serving speed result**.
The checked Rust family, owning publication and complete serving-reader
integration remain separate work.

## Source and build boundary

CUDA source is commit `3332c36e11189cb010a81f2aa1cabb54c2103814`.
Six existing prototype helpers were moved into the GLM-selected
`deepseek-v4-flash/nvfp4` source tree with unchanged mathematical bodies;
the old script paths are include shims. Two new translation units and the
existing grouped translation unit expose the family. No common kernel,
model selection, serving flag, loader or Rust dispatch changed in that commit.
See the [promotion plan](../../../scripts/dev/glm_moe_btile_cuda_promotion_plan.md)
and [remaining integration plan](phase6-btile-integration-plan.md).

Native build finished `exit=0 oom=false`. The persistent build recipe compiles
five standalone fixtures for each of explicit `--fmad=false` and compiler-default
FMA policy, targeting `sm_121a`. The three production translation units were
also compiled separately to PTX with `--fmad=false --Werror all-warnings`.
The standalone combined legacy includes emitted `s_act` linkage warnings;
the successful build is not a warning-free claim for those fixtures.

The first real PTX gate safely refused NVCC's valid
`.param .u64 .ptr .align 1 name` syntax before GPU tests started.
Python-only commit `aa88a9b004c587205729e874987eb62944722852` added narrowly
validated pointer annotations without changing CUDA or artifacts. Actual-artifact
and literal-fixture RED preceded the correction; the complete 12-test checker
suite then passed, retaining arity/type/width/duplicate/visibility/target and
malformed-annotation refusal. No native artifact was rewritten to pass.

## Actual ABI evidence

The strict checker accepted three distinct actual PTX artifacts, exact
`sm_121a`/64-bit metadata and every required visible definition. Independent
read-only revalidation matched their bytes, SHA256 and complete ordered
signatures to `native-ptx-verified.json`.

| Production family | Required exports | Ordered parameter count |
|---|---:|---|
| Grouped M16/M64 | 8 | Four fused: 18; two dense: 11; two separate compact: 14 |
| Word/vector BF16 decode, rows 1/2/3 | 6 | 21, including two FP32 shared scalars |
| Native packed-to-B-tile permutation | 1 | 4: two pointers, two u32 |

PTX SHA256 (the signature gate is not itself a numerical oracle):

- Grouped, 1,942,130 bytes: `deb8f2ff2ced3c549b14795f71588163309a5e720f593a6b60d505a6cab60d40`.
- Decode, 248,387 bytes: `ee283c1dcbae8693cd6a5e29b8d88d32404cb7c8a55800ddcad45d34b43900c9`.
- Repack, 1,995 bytes: `ac1007e6964214943994705a9412b4d26996f8b429e7556684774fb3dc3b9fe4`.

## Standalone GPU evidence

Root ran the fixtures sequentially with models/builders stopped. Each of the
five fixtures ran once normally and once under compute-sanitizer memcheck,
for each of the two FMA policies: **20 completed fixture executions and ten
memchecks, all zero errors**. The gate container ended `exit=0 oom=false`.
All numerical comparisons are within the same build policy, not a claim of
cross-policy bit identity.

| Fixture | Explicit device bytes | Evidence per normal or memcheck execution |
|---|---:|---|
| Native permutation | 13,632,768 | Three patterns × two poison values; every packed/scale byte, independent native→T→tile reference, inverse, scratch reuse and guards |
| Register decode | 45,799,520 | Rows 1/2/3 × eight route/shared cases × eager/graph; word/vector full-output bit identity, CPU columns, immutable inputs and canaries |
| M16 C4 | 36,474,632 | Seven routing/tail/empty/remote cases × eager/graph; both outputs bit-identical to original-layout reference, CPU gather columns and guards |
| M16 K5 | 36,674,600 | Same seven-case eager/graph matrix at five rows |
| M64 arena envelope | 57,211,784 | 27 cases × gathered/route-major × eager/graph; dense/separate-compact/fused and both scale policies; full outputs, CPU columns, zero-source and guards |

All fixtures are below the explicit 64 MiB device-allocation cap; this does
not include CUDA context overhead. Native permutation additionally tracks
18,415,616 host payload bytes and exactly five guarded GPU allocations.
Routed fixtures use two local gate/up pairs, not all resident model experts.

The M64 cases include rows
1/2/3/4/5/15/16/17/63/64/65/127/128/129/130/148/255/256/257/1023/1024/
1025/1028/1087/1088 plus empty and remote-only cases. The fixture uses 1,088
source rows, 1,152 route capacity and 544 worklist items; twelve fixed-pointer
graphs cover all ABI/scale/gather combinations with refreshed metadata.
These are bounded synthetic-layout correctness tests, not exhaustive real
288-expert routing, loader ownership or model graph-refresh proof.

## Persistent receipts and limits

Raw files live outside the repository under
`/home/abc/storage/models/atlas-campaigns/20260908/btile-cuda-promotion/`.
All 857 gate-log lines were checked for ten binary sections, two final PASS
records per section, exactly one zero-error memcheck per section and the
successful terminal status. Executable SHA256 values for all ten binaries
are recorded at the start of their corresponding sections.

| Receipt | SHA256 |
|---|---|
| `native-build.log` | `e02a9b947a708d2ffc6e0d9a393ae960eacb385fa6b8179d5c259e1c48de4828` |
| `native-gates.log` | `0e8e4c4fc49f42fa782aaca755d429b1372a8d436828953c7d76494e3f12a193` |
| `native-ptx-verified.json` | `7f1b7ff695dd1a347136e5cf4e8e1037635ce51697fafaa25d658332c60c1b6f` |

The campaign parent contains `compile-btile-promoted.sh` (SHA256
`9548f33357955bc342717e91c4cc4cf5585d4339e0450d485ea3d993a2dfa0cd`)
and `gate-btile-promoted.sh` (SHA256
`ebea7c414fce9fc9300f23499287c118c638fee06dfeef9ed85f3fa0ba25fc85`).
The gate recipe uses fail-fast shell execution, 180-second ordinary and
300-second memcheck timeouts, and memcheck error exit code 99.

No model weights were converted in this gate window. No serving throughput
was measured or improved by registration alone. Prior prototype timings in
[phase6-results.md](phase6-results.md) remain separate evidence; these reruns
do not select a word/vector winner or establish the 30 C1 / 60 C4 target.
