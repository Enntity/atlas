# Native-source B-tile permutation prerequisite

Scope: new standalone files in `scripts/dev` only. Root alone owns commits,
native builds, node/GPU operations and any future model repacking. This task
does not authorize production weight mutation, loader or dispatch integration.
No timing or arithmetic/activation-precision change is proposed.

## Inspected production contract

`QuantizedWeight::transpose_for_gemm_gs` starts with packed `[N,K/2]` and scale
`[N,K/16]` for native NVFP4; `transpose_u8` produces `[K/2,N]` / `[K/16,N]`.
The original scalar `weight_scale_2` and pointer metadata are carried unchanged.
Current B-tile fixtures instead begin with those already-transposed packed
bytes. A production in-place replacement needs its own native-source oracle.

Exact scoped geometry is N2048/K4096, group16, packed4MiB and scales512KiB.
Destination B-tile order remains `[N/128,K/64,128,32]`. At destination
`(nt,kt,n,b)`, the native byte is `(nt*128+n)*(K/2)+kt*32+b`.
No nibble unpacking/swapping, scale interpretation or floating-point conversion.

## Proposed files and sequence (root review before implementation)

1. New `glm_moe_btile_native_layout.h`: checked exact geometry plus shared
   host/device native-source-index and inverse index helpers. Add an actual
   CPU test against the initially incorrect identity source helper, observe
   RED, then implement the byte mapping. Full packed-byte bijection/inverse,
   native-to-T-to-Btile equivalence and scale-transpose bijection are required.
2. New `glm_moe_btile_native_repack.cuh`: a bounded native-source gather into
   the original packed destination. Byte-only CUDA body, consuming scratch;
   no production symbols or build integration. Use the actual existing
   `kernels/gb10/common/transpose_u8.cu` for scales in the standalone fixture.
3. New `bench_glm_moe_btile_native_repack.cu`: host-only CPU mode plus native
   CUDA mode. One original packed allocation and one original scale allocation
   retain their addresses. Copy original packed to one reusable4MiB scratch,
   launch native-to-tile from scratch back into packed, synchronize before
   reusing scratch. Copy original scales to scratch, transpose back into the
   original scales, synchronize again. Scalar bits and owner addresses remain
   unchanged; allocation count remains fixed across the repack transaction.
4. Compare EVERY resulting packed/scaled byte to independent host references.
   Construct the packed reference through the existing T-to-tile formula, with
   a separate explicit native-to-T transpose; compare it against the new native
   mapping. Exercise three deterministic byte patterns, including all nibble
   and scale byte values. No numerical tolerance or BF16 arithmetic.
5. Fresh byte poison in destination staging/reference construction; immutable
   separate native GPU reference copies, guards on all GPU allocations, and
   chunked full readback. After scale reuse, verify scratch prefix is the exact
   native scale bytes and its untouched tail still equals native packed bytes.
   Full source/destination extents, disjoint spans, exact geometry and scratch
   capacity validate before any CUDA copy/launch. Reject same/overlapping spans
   rather than claiming an unsafe in-place kernel is valid.

Byte data has no invalid sentinel. Native fixture therefore repeats each of the
three patterns with complementary destination poisons0xa5 and0x5a, applied only
after copying the original source into scratch on the same stream. A skipped
write cannot equal both expected outputs. These extra memsets are diagnostic
fixture work, not proposed production loader work. Six transactions share the
same five allocations, stream and owner addresses; no timing is performed.

## Explicit bounded memory

Five GPU allocations: packed4MiB, scales512KiB, scratch4MiB, immutable native
packed reference4MiB and immutable native scales reference512KiB. Each has
128-byte guards on both sides. Exact explicit device bytes: **13,632,768**,
below64MiB; assert allocation accounting before and after each transaction.

Host buffers are allocated once: native packed4MiB/scales512KiB, transposed
packed4MiB/scales512KiB, expected tiled packed4MiB, packed seen-map4MiB,
scale seen-map512KiB, and a64KiB reusable D2H chunk. Exact host buffer payload:
**18,415,616 bytes**, below32MiB. Guard readback adds a fixed256-byte stack
buffer; small descriptors/counters are fixed-size. No other tensor-sized
host/device temporary, runtime library bookkeeping excluded explicitly.

## Required evidence

Persist CPU RED/GREEN and exact budgets under
`/home/abc/storage/models/atlas-campaigns/20260908/btile-native-repack/`.
Independent source review and frozen hashes precede root-owned native CUDA
compilation, ordinary runs and memcheck (both compiler FMA policies, although
the new permutation has no FP arithmetic). No performance/model claims follow
from these byte-permutation gates; no actual model repacking in this task.
