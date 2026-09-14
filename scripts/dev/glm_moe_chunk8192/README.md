<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# GLM 8K chunk canary

Standalone comparison of the unchanged production M64 grouped NVFP4 kernel at
8192 versus 4096+4096 rows, and 8196 versus 4096+4100 rows. This does not enable
8K chunks in the engine or change the startup memory guards.

Each uniform/skew fixture uses top-8 routing across 288 expert IDs, with full
weights for 144 local experts and NULL pointers for the remote half. The gate
shape is N=2048/K=4096 and down is N=4096/K=2048. Expert weights occupy 648 MiB.
The combined proxy is twice the measured gate shape plus the measured down
shape; it excludes routing, packing, communication, and other model work.

The split and whole launches share canonical activation bytes and weights.
Gate inputs use global-token views. Down inputs gather a canonical route-major
buffer using an independently checked expert/global-token permutation. Host
tests establish exact routing equivalence; native tests compare every local
output bit, reject nonfinite outputs, and verify remote poison plus buffer
guards. Five alternating rounds of five launches produce median CUDA timings.
Three warmup pairs precede timing. A split sample includes both launches.

Run the cheap host gate first:

```sh
c++ -std=c++17 -O2 -Wall -Wextra -Werror -DATLAS_HOST_ONLY -x c++ \
  scripts/dev/glm_moe_chunk8192/bench.cu -o /tmp/atlas-moe-chunk8192-host
/tmp/atlas-moe-chunk8192-host --host-test
```

Native CUDA 13 build and run, only during a coordinated idle Spark window:

```sh
nvcc -O3 -std=c++17 -gencode=arch=compute_121a,code=sm_121a --fmad=false \
  -Xcompiler=-ffp-contract=off -Xptxas=-v \
  scripts/dev/glm_moe_chunk8192/bench.cu -o /tmp/atlas-moe-chunk8192
/tmp/atlas-moe-chunk8192 --run 1.15
```

The default minimum weighted gain is 1.15x for every fixture. Exit 3 rejects
the first fixture below that threshold; nonzero oracle failures stop earlier.
Device allocations are capped at 3 GiB, with 9 GiB free required before any
allocation to retain space for host oracle copies and a 4 GiB reserve. Compiler
and run containers should be capped at 8 CPUs and 6 GiB without swap.

## Native result (2026-09-11)

The first uniform 8192 fixture passed exact-bit and nonfinite checks but failed
the 1.15x speed gate. Gate: 8.849748 ms for two 4K chunks versus 8.189664 ms
for 8K (1.081x). Down: 8.484154 versus 8.083155 ms (1.050x). The weighted
proxy was 26.183649 versus 24.462482 ms, or **1.070x**. The harness exited 3
as intended; skew and tail native cases were not run. No 8K engine integration
is justified by this result. Host route checks cover all four fixtures.

Compilation used CUDA 13 and the explicit architecture-specific gencode above.
The generic sm_121 target, including nvcc's generic fallback from `-arch=sm_121a`,
cannot assemble the production block-scaled MMA instructions. Failed builds
are retained as compile receipts, not performance evidence. Tested production
kernel SHA256: `bad2afc59183a4b46fd094864ef88ae428517a23dee0d00f1d9e87de7503b33b`.
The bench SHA256 is `3e0fe4ec78a3c9d2e37e8878f1a0198e5f413bba52cf5990310ad77706638e76`.
Logs are in the research experiment's `prefill-parity/moe-chunk8192-r1/`.
