# Eight-warp QK standalone candidate

No production source changes. Generated from the current r7/r8 bank-padded
K=V sparse attention implementation, with its source SHA256 pinned.

Each of eight warps handles one 16-head x8-key QK tile instead of two warps
handling four tiles each. Every individual score follows the identical
32-step K16 MMA accumulation chain. A separate 4096-byte FP32 shared exchange
uses `[producer_warp][component][lane]` order, so every store/reload is bank
conflict free. After one CTA barrier, original warps0/1 reload the original
`acc_s[4][4]` fragments. Scaling, masking, 32-key online softmax, BF16
probability casts, PV MMA, next-cache loads, and output conversion remain
unchanged. The generator asserts the entire post-QK suffix is identical.

Dynamic shared memory rises from69,376 to73,472 bytes. The current baseline
already fits only one such CTA per GB10 SM by shared-memory capacity; native
register/occupancy reports determine whether another limit changes.

```sh
python3 scripts/dev/glm_sparse_qk8/generate.py
nvcc -O3 -std=c++17 -arch=sm_121a --fmad=false -Xptxas=-v \
  scripts/dev/glm_sparse_qk8/bench.cu -o /tmp/atlas-sparse-qk8
# Only after the coordinator grants an idle GPU window:
/tmp/atlas-sparse-qk8
```

The CPU generator proves unique coverage of all1024 scores, producer/reloader
fragment identity, and conflict-free exchange addresses. Native compilation
has not yet run. The benchmark includes allocation guards, distinct output
poison, finite checks, and full BF16 bit identity against the actual current
padded-KV kernel. Small cases also compare to independent FP64 attention,
including widths0/1/19/257/2051, partial heads1/13/33, all-masked rows,
internal holes, duplicate indices, reversed physical pages, and two scales.

Timing uses five paired alternating whole-kernel event trials at2048 and4096
rows,32heads, actual2051 index stride (2048 selected keys plus three masked
tail slots), and16K cache. Either case below1.5x stops with exit3. After both
pass, a4100-row masked case checks the actual alignment tail. No isolated QK
speedup substitutes for the whole-kernel gate. A5GiB free-memory guard and
1GiB fixture allocation ceiling retain at least4GiB reserve. Production
integration is explicitly pending numerical/performance results.

## Native result: rejected

CUDA compilation passed: baseline126 registers, candidate114, no spills,
oneCTA/SM both. All six small cases and all33,554,432 output values at2048
rows matched the current production baseline bit for bit. Whole-kernel time
was15.984480ms baseline versus16.436096ms candidate (0.973x). Exit3 stopped
before4096/4100 as designed. This candidate is not integrated.
The experiment's `prefill-parity/qk8-receipt.json` and logs retain evidence.
