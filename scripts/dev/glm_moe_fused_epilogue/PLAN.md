# One-screen gate/up + SiLU/FP4 fusion

Keep the M64/N128/K64 MMA body and FP32 accumulation order unchanged. Run gate
then up in one CTA, retaining the gate's BF16-rounded outputs in32 packed
register words per lane. After up, consume corresponding gate values, preserve
the existing clamp/sigmoid/BF16-rounding arithmetic, reduce each16-value scale
over four adjacent lanes and write packed down input directly. No production
selector changes until a standalone chained gate passes.

Expected tradeoff: remove BF16 intermediate stores, reads and staged copy at the
cost of32 additional live registers and reduced occupancy. Register count alone
is not a speed prediction. Whole chained comparison against current gate,up,
SiLU/quant,copy,down determines acceptance. Require1.2x on4096 uniform full144
local-expert fixture before skew and4100 tail. Compare all locally owned packed
bytes, scales and final BF16 output bits; preserve remote-row poisons and guards.
A failure rejects this candidate without loading the full model again.

## Native disposition

Original candidate passed18,840,960 locally owned packed/scale bytes and
66,990,080 downstream BF16 elements exactly on4096 uniform routing. Full144
local/288 global experts, top8; fixture2,829MiB. Median chain14.890208ms ->
13.081600ms (1.138256x), below1.2x gate. Skew/tail fixtures skipped.127 ->255
registers/thread,4 ->2 resident CTAs,23,296 shared bytes, no spills. No model
integration. Timing excludes common input prequantization but includes both
projections, activation/packing, staging copy and down projection.

One shared-staging structural followup kept the same shared allocation and
MMA/rounding arithmetic, but still compiled at255 registers with zero spills.
Rejected at compile gate, without GPU timing. Both candidates remain
standalone-only. Native build uses CUDA13.0, O3, fmad=false and explicit
compute_121a/code=sm_121a gencode. DeepSeek drafted epilogue transformations;
parent review corrected a bogus builtin in the first draft before execution.

Source provenance: MMA body adapted from Atlas dd0ffd157fb7c6e96d2f3858c888513c6f142cf6,
original imported fork https://github.com/Mango-kid/atlas at90b3584 (see
repository provenance). Generator pins the exact source hash. Harness derives
from scripts/dev/glm_moe_m128/bench.cu. Epilogue packing/layout and shared
staging are newly written for this rejected experiment; retain AGPL notices.
