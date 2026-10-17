const brief = '/home/colum/Workspace/RustroverProjects/atlas/prefill-kernel-brief-20261016.md';
const findings = '/home/colum/Workspace/RustroverProjects/atlas/prefill-investigation-20261016.md';
const repo = '/home/colum/Workspace/RustroverProjects/atlas-perf';

const exl3Task = 'Read-only analysis (do not edit any file). Repository (read-only): ' + repo + '\n' +
'Required reading first: ' + brief + ' then ' + findings + '.\n\n' +
'Goal: a rigorous, evidence-cited audit of the EXL3 packed-matmul kernel family, which is 96.8% of Atlas prefill GPU time (measured: 43.9 s MoE + 38.3 s dense in an 84.9 s trace for a 1024-token prefill).\n\n' +
'Cover EVERY entry point (enumerate them all, do not sample) in:\n' +
'- kernels/gb10/common/exl3_linear_mma.cu\n' +
'- kernels/gb10/common/exl3_moe_mma.cu\n' +
'- kernels/gb10/common/exl3_moe_precision.cu\n' +
'- kernels/gb10/common/exl3_common.cuh, exl3_expert.cuh, exl3_moe_thin.cuh, exl3_mma.cuh, exl3_mma_split.cuh\n' +
'- kernels/gb10/common/exl3_coop_gemm.cu (every __global__ in it)\n' +
'- kernels/gb10/common/exllamav3_gemm_inner.cuh and exllamav3_kernel_map.cuh\n' +
'and the Rust that selects them:\n' +
'- crates/spark-runtime/src/weights/exl3/native.rs, native/coop.rs, native/numerics.rs\n' +
'- crates/spark-model/src/layers/glm5_exl3_ffn/moe/coop.rs, moe/forward.rs, moe/project.rs\n\n' +
'For each kernel function output a table row: name | what it computes | grid/block | weight+activation bytes moved per output FLOP | are weight bytes re-read per M row/tile (yes/no, factor) | dominant instruction cost | share of prefill.\n\n' +
'Then give ranked, concrete proposals by expected prefill win. You MUST address:\n' +
'1. The M=1-per-CTA structure: exl3_mma.cuh states "One actual row is padded to M16; only lane<4 owns that row\'s C0/C1 outputs", and grid.y = rows (or rows*topk). Quantify: fraction of MMA rows that are padding, how many times the packed B fragments are decoded/loaded per output element, and what M-tiling to 16/32/64/128 rows would recover. Give the arithmetic.\n' +
'2. The EXL3 trellis decode: read tile_state/mul1 in exl3_common.cuh. Count the instructions per decoded weight, and determine whether the same packed 16x16 tile words are re-read per (i,j) lane (memory vs L1 vs shared vs registers). Propose concrete fixes: shared-memory staged tiles, precomputed decode LUTs, warp-cooperative decode, reordering for reuse. State the expected bytes/instruction change.\n' +
'3. Whether exllamav3_gemm_inner.cuh (static_assert TILESIZE_M == 16) can support larger M and exactly what must change (shared-memory staging plan vs SMEM_MAX = 92160 B, fragment maps, split-K fold, lock protocol) without breaking bit-identity.\n' +
'4. The fat-vs-thin numerics split (thin rounds each k-fold partial to FP16; fat accumulates unrounded F32) and how a grouped multi-row fat kernel preserves it per row. Cite exl3_moe_mma.cu and moe/coop.rs.\n' +
'5. Redundant work: repeated hadamard launches, f32_to_f16 round trips, lock/atomic protocols, oversized grid dimensions.\n\n' +
'Rules: cite file:line for every claim. Quantify everything you can. Do NOT propose changes that alter per-row numerics without stating the exact bit-level difference and the flag that would gate it. Return one markdown report.';

const otherTask = 'Read-only analysis (do not edit any file). Repository (read-only): ' + repo + '\n' +
'Required reading first: ' + brief + ' then ' + findings + '.\n\n' +
'Context: in the current prefill trace each of these kernel families is under 1% of GPU time, so rank proposals as "matters once the packed matmuls are 5-10x faster". GB10 is a memory-throughput-constrained device: flag every redundant pass over activations, every unfused elementwise chain, and every kernel whose achievable bandwidth is far below the device roofline.\n\n' +
'Cover EVERY prefill kernel function in these families and report per kernel: name | what it does | grid/block | bytes moved | achieved vs theoretical bandwidth (or a reasoned estimate) | concrete micro-optimization.\n' +
'- KDA / SSM prefill: every kernel named kda_* and kda_fla_* (search kernels/gb10/common/ and kernels/gb10/glm-5.3-flash-exl3/), including kda_fla_conv_prefill, kda_fla_l2norm, kda_fla_beta, kda_fla_gate_cumsum, kda_fla_kkt_inter, kda_fla_kkt_intra, kda_fla_solve_tril, kda_fla_recompute_w_u, kda_fla_delta_h, kda_fla_output, kda_pack_qkv, kda_sigmoid_gated_rms_norm.\n' +
'- Sparse MLA and indexer: glm_sparse_mla_prefill_*, glm_index_*, mla_batched_gemv, grouped_gemm_mla, inferspark_prefill_paged_512, mla_cache_assemble_batched, glm_latent_*.\n' +
'- mHC: exl3_mhc_pre_norm, exl3_mhc_post_unrounded, exl3_mhc_contract.\n' +
'- Hadamard / activation / norm / elementwise: exl3_input_hadamard_f16, exl3_output_hadamard, exl3_silu_mul_dense, rms_norm_vanilla, ssm_state_clamp_norm_fused, bf16_add_inplace, bf16_to_f32.\n' +
'Also flag any kernel launched in the prefill path that is a pure memory copy or trivially fusible neighbour.\n\n' +
'Rules: cite file:line for every claim. Be quantitative. Rank by expected win once the packed matmuls are fixed. Return one markdown report.';

const outcomes = await runs.all([
  { key: 'exl3-audit', agent: 'reviewer', task: exl3Task },
  { key: 'other-kernels-audit', agent: 'reviewer', task: otherTask },
]);
return { exl3Audit: outcomes[0].output, otherKernelsAudit: outcomes[1].output };
