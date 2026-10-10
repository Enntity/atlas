// SPDX-License-Identifier: AGPL-3.0-only

// EXL3-only reference-policy (FLA bit-parity) KDA kernels, module "kda_fla".
// Each kernel mirrors one pinned Triton reference kernel op for op; see
// docs/design/exl3-kda-reference-core.md §3.6a and the byte-exact tests in
// tests/exl3_validation/kda_fla/. Selected only by ATLAS_KDA_FLA_REFERENCE=1.
//
// Prefill launch order (one sequence, T > 1 rows, one stream):
//   kda_fla_conv_prefill -> kda_fla_l2norm -> kda_fla_beta -> kda_fla_gate_cumsum
//   -> (zero A, Aqk) kda_fla_kkt_inter -> kda_fla_kkt_intra -> kda_fla_solve_tril
//   -> kda_fla_recompute_w_u -> kda_fla_delta_h -> kda_fla_output
// Decode (T == 1, matching vLLM's query_len == 1 classification):
//   kda_fla_conv_decode -> kda_fla_recurrent_decode

#include "kda_fla_prep.cuh"
#include "kda_fla_kkt.cuh"
#include "kda_fla_wy.cuh"
#include "kda_fla_state.cuh"
#include "kda_fla_decode.cuh"
