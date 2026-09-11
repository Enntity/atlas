// SPDX-License-Identifier: AGPL-3.0-only
// Single-launch Sm120 NVFP4 grouped GEMM for Holo MoE Phase-2.
// Replaces the per-expert dense-collective loop (atlas_cutlass_nvfp4_grouped_gate_up)
// with one GemmUniversalMode::kGrouped launch over all active experts.
//
// Style/types mirror the dense binding cutlass_nvfp4_gemm.cu: same Sm120 /
// OpClassBlockScaledTensorOp / nv_float4_t<e2m1> / float_ue4m3_t SF collective,
// same #ifdef arch guard, same extern "C" status-code convention. The only new
// machinery here is the single-launch grouped host assembly (per-group problem
// shapes / pointers / strides / SF-layouts) plus a per-group activation pack into
// the grouped SFA atom.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>
#include <vector>

#include "cute/tensor.hpp"
#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/layout/matrix.h"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

// ─── element + layout aliases (IDENTICAL to dense cutlass_nvfp4_gemm.cu:23-42) ───
using ElementInput = cutlass::float_e2m1_t;
using ElementA = cutlass::nv_float4_t<ElementInput>;
using ElementB = cutlass::nv_float4_t<ElementInput>;
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using ElementSF = cutlass::float_ue4m3_t;
using ElementAccumulator = float;
using ElementCompute = float;

// pointer-to-layout  ⇒  selects GROUPED (IsGroupedGemmKernel)
using GmemLayoutA = cutlass::layout::RowMajor;
using GmemLayoutB = cutlass::layout::ColumnMajor;
using GmemLayoutC = cutlass::layout::RowMajor;  // dense path uses RowMajor C/D; keep it
using GmemLayoutD = cutlass::layout::RowMajor;

constexpr int AlignmentA = 32;  // = 128 / 4   (FP4 elems)
constexpr int AlignmentB = 32;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;  // = 8
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;  // = 8

using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using TileShape = Shape<_128, _128, _128>;  // matches dense ThreadBlockShape
using ClusterShape = Shape<_1, _1, _1>;

// ─── EPILOGUE (plain LinearCombination; beta=0; per-expert scale2 via alpha_ptr) ───
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag,
    OperatorClass,
    TileShape,
    ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator,
    ElementCompute,
    ElementC,
    GmemLayoutC*,  // trailing '*' ⇒ grouped
    AlignmentC,
    ElementD,
    GmemLayoutD*,
    AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

// ─── MAINLOOP (pointer-to-layout ⇒ grouped; all-TMA pingpong) ───
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag,
    OperatorClass,
    ElementA,
    GmemLayoutA*,
    AlignmentA,
    ElementB,
    GmemLayoutB*,
    AlignmentB,
    ElementAccumulator,
    TileShape,
    ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong>::CollectiveOp;
// Fallback if can_implement rejects pingpong on sm_121f:
//   cutlass::gemm::collective::KernelScheduleAuto   (cooperative)

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>,
    CollectiveMainloop,
    CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// per-group internal types (pointer types at the Arguments boundary)
using StrideA = typename Gemm::GemmKernel::InternalStrideA;
using StrideB = typename Gemm::GemmKernel::InternalStrideB;
using StrideC = typename Gemm::GemmKernel::InternalStrideC;
using StrideD = typename Gemm::GemmKernel::InternalStrideD;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
using Sm1xxBlkScaledConfig =
    typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
using ProblemShape = cute::Shape<int, int, int>;

static inline size_t align_up_(size_t x, size_t a) {
  return (x + a - 1) & ~(a - 1);
}


struct GroupedAPrep {
  int status = 0;                          // non-zero on failure
  int G = 0;                               // active groups
  std::vector<ProblemShape> host_shapes;   // {m_e, n, k} per group
  std::vector<int> ms;                     // sorted-row start per group (C offset)
  std::vector<int> me;                     // m_e per group
  std::vector<int> eidx;                   // expert index per group (B/scale2 lookup)
  ProblemShape* dShapes = nullptr;
  const ElementA::DataType** dA = nullptr;
  const ElementSF** dSFA = nullptr;
  StrideA* dsA = nullptr;
  LayoutSFA* dlSFA = nullptr;
  size_t cursor = 0;                       // free ws offset after the A-side arrays
};

static int launch_projection(
    const GroupedAPrep& a,
    const unsigned long long* packed_ptrs,
    const unsigned long long* sfb_ptrs,
    const float* scale2_vals,
    __nv_bfloat16* C_bf16,
    int n,
    int k,
    unsigned char* ws,
    size_t cursor_start,
    size_t workspace_size,
    cudaStream_t stream,
    int tag, bool dry_run, size_t* required) {
  int G = a.G;
  if (G == 0) {
    return 0;
  }
  std::vector<const ElementB::DataType*> hB(G);
  std::vector<const ElementSF*> hSFB(G);
  std::vector<const ElementC*> hC(G);
  std::vector<ElementD*> hD(G);
  std::vector<StrideB> sB(G);
  std::vector<StrideC> sC(G);
  std::vector<StrideD> sD(G);
  std::vector<LayoutSFB> lSFB(G);
  std::vector<float> alpha_host(G);
  for (int g = 0; g < G; ++g) {
    int e = a.eidx[g];
    int m_e = a.me[g];
    size_t ms = (size_t)a.ms[g];
    hB[g] = reinterpret_cast<const ElementB::DataType*>(packed_ptrs[e]);
    hSFB[g] = reinterpret_cast<const ElementSF*>(sfb_ptrs[e]);
    hC[g] = reinterpret_cast<const ElementC*>(C_bf16 + ms * n);
    hD[g] = reinterpret_cast<ElementD*>(C_bf16 + ms * n);
    sB[g] = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
    sC[g] = cutlass::make_cute_packed_stride(StrideC{}, {m_e, n, 1});
    sD[g] = cutlass::make_cute_packed_stride(StrideD{}, {m_e, n, 1});
    lSFB[g] =
        Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m_e, n, k, 1));
    alpha_host[g] = scale2_vals[e];
  }

  size_t cursor = cursor_start;
  auto put = [&](const void* src, size_t bytes) -> void* {
    if (cursor > workspace_size || bytes > workspace_size - cursor) { std::fprintf(stderr, "metadata capacity exceeded\n"); std::abort(); }
    void* dst = ws + cursor;
    cursor = align_up_(cursor + bytes, 256);
    if (!dry_run) CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream));
    return dst;
  };
  auto* dB = (const ElementB::DataType**)put(hB.data(), G * sizeof(void*));
  auto* dSFB = (const ElementSF**)put(hSFB.data(), G * sizeof(void*));
  auto* dC = (const ElementC**)put(hC.data(), G * sizeof(void*));
  auto* dD = (ElementD**)put(hD.data(), G * sizeof(void*));
  auto* dsB = (StrideB*)put(sB.data(), G * sizeof(StrideB));
  auto* dsC = (StrideC*)put(sC.data(), G * sizeof(StrideC));
  auto* dsD = (StrideD*)put(sD.data(), G * sizeof(StrideD));
  auto* dlSFB = (LayoutSFB*)put(lSFB.data(), G * sizeof(LayoutSFB));
  auto* dAlpha = (float*)put(alpha_host.data(), G * sizeof(float));
  // Per-group alpha (scale2) needs alpha_ptr_array (G POINTERS, one per group →
  // &dAlpha[g]); the scalar alpha_ptr would apply dAlpha[0] to every group.
  std::vector<const float*> hAlphaPtr(G);
  for (int g = 0; g < G; ++g) {
    hAlphaPtr[g] = dAlpha + g;
  }
  auto* dAlphaPtr = (const float**)put(hAlphaPtr.data(), G * sizeof(const float*));

  cutlass::KernelHardwareInfo hw{};
  hw.device_id = 0;
  hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);

  typename Gemm::GemmKernel::CollectiveMainloop::Arguments mainloop_args{
      a.dA, a.dsA, dB, dsB, a.dSFA, a.dlSFA, dSFB, dlSFB};

  typename Gemm::GemmKernel::CollectiveEpilogue::Arguments epi_args{};
  epi_args.thread.alpha = 1.0f;
  epi_args.thread.beta = 0.0f;
  epi_args.thread.alpha_ptr_array = dAlphaPtr;
  epi_args.ptr_C = dC;
  epi_args.dC = dsC;
  epi_args.ptr_D = dD;
  epi_args.dD = dsD;

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {G, a.dShapes, const_cast<ProblemShape*>(a.host_shapes.data())},
      mainloop_args,
      epi_args,
      hw};

  Gemm gemm;
  size_t need = Gemm::get_workspace_size(args);
  if (required) *required = cursor + need;
  if (cursor > workspace_size || need > workspace_size - cursor) {
    return -2;
  }
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) {
    return tag + (-50);
  }
  if (dry_run) return 0;
  cutlass::Status st = gemm.initialize(args, ws + cursor, stream);
  if (st != cutlass::Status::kSuccess) {
    return tag + static_cast<int>(st);
  }
  st = gemm.run(stream);
  return st == cutlass::Status::kSuccess ? 0 : tag + static_cast<int>(st);
}

#endif
