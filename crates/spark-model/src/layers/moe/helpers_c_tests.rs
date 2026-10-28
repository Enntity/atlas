// SPDX-License-Identifier: AGPL-3.0-only

use super::{RouterPrefillGemm::*, glm_router_prefill_gemm as gemm};

#[test]
fn glm_router_prefill_cutlass_needs_both_flags_and_unsplit_chunks() {
    // Both flags: only chunks cuBLASLt runs without split-K move.
    for m in [6144, 8196, 16392] {
        assert_eq!(gemm(true, m, false, true, true), Some(Cutlass));
    }
    for m in [65, 300, 2048, 3692, 4096, 6143] {
        assert_eq!(gemm(true, m, false, true, true), Some(CublasLt));
    }
    // The CUTLASS flag alone changes nothing; nor does it without cuBLAS.
    assert_eq!(gemm(true, 8196, false, true, false), Some(CublasLt));
    assert_eq!(gemm(true, 8196, false, false, true), None);
    // Other models, decode/verify widths and graph capture keep BN32.
    assert_eq!(gemm(false, 8196, false, true, true), None);
    assert_eq!(gemm(true, 64, false, true, true), None);
    assert_eq!(gemm(true, 8196, true, true, true), None);
}
