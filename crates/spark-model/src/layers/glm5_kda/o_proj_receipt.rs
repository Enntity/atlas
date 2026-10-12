// SPDX-License-Identifier: AGPL-3.0-only
//! Diagnostic GEMM receipt for the local o_proj projection (experiment C1).

use super::projection::ProjectionPath;
use super::*;

/// Diagnostic dispatch identity for the local o_proj GEMM, recorded
/// beside the `atlas-local-o-proj-binding-v1` record (experiment C1).
/// Pure mirror of the `project_hot_cast`/`project_hot_verify` dispatch;
/// launches nothing and changes no arithmetic. Grid/block reuse
/// `ops::w4a16_gemm_launch_dims`, never a copied formula.
#[allow(clippy::too_many_arguments)]
pub(super) fn o_proj_gemm_receipt(
    verify_entry: bool,
    decode: bool,
    tokens: u32,
    exl3: bool,
    has_transposed: bool,
    fp8_prefill_eligible: bool,
    gemm_k: KernelHandle,
    gemv_k: KernelHandle,
    gemm_t_m128_k: KernelHandle,
    n: u32,
    k: u32,
) -> serde_json::Value {
    let dispatch = if exl3 {
        serde_json::json!({
            "provider": "exl3-native",
            "dispatch_path": "project_exl3 -> Exl3LinearWorkspace::launch_local_bf16 (rank-local; the layer owns the TP reduce)",
            "kernel_handle": null, "kernel_module": null, "kernel_symbol": null,
            "grid": null, "block": null,
        })
    } else if verify_entry && tokens >= 2 {
        serde_json::json!({
            "provider": "atlas-nvfp4",
            "dispatch_path": "project_hot_verify -> w4a16 multi-decode batch tiers (kernel varies with the row count)",
            "kernel_handle": null, "kernel_module": null, "kernel_symbol": null,
            "grid": null, "block": null,
        })
    } else {
        // The verify entry falls back to `project_hot(..., decode=false)`;
        // the production entry dispatches with its own decode flag.
        let effective_decode = !verify_entry && decode;
        if !effective_decode && fp8_prefill_eligible {
            serde_json::json!({
                "provider": "atlas-lt-fp8",
                "dispatch_path": "project_hot_cast transient E4M3 Lt FP8",
                "kernel_handle": null, "kernel_module": null, "kernel_symbol": null,
                "grid": null, "block": null,
            })
        } else {
            match ProjectionPath::for_forward(effective_decode, has_transposed, tokens) {
                ProjectionPath::DecodeGemv => serde_json::json!({
                    "provider": "atlas-nvfp4",
                    "dispatch_path": "project_hot -> w4a16_decode_gemv",
                    "kernel_handle": gemv_k.0,
                    "kernel_module": "w4a16_gemv", "kernel_symbol": "w4a16_gemv",
                    "grid": null, "block": null,
                }),
                ProjectionPath::PrefillTransposedM128 => serde_json::json!({
                    "provider": "atlas-nvfp4",
                    "dispatch_path": "project_hot -> w4a16_gemm_n128_m128 (transposed prefill weight)",
                    "kernel_handle": gemm_t_m128_k.0,
                    "kernel_module": "w4a16", "kernel_symbol": "w4a16_gemm_t_m128",
                    "grid": null, "block": null,
                }),
                ProjectionPath::PrefillBase => {
                    let (grid, block) = ops::w4a16_gemm_launch_dims(tokens, n);
                    serde_json::json!({
                        "provider": "atlas-nvfp4",
                        "dispatch_path": "project_hot -> w4a16_gemm",
                        "kernel_handle": gemm_k.0,
                        "kernel_module": "w4a16", "kernel_symbol": "w4a16_gemm",
                        "grid": grid, "block": block,
                    })
                }
            }
        }
    };
    serde_json::json!({
        "entry": if verify_entry { "project_hot_verify" } else { "project_hot" },
        "decode": decode,
        "tokens": tokens,
        "fp8_prefill_eligible": fp8_prefill_eligible,
        "output_features": n,
        "input_features": k,
        "dispatch": dispatch,
        "placement": {
            "tp_pre_reduce": true,
            "capture_point": "local o_proj GEMM output before the TP reduce",
            "reduce_site": "forward_attention_rows TP reduce (all_reduce_async / reduce_scatter / fused peer exchange)",
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight_map::QuantizedWeight;
    use spark_runtime::gpu::mock::MockGpuBackend;

    fn receipt(
        verify: bool,
        decode: bool,
        tokens: u32,
        exl3: bool,
        has_t: bool,
        fp8: bool,
    ) -> serde_json::Value {
        o_proj_gemm_receipt(
            verify,
            decode,
            tokens,
            exl3,
            has_t,
            fp8,
            KernelHandle(7),
            KernelHandle(8),
            KernelHandle(9),
            4096,
            4096,
        )
    }

    #[test]
    fn exl3_residency_reports_the_native_rank_local_provider() {
        let value = receipt(false, false, 10, true, true, false);
        assert_eq!(value["dispatch"]["provider"], "exl3-native");
        assert!(
            value["dispatch"]["dispatch_path"]
                .as_str()
                .unwrap()
                .contains("launch_local_bf16")
        );
        assert_eq!(value["dispatch"]["kernel_handle"], serde_json::Value::Null);
        assert_eq!(value["entry"], "project_hot");
        assert_eq!(value["placement"]["tp_pre_reduce"], true);
        assert_eq!(value["tokens"], 10);
        assert_eq!(value["fp8_prefill_eligible"], false);
    }

    #[test]
    fn nvfp4_dispatch_mirrors_projection_path_with_shared_launch_dims() {
        // Prefill base: the receipt must agree with the real launcher's
        // grid/block rather than a copied formula.
        let value = receipt(false, false, 10, false, true, false);
        assert_eq!(value["dispatch"]["provider"], "atlas-nvfp4");
        assert_eq!(value["dispatch"]["kernel_handle"], 7);
        assert_eq!(value["dispatch"]["kernel_module"], "w4a16");
        assert_eq!(value["dispatch"]["kernel_symbol"], "w4a16_gemm");
        let gpu = MockGpuBackend::new();
        let ptr = gpu.alloc(16).unwrap();
        let output = gpu.alloc(16).unwrap();
        ops::w4a16_gemm(
            &gpu,
            KernelHandle(7),
            ptr,
            &QuantizedWeight::null(),
            output,
            10,
            4096,
            4096,
            8,
        )
        .unwrap();
        let launches = gpu.launches_snapshot();
        assert_eq!(launches.len(), 1);
        assert_eq!(
            value["dispatch"]["grid"],
            serde_json::json!(launches[0].grid)
        );
        assert_eq!(
            value["dispatch"]["block"],
            serde_json::json!(launches[0].block)
        );
        gpu.free(ptr).unwrap();
        gpu.free(output).unwrap();
        // Decode and transposed-prefill arms name their own kernels.
        let decode = receipt(false, true, 1, false, true, false);
        assert_eq!(decode["dispatch"]["kernel_symbol"], "w4a16_gemv");
        assert_eq!(decode["dispatch"]["kernel_handle"], 8);
        assert_eq!(decode["dispatch"]["grid"], serde_json::Value::Null);
        let transposed = receipt(false, false, 129, false, true, false);
        assert_eq!(transposed["dispatch"]["kernel_symbol"], "w4a16_gemm_t_m128");
        assert_eq!(transposed["dispatch"]["kernel_handle"], 9);
        // The transient FP8 arm supersedes the NVFP4 path when eligible.
        let fp8 = receipt(false, false, 2048, false, true, true);
        assert_eq!(fp8["dispatch"]["provider"], "atlas-lt-fp8");
        assert_eq!(fp8["fp8_prefill_eligible"], true);
    }

    #[test]
    fn verify_entry_reports_batch_tiers_and_decode_flag() {
        let verify = receipt(true, false, 10, false, true, false);
        assert_eq!(verify["entry"], "project_hot_verify");
        assert!(
            verify["dispatch"]["dispatch_path"]
                .as_str()
                .unwrap()
                .contains("multi-decode batch tiers")
        );
        assert_eq!(verify["dispatch"]["kernel_handle"], serde_json::Value::Null);
        // Verify's single-row fallback dispatches as a non-decode prefill.
        let single = receipt(true, true, 1, false, true, false);
        assert_eq!(single["entry"], "project_hot_verify");
        assert_eq!(single["dispatch"]["kernel_symbol"], "w4a16_gemm");
        assert_eq!(
            single["decode"], true,
            "outer decode flag is recorded verbatim"
        );
    }
}
