// SPDX-License-Identifier: AGPL-3.0-only

//! Source contract between [`PDL_KERNELS`] and the GLM batched-GEMV kernels.

use super::PDL_KERNELS;

/// `(name, waits)` for every `__global__` kernel in `src`; `waits` when
/// `atlas_pdl_enter();` is the first statement of its body.
fn kernels(src: &str) -> Vec<(&str, bool)> {
    src.match_indices("__global__")
        .filter_map(|(at, _)| {
            let (signature, body) = src[at..].split_once('{')?;
            let is_name = |c: char| c.is_ascii_alphanumeric() || c == '_';
            // The parameter list is the last `(` of the signature; an earlier
            // one belongs to `__launch_bounds__(…)`.
            let head = signature[..signature.rfind('(')?].trim_end();
            let name = &head[head.rfind(|c| !is_name(c)).map_or(0, |i| i + 1)..];
            Some((name, body.trim_start().starts_with("atlas_pdl_enter();")))
        })
        .collect()
}

/// A kernel is launched with PDL exactly when it is on [`PDL_KERNELS`]. One
/// that is listed but does not wait reads its predecessor's output early; one
/// that waits but is not listed pays the launch gap PDL exists to remove (the
/// five-row KDA triple did, at every width-5 verify step).
#[test]
fn glm_batched_gemv_kernels_wait_exactly_when_listed() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../kernels/gb10/glm-5.3-flash/nvfp4/dense_gemv_bf16_batchm.cu");
    let src = std::fs::read_to_string(&path).unwrap();
    let kernels = kernels(&src);
    let names: Vec<_> = kernels.iter().map(|k| k.0).collect();
    for expected in [
        "dense_gemv_bf16_batchm",
        "dense_gemv_bf16_batchm_ahead",
        "dense_gemv_bf16_batch5_triple_n",
        "dense_gemv_bf16_batchm_dual_k128",
        "dense_gemv_bf16_tc32",
    ] {
        assert!(
            names.contains(&expected),
            "{expected} not parsed: {names:?}"
        );
    }
    for (name, waits) in kernels {
        assert_eq!(
            waits,
            PDL_KERNELS.contains(&name),
            "{name}: waits on PDL entry = {waits}, but listed = {}",
            !waits
        );
    }
}
