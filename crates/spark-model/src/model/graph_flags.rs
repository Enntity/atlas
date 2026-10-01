// SPDX-License-Identifier: AGPL-3.0-only

//! The switches that let a multi-rank decode or verify step run as a CUDA
//! graph.
//!
//! A capturing forward takes other collective paths than an eager one: the
//! layers branch on `ForwardContext::graph_capture` (a blocking all-reduce
//! for the event-ordered one, no pair exchange in place of a reduce), and
//! the RDMA pair refuses a capturing stream, which falls through to NCCL.
//! So a rank that captures alone mispairs the step's collectives, and every
//! rank must run the same values (`startup_parity`).

fn on_or_true(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| v == "1" || v == "true")
}

/// `ATLAS_EP_GRAPHS=1`: capture the single-sequence decode under EP.
pub(crate) fn ep_graphs() -> bool {
    on_or_true("ATLAS_EP_GRAPHS")
}

/// `ATLAS_GDN_DECODE_GRAPH=1`: capture the GDN HeadParallel TP decode.
pub(crate) fn gdn_decode_graph() -> bool {
    on_or_true("ATLAS_GDN_DECODE_GRAPH")
}

/// `ATLAS_NO_DECODE_GRAPHS=1`: force the single-sequence decode eager.
pub(crate) fn no_decode_graphs() -> bool {
    std::env::var("ATLAS_NO_DECODE_GRAPHS").is_ok_and(|v| v == "1")
}

/// Multi-seq decode CUDA graphs: **ON by default**, disabled by
/// `ATLAS_NO_DECODE_GRAPHS_MULTISEQ=1`.
///
/// Strict `== "1"` on an `ATLAS_NO_*` name rather than a presence check —
/// presence-checked flags here are ENABLED by `=0`. Read once per process.
pub(crate) fn multiseq_graphs_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_NO_DECODE_GRAPHS_MULTISEQ").as_deref() != Ok("1"))
}

/// `ATLAS_GLM_TP_VERIFY_GRAPH=1`: capture the GLM TP2 K=5 verify. Read once.
pub(crate) fn glm_tp_verify_graph() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("ATLAS_GLM_TP_VERIFY_GRAPH")
            .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
    })
}

/// `ATLAS_GLM_MTP1_VERIFY_GRAPH=1`: capture the repaired GLM C1 K=2 verify.
pub(crate) fn glm_mtp1_verify_graph() -> bool {
    std::env::var("ATLAS_GLM_MTP1_VERIFY_GRAPH").as_deref() == Ok("1")
}

/// `ATLAS_DEBUG_NO_GRAPH=1`: start with graphs suppressed (PCND diagnostic).
pub(crate) fn debug_no_graph() -> bool {
    std::env::var("ATLAS_DEBUG_NO_GRAPH").as_deref() == Ok("1")
}
