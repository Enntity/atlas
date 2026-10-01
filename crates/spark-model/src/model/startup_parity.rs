// SPDX-License-Identifier: AGPL-3.0-only

//! Startup agreement on the settings every rank must run with the same value.
//!
//! Each rank reads its switches from its own environment and command line,
//! and the worker mirrors the head command by command. A rank whose value of
//! one of these differs (a stale profile on one node) issues other
//! collectives or command words than its peer, takes another share of work
//! the pair splits, or fills a cache differently. The pair then deadlocks or
//! pairs unrelated collectives at some prompt-dependent step (one rank's
//! restore-depth reduction read as the other's KV-admission vote, which
//! refuses a cold prompt), or serves wrong numerics silently.
//!
//! So every rank gathers every rank's values once, right after the
//! communicator comes up, and fails on any difference, naming the setting and
//! both values. The gather is itself a collective: it runs whatever the
//! settings are, and its size is the length of [`SETTINGS`] plus the
//! caller's, which no setting changes. Only a build changes it, so a gather
//! of one word, the table's id, goes first and fails ranks on different
//! builds before their settings gathers could mispair.
//!
//! # What belongs in the table
//!
//! A setting whose mismatch changes the command words or collectives of a
//! step (their count, order, size or transport), the share of a split
//! computation a rank takes, or a cache both ranks must fill alike. Its value
//! comes from the parser the feature itself reads, never from a second
//! reading of the variable. A setting that only matters under a switch reads
//! 0 while that switch is off ([`while_on`]), so stale leftovers do not fail
//! a boot.
//!
//! Not here: what the ranks already reconcile (`ATLAS_KV_MAX_BLOCKS` takes
//! the pair's minimum; the RDMA pair compares its capacity and one-shot
//! settings at bootstrap), what is local to a rank (the drafter, the rails,
//! logging), and kernel choices that only reorder one rank's own arithmetic.

use anyhow::{Result, ensure};
use spark_comm::CommBackend;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::radix_tree::glm_pc_evict_enabled;

use super::trait_impl::finish_leaf;
use super::trait_impl::prefill_b::pc_policy as pc;
use super::{glm_independent, glm_vocab_split, graph_flags, mtp_carry, verify_pieces};
use crate::layer::glm_long_owner;
use crate::layers::qwen3_attention::{index_split_words, write_floor_legacy};
use crate::layers::{glm_sp, moe, ops};
use crate::speculative::glm_repair_policy;

/// One agreed setting: its name and a rank's value.
pub type Setting = (&'static str, u64);

/// A setting that only matters while its switch is `on`.
fn while_on(on: bool, value: fn() -> usize) -> Result<u64> {
    Ok(if on { value() as u64 } else { 0 })
}

/// The settings read in this crate: name, and this process's value from the
/// feature's own parser. One entry per setting.
const SETTINGS: &[(&str, fn() -> Result<u64>)] = &[
    // The head's command words.
    ("ATLAS_EP_PROTOCOL=v2", || {
        Ok(super::ep_protocol_v2_requested() as u64)
    }),
    // The passes of a prefill chunk and the rows each rank runs.
    ("ATLAS_NO_TAIL_SPLIT", || {
        Ok(pc::tail_split_disabled() as u64)
    }),
    ("ATLAS_GLM_PREFILL_SP", || Ok(glm_sp::requested() as u64)),
    // Sequence-parallel prefill needs the BF16 highway, which needs the
    // DFlash lane.
    ("ATLAS_GLM_HC_BF16", || {
        Ok(ops::hc_bf16_for("glm5_next") as u64)
    }),
    ("ATLAS_GLM_INDEX_SPLIT", || Ok(index_split_words()?[0])),
    ("ATLAS_GLM_INDEX_SPLIT_MIN_CTX", || {
        Ok(index_split_words()?[1])
    }),
    (
        "ATLAS_GLM_INDEX_SPLIT_CHECK",
        || Ok(index_split_words()?[2]),
    ),
    // Prefix-cache policy: the min-reductions of a prefill, its restore
    // depth (so its row range) and the head's cache-sequence command.
    ("ATLAS_GLM_PC_EVICT", || Ok(glm_pc_evict_enabled() as u64)),
    ("ATLAS_GLM_PC_BRANCH", || {
        Ok(pc::glm_pc_branch_enabled() as u64)
    }),
    ("ATLAS_GLM_PC_BRANCH_MIN", || {
        while_on(pc::glm_pc_branch_enabled(), pc::glm_pc_branch_min_tokens)
    }),
    // In effect, that is with the preconditions `finish_leaf::flag` checks.
    ("ATLAS_GLM_PC_FINISH_LEAF", || {
        Ok(finish_leaf::enabled() as u64)
    }),
    ("ATLAS_GLM_PC_FINISH_LEAF_BLOCKS", || {
        while_on(finish_leaf::enabled(), finish_leaf::span_blocks)
    }),
    ("ATLAS_MARCONI_EXACT", || {
        Ok(mtp_carry::marconi_exact() as u64)
    }),
    ("ATLAS_MARCONI_MIN_TOKENS", || {
        Ok(mtp_carry::marconi_min_tokens() as u64)
    }),
    ("ATLAS_MARCONI_PREFILL_ONLY", || {
        Ok(mtp_carry::marconi_prefill_only() as u64)
    }),
    // Which cached rows a prefill rewrites, so what each rank attends.
    ("ATLAS_GLM_PC_WRITE_FLOOR", || {
        Ok(pc::glm_pc_write_floor_enabled() as u64)
    }),
    ("ATLAS_GLM_KV_WRITE_FLOOR_LEGACY", || {
        Ok(write_floor_legacy() as u64)
    }),
    // Work the pair splits, and the exchanges that stand in for a reduce.
    ("ATLAS_GLM_SHARED_TP_SPLIT", || {
        Ok(moe::shared_tp_split_requested() as u64)
    }),
    ("ATLAS_MOE_SHARED_REDUCE_OVERLAP", || {
        Ok(moe::shared_reduce_overlap_requested() as u64)
    }),
    ("ATLAS_GLM_K5_FUSED_TP_HC", || {
        Ok(crate::layers::verify_fused_tp_hc_enabled() as u64)
    }),
    ("ATLAS_GLM_VERIFY_VOCAB_SPLIT", || {
        Ok(glm_vocab_split::enabled() as u64)
    }),
    // The lane a decode or verify step takes on both ranks.
    ("ATLAS_GLM_DFLASH", || {
        Ok(glm_repair_policy::dflash_enabled() as u64)
    }),
    ("ATLAS_GLM_DFLASH_PREFILL_VERIFY", || {
        Ok(glm_repair_policy::dflash_prefill_verify() as u64)
    }),
    ("ATLAS_GLM_MTP_REPAIR", || {
        Ok(glm_repair_policy::enabled() as u64)
    }),
    ("ATLAS_GLM_MTP_LONG_CONTEXT", || {
        Ok(glm_repair_policy::long_context_enabled() as u64)
    }),
    ("ATLAS_GLM_LONG_BATCH_VERIFY", || {
        Ok(glm_long_owner::enabled()? as u64)
    }),
    ("ATLAS_GLM_INDEPENDENT_DECODE", || {
        Ok(glm_independent::enabled("glm5_next")? as u64)
    }),
    // Whether a step runs as a CUDA graph: a capturing forward takes other
    // collective paths than an eager one (`graph_flags`).
    ("ATLAS_EP_GRAPHS", || Ok(graph_flags::ep_graphs() as u64)),
    ("ATLAS_GDN_DECODE_GRAPH", || {
        Ok(graph_flags::gdn_decode_graph() as u64)
    }),
    ("ATLAS_NO_DECODE_GRAPHS", || {
        Ok(graph_flags::no_decode_graphs() as u64)
    }),
    ("ATLAS_NO_DECODE_GRAPHS_MULTISEQ", || {
        Ok(!graph_flags::multiseq_graphs_enabled() as u64)
    }),
    ("ATLAS_GLM_TP_VERIFY_GRAPH", || {
        Ok(graph_flags::glm_tp_verify_graph() as u64)
    }),
    ("ATLAS_GLM_MTP1_VERIFY_GRAPH", || {
        Ok(graph_flags::glm_mtp1_verify_graph() as u64)
    }),
    ("ATLAS_DEBUG_NO_GRAPH", || {
        Ok(graph_flags::debug_no_graph() as u64)
    }),
    ("ATLAS_GLM_VERIFY_GRAPH", || {
        Ok(verify_pieces::requested() as u64)
    }),
];

/// This process's settings: [`SETTINGS`], then the `caller`'s.
fn settings(caller: &[Setting]) -> Result<Vec<Setting>> {
    SETTINGS
        .iter()
        .map(|&(name, read)| Ok((name, read()?)))
        .chain(caller.iter().copied().map(Ok))
        .collect()
}

/// Call on every rank right after a multi-rank communicator comes up, before
/// any other collective. Fails, on every rank, when a setting differs across
/// the ranks, and on a setting its parser refuses. `caller` carries the
/// settings resolved outside this crate: the same names in the same order on
/// every rank.
pub fn agree(comm: &dyn CommBackend, gpu: &dyn GpuBackend, caller: &[Setting]) -> Result<()> {
    agree_on(&settings(caller)?, comm, gpu)
}

/// Names a table: FNV-1a over its setting names, in order.
fn table_id(settings: &[Setting]) -> u64 {
    settings
        .iter()
        .flat_map(|s| s.0.bytes().chain([0]))
        .fold(0xcbf2_9ce4_8422_2325, |id, b| {
            (id ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
        })
}

fn agree_on(ours: &[Setting], comm: &dyn CommBackend, gpu: &dyn GpuBackend) -> Result<()> {
    let me = comm.rank();
    // The table is part of the build and the second gather is as long as the
    // table, so first compare the tables, in a gather of one word.
    let table = table_id(ours);
    ensure!(
        gather_words(comm, gpu, &[table])?
            .iter()
            .all(|&t| t == table),
        "rank {me} compares other settings at startup than its peers: \
         the ranks run different builds"
    );
    let values: Vec<u64> = ours.iter().map(|s| s.1).collect();
    let all = gather_words(comm, gpu, &values)?;
    let differ: Vec<String> = all
        .chunks(ours.len())
        .enumerate()
        .flat_map(|(rank, theirs)| {
            ours.iter()
                .zip(theirs)
                .filter(|(ours, theirs)| ours.1 != **theirs)
                .map(move |((name, ours), theirs)| {
                    format!("{name}: rank {me} has {ours}, rank {rank} has {theirs}")
                })
        })
        .collect();
    ensure!(
        differ.is_empty(),
        "every rank must run the same settings ({})",
        differ.join("; ")
    );
    Ok(())
}

/// Every rank's `words`, in rank order: one all-gather of `8 * words.len()`
/// bytes a rank.
pub(crate) fn gather_words(
    comm: &dyn CommBackend,
    gpu: &dyn GpuBackend,
    words: &[u64],
) -> Result<Vec<u64>> {
    let ours: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let mut all = vec![0u8; ours.len() * comm.world_size()];
    let buf = gpu.alloc(ours.len() + all.len())?;
    let recv = buf.offset(ours.len());
    let gathered = gpu
        .copy_h2d(&ours, buf)
        .and_then(|()| comm.all_gather(buf.0, recv.0, ours.len()))
        .and_then(|()| gpu.copy_d2h(recv, &mut all));
    gpu.free(buf)?;
    gathered?;
    Ok(all
        .chunks_exact(8)
        .map(|w| u64::from_le_bytes(w.try_into().expect("8-byte words")))
        .collect())
}

#[cfg(test)]
#[path = "startup_parity_tests.rs"]
mod tests;
