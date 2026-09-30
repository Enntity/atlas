// SPDX-License-Identifier: AGPL-3.0-only

//! Prefill-chunk KV preemption, the chunk counterpart of
//! `preempt::decode_batch_with_preemption`.
//!
//! A chunk that cannot get its KV blocks is refused on every rank before its
//! forward pass, each rank's reservation rolled back
//! (`spark_model::model::kv_admission`). The head then kills a larger
//! in-flight sequence (its free is mirrored to the worker) and re-sends the
//! whole chunk command; the worker, which kept waiting, runs it again. Only
//! that agreed exhaustion refusal is retried. An agreed refusal for any other
//! reservation error fails just this request (every rank rolled back and the
//! worker kept waiting, so the normal release reaches it), and any other
//! error, including a "KV cache exhausted" raised once the forward has
//! begun, may have mutated state on some rank, so it fails this request as
//! before.
//!
//! The one progress a refused chunk can keep is a completed tail-checkpoint
//! split half (`prefill_b.rs`: tokens appended, recurrent state advanced to
//! the cut). `seq.seq_len` records it on both ranks, so every attempt starts
//! there ([`resume_point`]) instead of running that half twice.
//!
//! One chunk error is not a request failure at all ([`or_end_pair`]): a peer
//! whose index-split rows were out of range is desynchronized, and this rank
//! has cached blocks computed from the guarded selection. The worker exits on
//! any step error; the head stops here too, as for a decode step error
//! (`preempt::decode_batch_with_preemption`), before any other request can
//! look that cache up.

use anyhow::Result;
use spark_model::layers::qwen3_attention::index_split_peer_fault;
use spark_model::model::kv_admission::kv_admission_refusal;
use spark_model::traits::{Model, SequenceState};
use spark_runtime::gpu::DevicePtr;

use super::lifecycle::send_error;
use super::types::ActiveSeq;

/// Where an attempt at chunk `[offset, end)` starts: the sequence's recorded
/// progress, which a refused attempt leaves at `offset` or at a completed
/// split cut inside the chunk.
pub(super) fn resume_point(offset: usize, end: usize, seq_len: usize) -> Result<usize> {
    anyhow::ensure!(
        (offset..end).contains(&seq_len),
        "prefill progress {seq_len} lies outside chunk {offset}..{end}"
    );
    Ok(seq_len)
}

/// Whether a chunk error leaves the pair unable to continue: the typed peer
/// fault, never an error that merely reads like it.
pub(super) fn ends_the_pair(e: &anyhow::Error) -> bool {
    index_split_peer_fault(e).is_some()
}

/// A chunk's result as the model returned it, unless its error ends the
/// pair: then this rank stops right here. Nothing may run first, since the
/// next prefix lookup could reuse the chunk's blocks; the peer's lifeline
/// takes it down with us and a supervisor restarts the pair.
pub(super) fn or_end_pair<T>(chunk: Result<T>) -> Result<T> {
    if let Err(e) = &chunk
        && ends_the_pair(e)
    {
        eprintln!("EP head: {e:#}; terminating (the peer exits with us)");
        crate::ep_peer_lifeline::terminate();
    }
    chunk
}

/// Send (EP) and run the prompt chunk `[offset, end)`, killing the largest
/// grammar-free active sequence after each agreed KV exhaustion refusal.
/// Falls through with the refusal when it is final or nothing can be evicted.
#[allow(clippy::too_many_arguments)]
pub(super) fn prefill_chunk_with_preemption(
    model: &dyn Model,
    prompt: &[u32],
    seq: &mut SequenceState,
    disable_mtp: bool,
    offset: usize,
    end: usize,
    is_last: bool,
    stream: u64,
    active: &mut Vec<ActiveSeq>,
) -> Result<DevicePtr> {
    loop {
        match send_and_run(
            model,
            prompt,
            seq,
            disable_mtp,
            offset,
            end,
            is_last,
            stream,
        ) {
            Err(e) if kv_admission_refusal(&e).is_some_and(|r| r.retryable) => {
                let Some(vi) = active
                    .iter()
                    .enumerate()
                    .filter(|(_, a)| a.grammar_state.is_none())
                    .max_by_key(|(_, a)| a.seq.block_table.len())
                    .map(|(i, _)| i)
                else {
                    return Err(e);
                };
                let mut victim = active.remove(vi);
                tracing::warn!(
                    "{e:#}: preempting slot={} ({} blocks) so this prefill can proceed \
                     ({} sequence(s) remain)",
                    victim.seq.slot_idx,
                    victim.seq.block_table.len(),
                    active.len(),
                );
                send_error(
                    model,
                    &mut victim,
                    "preempted: KV cache exhausted (a prefill needed its blocks)",
                );
            }
            res => return res,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn send_and_run(
    model: &dyn Model,
    prompt: &[u32],
    seq: &mut SequenceState,
    disable_mtp: bool,
    offset: usize,
    end: usize,
    is_last: bool,
    stream: u64,
) -> Result<DevicePtr> {
    let start = resume_point(offset, end, seq.seq_len)?;
    if start > offset {
        tracing::info!("prefill chunk {offset}..{end} resumes at recorded progress {start}");
    }
    let slot = seq.slot_idx as u32;
    // EP: the worker mirrors this exact command (bulk tokens, one NCCL op).
    model.ep_broadcast_disable_mtp_for_seq(slot, disable_mtp)?;
    model.ep_broadcast_cmd_for_seq(slot, 0xFFFFFFF0)?;
    model.ep_broadcast_cmd((end - start) as u32)?;
    model.ep_broadcast_cmd(start as u32)?;
    model.ep_broadcast_cmd(prompt.len() as u32)?;
    model.ep_broadcast_tokens(prompt)?;
    or_end_pair(model.prefill_chunk(prompt, seq, start, end - start, is_last, stream))
}
