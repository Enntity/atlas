// SPDX-License-Identifier: AGPL-3.0-only

//! GPU init + pre-load reserve preflight + post-load OOM check.

use anyhow::{Context, Result};

use atlas_core::config::ModelConfig;

use crate::cli;

mod independent;
mod ssm_h_fp16;
pub(crate) use independent::prepare_reserve;
use ssm_h_fp16::ssm_h_fp16_preconditions;

pub(crate) struct ReservePreflight {
    pub(crate) inference_reserve: usize,
    pub(crate) buffer_arena_bytes: usize,
    pub(crate) gdn_two_phase_bytes: usize,
    pub(crate) ssm_prefill_chunk: usize,
    pub(crate) max_batch_tokens_pre: usize,
    pub(crate) resolved_prefill: Option<super::PrefillBudget>,
}

fn glm5_dual_spark_parallelism(world: usize, tp: usize, ep: usize) -> bool {
    world == 2 && ep == 2 && matches!(tp, 1 | 2)
}

fn glm5_concurrency_supported(
    max_batch: usize,
    max_num_seqs: usize,
    ep_v2: bool,
    c4: bool,
) -> bool {
    (c4 && max_batch == 4 && max_num_seqs == 4 && ep_v2)
        || ((1..=3).contains(&max_batch)
            && (max_batch..=5).contains(&max_num_seqs)
            && (max_batch == 1 || ep_v2))
}

fn glm5_context_supported(max_seq_len: usize, max_prefill_tokens: usize, model_max: usize) -> bool {
    max_seq_len <= model_max && max_prefill_tokens > 0
}

fn glm5_mtp_context_supported(max_seq_len: usize, num_drafts: usize, index_topk: usize) -> bool {
    // Scheduler clamps zero drafts to one; the verifier has no semantic index.
    max_seq_len
        .checked_add(num_drafts.max(1))
        .is_some_and(|verify_end| verify_end <= index_topk)
}

fn glm5_long_context_concurrency_supported(
    max_seq_len: usize,
    index_topk: usize,
    max_batch: usize,
    max_num_seqs: usize,
    sparse_decode: bool,
    c4_sparse: bool,
) -> bool {
    max_seq_len <= index_topk
        || (max_batch == 1 && max_num_seqs == 1)
        || (sparse_decode && matches!(max_batch, 2 | 3) && (max_batch..=5).contains(&max_num_seqs))
        || (sparse_decode
            && c4_sparse
            && spark_model::model::glm_c4::validate_launch_limits_for_mode(
                max_batch,
                max_num_seqs,
                max_seq_len,
                true,
                true,
            )
            .is_ok())
}

pub(crate) fn preflight_reserve(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    free_mem: usize,
) -> Result<ReservePreflight> {
    let independent = spark_model::model::glm_independent::enabled(&config.model_type)?;
    let long_mtp = spark_model::speculative::glm_repair_policy::long_context_enabled();
    anyhow::ensure!(
        !long_mtp || spark_model::speculative::glm_repair_policy::enabled(),
        "GLM long-context MTP requires repaired ownership"
    );
    anyhow::ensure!(
        !long_mtp || (!args.glm_paired_mtp && !independent),
        "GLM long-context MTP2 uses its retained serial verifier, not another dispatcher"
    );
    // Selected dispatchers retain TP-local topology and the exact row budget.
    // Speculative profiles keep their intermediates and full CUDA headroom.
    let bounded = independent || args.glm_paired_mtp || long_mtp;
    if spark_model::speculative::glm_repair_policy::parse(
        std::env::var("ATLAS_GLM_MTP_REPAIR").ok().as_deref(),
    )? {
        anyhow::ensure!(
            !args.high_speed_swap && args.swap_space_gb == 0,
            "GLM repair first lane requires resident state with swap disabled"
        );
        spark_model::speculative::glm_repair_policy::RepairPolicy {
            model_type: &config.model_type,
            world: args.world_size,
            tp: args.tp_size,
            ep: args.ep_size,
            active: args.max_batch_size,
            admitted: args.max_num_seqs,
            // The repair verifier's indexed domain ends at 32K. A larger
            // served native context remains usable, but requests whose
            // prompt+max_tokens budget crosses that domain are fenced into
            // native decode by scheduler admission.
            context: args
                .max_seq_len
                .min(spark_model::speculative::glm_repair_policy::MAX_LONG_CONTEXT),
            drafts: args.resolved_num_drafts(),
            native_only: args.speculative
                && !(args.dflash || args.self_speculative || args.ngram_speculative),
            bf16: args.kv_cache_dtype.as_deref() == Some("bf16"),
            prefix_reuse: args.enable_prefix_caching,
            force: args
                .mtp_gate
                .as_deref()
                .map(|v| v == "force")
                .unwrap_or_else(|| std::env::var("ATLAS_MTP_GATE_FORCE").as_deref() == Ok("1")),
            long_context: long_mtp,
        }
        .validate()?;
        spark_model::speculative::glm_repair_policy::validate_environment()?;
    }
    let c4_sparse = spark_model::model::glm_c4::validate_sparse_flag(
        &config.model_type,
        std::env::var("ATLAS_GLM_C4_SPARSE").ok().as_deref(),
    )?;
    if config.model_type == "glm5_next" {
        let sparse_decode =
            spark_model::layers::qwen3_attention::glm_multi_seq_sparse_enabled(&config.model_type);
        anyhow::ensure!(
            !sparse_decode
                || !(args.speculative || args.self_speculative || args.ngram_speculative),
            "ATLAS_GLM_MULTI_SEQ_SPARSE=1 supports independent non-speculative decode only"
        );
        anyhow::ensure!(
            glm5_context_supported(
                args.max_seq_len,
                args.max_prefill_tokens,
                config.max_position_embeddings,
            ),
            "GLM-5 requires --max-seq-len <= {} and a non-zero --max-prefill-tokens for bounded chunked prefill",
            config.max_position_embeddings,
        );
        let ep_v2 = matches!(std::env::var("ATLAS_EP_PROTOCOL").as_deref(), Ok("v2"));
        anyhow::ensure!(
            !long_mtp || args.max_batch_size == 1 || ep_v2,
            "GLM concurrent MTP2 requires ATLAS_EP_PROTOCOL=v2 for owner slot identity"
        );
        let c4 = spark_model::model::glm_c4::enabled(&config.model_type);
        anyhow::ensure!(
            independent || std::env::var("ATLAS_GLM_C4_GROUPED_MOE").as_deref() != Ok("1") || c4,
            "ATLAS_GLM_C4_GROUPED_MOE=1 requires ATLAS_GLM_C4_DECODE=1"
        );
        if !independent
            && !args.glm_paired_mtp
            && !long_mtp
            && (c4 || c4_sparse || args.max_batch_size == 4)
        {
            spark_model::model::glm_c4::validate_prefill_budget(
                args.max_prefill_tokens,
                c4_sparse,
            )?;
            spark_model::model::glm_c4::policy_from_env(
                &config.model_type,
                args.world_size,
                args.tp_size,
                args.ep_size,
                ep_v2,
                !(args.speculative
                    || args.self_speculative
                    || args.ngram_speculative
                    || args.dflash),
            )
            .validate()?;
            spark_model::model::glm_c4::validate_launch_limits(
                args.max_batch_size,
                args.max_num_seqs,
                args.max_seq_len,
                args.kv_cache_dtype.as_deref() == Some("bf16"),
            )?;
        }
        anyhow::ensure!(
            independent
                || long_mtp
                || args.glm_paired_mtp
                || glm5_concurrency_supported(args.max_batch_size, args.max_num_seqs, ep_v2, c4),
            "GLM-5 dual-Spark concurrency supports --max-batch-size 1..=3 and \
             --max-num-seqs max_batch..=5, or explicitly opted-in C4 with active/admitted4; \
             batches above one require ATLAS_EP_PROTOCOL=v2"
        );
        anyhow::ensure!(
            long_mtp
                || glm5_long_context_concurrency_supported(
                    args.max_seq_len,
                    config.index_topk,
                    args.max_batch_size,
                    args.max_num_seqs,
                    sparse_decode,
                    c4_sparse,
                ),
            "GLM-5 context above index_topk={} requires one active/admitted sequence, opt-in sparse C2/C3, or guarded eager ATLAS_GLM_C4_SPARSE=1",
            config.index_topk,
        );
        anyhow::ensure!(
            !(args.self_speculative || args.ngram_speculative),
            "GLM-5 supports its checkpoint MTP layer via --speculative; self/ngram speculative modes are unsupported"
        );
        if args.speculative {
            anyhow::ensure!(
                long_mtp
                    || glm5_mtp_context_supported(
                        args.max_seq_len,
                        args.resolved_num_drafts(),
                        config.index_topk,
                    ),
                "GLM-5 MTP requires --max-seq-len + max(--num-drafts, 1) <= {}; verifier semantic indexing is not implemented",
                config.index_topk,
            );
            anyhow::ensure!(
                args.max_batch_size == 1
                    || long_mtp
                    || (args.glm_paired_mtp && (2..=8).contains(&args.max_batch_size)),
                "GLM-5 MTP requires C1 or the supervised bounded-owner dispatcher"
            );
        }
        anyhow::ensure!(
            independent || glm5_dual_spark_parallelism(args.world_size, args.tp_size, args.ep_size),
            "GLM-5 dual-Spark support requires --world-size 2 --ep-size 2 and either --tp-size 1 (EP fallback) or --tp-size 2 (overlapping TP+EP)"
        );
    }
    let h_state_bytes = config.ssm_h_state_bytes();
    let conv_state_bytes = config.ssm_conv_state_bytes();
    let spec_on_pool = args.speculative || args.self_speculative || args.ngram_speculative;
    ssm_h_fp16_preconditions(args, config)?;
    // SSM state pool = per-seq live state (max_batch blobs) + MTP verify
    // state (intermediates + checkpoint) for the slots spec dispatch can
    // actually reach. SSOT: `ssm_reserve::mtp_state_slots` — the SAME
    // number `SsmStatePool::new` allocates and the scheduler's spec
    // dispatch guard enforces. At bs<=32 this reproduces the historical
    // `max_batch × blob × (1 + (num_drafts+1) + 1)` byte-for-byte; above
    // 32 it stops reserving verify blobs for slots that can never verify
    // (25.4 GB at bs=64/K=4 on the 27B — the bs=64 preflight refusal).
    // Kill switch: ATLAS_MTP_POOL_FULL_WIDTH (presence) restores
    // full-width sizing on BOTH sides.
    let mtp_state_slots = spark_model::ssm_reserve::mtp_state_slots(args.max_batch_size);
    // Tiered verify slots (2026-08-16): the H-intermediate term is per-slot
    // (`verify_slot_h_intermediates`); DFlash pools are γ-sized and do not
    // follow the MTP ladder, so they reserve uniform full width — mirroring
    // `SsmStatePool::new`'s `num_intermediates != num_drafts + 1` condition.
    // Stage-3 f16-SIZED pool: the FP32 prefill staging arena, ONE blob per
    // slot (shared across layers — see `ssm_h_prefill_stage_bytes`). A
    // separate term for the same reason the replay ring is: it is sized by a
    // SINGLE layer's h blob, not by the across-layers per-seq total every
    // other term here uses. Zero on an FP32-sized pool. `max_batch_size`, not
    // `+1`: this preflight has never counted the pools' dummy slot.
    let ssm_h_stage_bytes = spark_model::ssm_reserve::ssm_h_prefill_stage_bytes(
        args.max_batch_size,
        h_state_bytes,
        spark_model::layers::qwen3_ssm::ssm_h_f16_pool_enabled(),
    );
    // Selected FP32 pools allocate one additional live dummy slot.
    // It has no prefix/rollback snapshots; do not inflate those counts.
    let live_slots = args.max_batch_size + usize::from(bounded);
    let mut ssm_pool_bytes = spark_model::ssm_reserve::ssm_pool_reserve_bytes(
        live_slots,
        config.num_ssm_layers() * h_state_bytes,
        config.num_ssm_layers() * conv_state_bytes,
        spec_on_pool,
        args.resolved_num_drafts(),
        mtp_state_slots,
        args.dflash,
        // Stage-3 f16-SIZED pool: mirrors `SsmStatePool::new`'s narrowing.
        // Unreachable today (ssm_h_fp16_preconditions refuses the mode
        // above), wired so preflight and allocator cannot diverge when the
        // refusal lifts.
        spark_model::layers::qwen3_ssm::ssm_h_f16_pool_enabled(),
        // `--ssm-rollback-mode` (published by serve_flags before this runs).
        // Replay drops every per-token verify intermediate; its input ring
        // is the separate term below.
        spark_model::ssm_reserve::ssm_rollback_mode(),
    );
    if long_mtp {
        // SsmStatePool also allocates an MTP dummy, separate from the live
        // dummy above. Its H snapshots always have full width, irrespective
        // of the real slots' ladder. Keep all of its bytes outside headroom.
        let dummy = spark_model::ssm_reserve::ssm_pool_reserve_bytes(
            0,
            config.num_ssm_layers() * h_state_bytes,
            config.num_ssm_layers() * conv_state_bytes,
            true,
            args.resolved_num_drafts(),
            1,
            true,
            spark_model::layers::qwen3_ssm::ssm_h_f16_pool_enabled(),
            spark_model::ssm_reserve::ssm_rollback_mode(),
        );
        ssm_pool_bytes = ssm_pool_bytes
            .checked_add(dummy)
            .context("long MTP dummy reserve overflow")?;
    }
    // Replay-mode verify-window input ring (EXPERIMENTAL scaffold): sized by
    // the SAME SSOT `SsmStatePool::new` allocates through. K ceiling is the
    // MTP `num_drafts + 1` — matching this preflight's existing convention
    // for the conv term (the DFlash γ=17 widening and the pools' dummy slot
    // have never been preflight-counted; the CUDA headroom absorbs them).
    let ssm_replay_ring = if spec_on_pool
        && spark_model::ssm_reserve::ssm_rollback_mode()
            == spark_model::ssm_reserve::SsmRollbackMode::Replay
    {
        spark_model::ssm_reserve::ssm_replay_ring_bytes(
            config.num_ssm_layers(),
            spark_model::ssm_reserve::ssm_replay_row_bytes(
                config.ssm_qkvz_size(),
                config.linear_num_value_heads,
            ),
            args.resolved_num_drafts() + 1,
            mtp_state_slots + usize::from(long_mtp),
        )
    } else {
        0
    };
    let spec_tokens_pre = if args.speculative || args.self_speculative || args.ngram_speculative {
        args.resolved_num_drafts() + 2
    } else {
        1
    };
    // B4 (chunked-prefill BF16 KV cliff): the prior `.min(8192)` cap forced
    // every prompt > 8 k to chunk, which compounds K-side BF16 rounding noise
    // at chunk boundaries (per the 4-agent audit 2026-05-27). When the user
    // explicitly passes `--max-prefill-tokens N` (anything other than the
    // default 8192), respect it — no hard cap. Otherwise default to 8192 to
    // bound GDN persistent-buffer reservation for unbounded `max_seq_len`.
    let ssm_prefill_chunk: usize = if config.num_ssm_layers() > 0 {
        if args.max_prefill_tokens != 8192 && args.max_prefill_tokens > 0 {
            args.max_seq_len.min(args.max_prefill_tokens)
        } else {
            args.max_seq_len.min(8192)
        }
    } else {
        0
    };
    let user_set_prefill_pre = args.max_prefill_tokens != 8192;
    let prefill_budget_pre = if user_set_prefill_pre && args.max_prefill_tokens > 0 {
        args.max_prefill_tokens
    } else if ssm_prefill_chunk > 0 {
        ssm_prefill_chunk
    } else if args.max_prefill_tokens > 0 {
        args.max_prefill_tokens
    } else {
        args.max_seq_len
    };
    // Issue #15 auto-clamp removed (2026-07-02): snapshot reachability is
    // handled by the tail-checkpoint split in `prefill_chunk_dispatch`, so
    // the budget (and this arena-sizing mirror) stays at full chunk size.
    let resolved_prefill = bounded.then(|| super::resolve_prefill_budget(args, ssm_prefill_chunk));
    let max_batch_tokens_pre = resolved_prefill.as_ref().map_or_else(
        || {
            prefill_budget_pre
                .max(spec_tokens_pre)
                .max(args.max_batch_size)
        },
        |budget| budget.max_batch_tokens,
    );
    // The selected envelope is bounded before BufferSizes' unchecked products.
    anyhow::ensure!(
        !bounded || max_batch_tokens_pre <= 65535,
        "selected arena rows exceed the CUDA grid limit"
    );
    let buffer_arena_bytes = spark_runtime::buffers::BufferSizes::from_config(
        config,
        max_batch_tokens_pre,
        args.max_seq_len,
        args.block_size,
        args.max_batch_size,
    )
    .total_bytes();
    // SSM snapshot pool = Marconi prefix-cache region + Phase-C
    // decode-rollback ring. The decode ring is sized per active
    // sequence (ring slots × `max_batch_size`) and only allocated for SSM
    // models. SSOT: `spark_model::ssm_reserve::decode_rollback_ring_slots`
    // makes the SAME decision (same env vars, same constant) the runtime
    // allocation in `TransformerModel::new` makes — including the skip under
    // `--speculative`/`--dflash` (the ring's save/rollback path only runs on
    // plain decode; the spec path rolls back through the verify snapshot).
    // Reserving the ring unconditionally while the runtime skipped it
    // stranded ~38 GB at bs32 on the 27B (75.2 GB SSM reserve vs an 85.2 GB
    // budget at util 0.70) and capped the native batch at ~20.
    // `use_speculative` here MUST mirror what `build_model` passes:
    // `args.speculative || args.dflash`.
    // Kill switch: `ATLAS_SSM_RESERVE_RING_FULL` present ⇒ restore the old
    // unconditional reservation (accounting-only, safe over-reserve;
    // presence-style — `=0` is NOT "off").
    let decode_ring_slots = if std::env::var("ATLAS_SSM_RESERVE_RING_FULL").is_ok() {
        if config.num_ssm_layers() > 0 {
            atlas_kernels::DECODE_ROLLBACK_RING_SLOTS
        } else {
            0
        }
    } else {
        spark_model::ssm_reserve::decode_rollback_ring_slots(
            config.num_ssm_layers(),
            args.speculative || args.dflash,
        )
        .slots
    };
    let ssm_snapshot_bytes = if bounded {
        args.ssm_cache_slots
            .checked_add(decode_ring_slots * args.max_batch_size)
            .and_then(|n| n.checked_mul(config.num_ssm_layers()))
            .and_then(|n| n.checked_mul(h_state_bytes + conv_state_bytes))
            .context("selected snapshot reserve overflow")?
    } else {
        (args.ssm_cache_slots + decode_ring_slots * args.max_batch_size)
            * config.num_ssm_layers()
            * (h_state_bytes + conv_state_bytes)
    };
    let cuda_headroom: usize =
        if args.speculative || args.self_speculative || args.ngram_speculative {
            4 * 1024 * 1024 * 1024
        } else {
            512 * 1024 * 1024
        };
    let gdn_two_phase_bytes: usize = {
        let key_dim = config.linear_num_key_heads * config.linear_key_head_dim;
        let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
        let nv = config.linear_num_value_heads;
        let conv_dim = key_dim * 2 + value_dim;
        if conv_dim > 0 && config.num_ssm_layers() > 0 {
            let sl = if bounded {
                max_batch_tokens_pre.min(args.max_seq_len)
            } else {
                max_batch_tokens_pre
            };
            sl * conv_dim * 2 + sl * nv * 2 * 4 + sl * value_dim * 2 + sl * value_dim * 2
        } else {
            0
        }
    };
    let paired_private_bytes = if args.glm_paired_mtp {
        spark_model::layers::Glm5MtpHead::paired_private_reserve_bytes(
            config,
            args.max_seq_len,
            args.max_batch_size,
        )?
    } else {
        0
    };
    let repair_private_bytes = if long_mtp {
        let private = spark_model::layers::Glm5MtpHead::repair_private_reserve_bytes(
            config,
            args.max_seq_len,
        )?;
        // Same bounded arena the runtime capture allocates: `arena_context`
        // is the SSOT, so the quote tracks the private block plan above.
        let capture = spark_model::speculative::glm_repair_policy::arena_context(
            &config.model_type,
            long_mtp,
            args.max_seq_len,
        )
        .checked_mul(config.hidden_size)
        .and_then(|n| n.checked_mul(2))
        .context("GLM MTP prompt capture reserve overflow")?;
        tracing::info!(
            "GLM MTP2 owned reserve: private={} MiB, prompt_capture={} MiB",
            private / (1024 * 1024),
            capture / (1024 * 1024)
        );
        private
            .checked_add(capture)
            .context("GLM MTP private reserve overflow")?
    } else {
        0
    };
    let inference_reserve: usize = if bounded || long_mtp {
        [
            ssm_pool_bytes,
            ssm_h_stage_bytes,
            ssm_replay_ring,
            ssm_snapshot_bytes,
            gdn_two_phase_bytes,
            cuda_headroom,
            paired_private_bytes,
            repair_private_bytes,
        ]
        .into_iter()
        .try_fold(0usize, |sum, bytes| sum.checked_add(bytes))
        .context("selected inference reserve overflow")?
    } else {
        ssm_pool_bytes
            + ssm_h_stage_bytes
            + ssm_replay_ring
            + ssm_snapshot_bytes
            + gdn_two_phase_bytes
            + cuda_headroom
    };
    let total_reserve = if bounded {
        inference_reserve
            .checked_add(buffer_arena_bytes)
            .context("selected total reserve overflow")?
    } else {
        inference_reserve + buffer_arena_bytes
    };
    if total_reserve > free_mem {
        let need_gb = total_reserve as f64 / (1024.0 * 1024.0 * 1024.0);
        let free_gb = free_mem as f64 / (1024.0 * 1024.0 * 1024.0);
        let fixed = ssm_pool_bytes + ssm_h_stage_bytes + ssm_snapshot_bytes + cuda_headroom;
        let budget_for_seq_term = free_mem.saturating_sub(fixed) / 2;
        let per_tok_bytes = {
            let key_dim = config.linear_num_key_heads * config.linear_key_head_dim;
            let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
            let nv = config.linear_num_value_heads;
            let conv_dim = key_dim * 2 + value_dim;
            if conv_dim > 0 && config.num_ssm_layers() > 0 {
                (conv_dim * 2) + (nv * 2 * 4) + (value_dim * 2) + (value_dim * 2)
            } else {
                0
            }
        };
        let suggested = budget_for_seq_term
            .checked_div(per_tok_bytes)
            .map(|q| q.max(2048))
            .unwrap_or(0);
        let hint = if suggested > 0 && suggested < args.max_seq_len {
            format!(
                " Try --max-seq-len {} (or lower --max-batch-size / --num-drafts).",
                suggested
            )
        } else if args.max_batch_size > 1 {
            " Reduce --max-batch-size.".to_string()
        } else {
            " Use a smaller model or a GPU with more memory.".to_string()
        };
        anyhow::bail!(
            "Preflight failed: inference buffers alone need {:.2} GB but only {:.2} GB is free on the GPU \
             (before weights load). SSM pool + GDN chunked prefill scales with --max-seq-len={} × --max-batch-size={}.{}",
            need_gb,
            free_gb,
            args.max_seq_len,
            args.max_batch_size,
            hint,
        );
    }
    tracing::info!(
        "Preflight reserve: inference={} MB, buffer_arena={} MB (pre-load free: {:.1} GB)",
        inference_reserve / (1024 * 1024),
        buffer_arena_bytes / (1024 * 1024),
        free_mem as f64 / (1024.0 * 1024.0 * 1024.0),
    );
    // Q09: per-component breakdown so future MTP/spec-decode reserve
    // jumps are diagnosable from the log alone. Each line is dropped at
    // debug to avoid noise on hot startup paths; flip to info if you
    // need to trace a specific deployment's reserve.
    let spec_on = spec_on_pool;
    tracing::debug!(
        "Preflight reserve breakdown: \
         ssm_pool={} MB ({} max_batch blobs + {} MTP-covered slots × {} verify blobs, \
         {} ssm_layers × (h+conv)), \
         ssm_snapshot={} MB ({} slots), \
         gdn_two_phase={} MB ({} tokens), \
         cuda_headroom={} MB ({}), \
         spec_on={}, num_drafts={}",
        ssm_pool_bytes / (1024 * 1024),
        live_slots,
        if spec_on_pool { mtp_state_slots } else { 0 },
        if spec_on_pool {
            args.resolved_num_drafts() + 2
        } else {
            0
        },
        config.num_ssm_layers(),
        ssm_snapshot_bytes / (1024 * 1024),
        args.ssm_cache_slots,
        gdn_two_phase_bytes / (1024 * 1024),
        max_batch_tokens_pre,
        cuda_headroom / (1024 * 1024),
        if spec_on { "spec/MTP on" } else { "no spec" },
        spec_on,
        if spec_on {
            args.resolved_num_drafts() as i64
        } else {
            -1
        },
    );
    Ok(ReservePreflight {
        inference_reserve,
        buffer_arena_bytes,
        gdn_two_phase_bytes,
        ssm_prefill_chunk,
        max_batch_tokens_pre,
        resolved_prefill,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        glm5_concurrency_supported, glm5_context_supported, glm5_dual_spark_parallelism,
        glm5_long_context_concurrency_supported, glm5_mtp_context_supported,
    };

    #[test]
    fn glm5_accepts_ep_fallback_and_overlapping_tp2_only() {
        assert!(glm5_dual_spark_parallelism(2, 1, 2));
        assert!(glm5_dual_spark_parallelism(2, 2, 2));
        assert!(!glm5_dual_spark_parallelism(2, 2, 1));
        assert!(!glm5_dual_spark_parallelism(4, 2, 2));
        assert!(!glm5_dual_spark_parallelism(2, 4, 2));
    }

    #[test]
    fn glm5_concurrency_is_bounded_and_requires_ep_v2() {
        assert!(glm5_concurrency_supported(1, 1, false, false));
        assert!(glm5_concurrency_supported(3, 5, true, false));
        assert!(!glm5_concurrency_supported(2, 5, false, false));
        assert!(!glm5_concurrency_supported(4, 5, true, false));
        assert!(!glm5_concurrency_supported(3, 2, true, false));
        assert!(!glm5_concurrency_supported(3, 6, true, false));
        assert!(!glm5_concurrency_supported(4, 4, true, false));
        assert!(glm5_concurrency_supported(4, 4, true, true));
        assert!(!glm5_concurrency_supported(4, 5, true, true));
        assert!(!glm5_concurrency_supported(4, 4, false, true));
    }

    #[test]
    fn glm5_long_context_stays_within_model_limit_and_uses_chunking() {
        assert!(glm5_context_supported(100_000, 1024, 1_048_576));
        assert!(!glm5_context_supported(1_048_577, 1024, 1_048_576));
        assert!(!glm5_context_supported(100_000, 0, 1_048_576));
    }

    #[test]
    fn glm5_mtp_context_includes_all_drafts_and_rejects_overflow() {
        assert!(glm5_mtp_context_supported(2044, 4, 2048));
        assert!(!glm5_mtp_context_supported(2045, 4, 2048));
        assert!(glm5_mtp_context_supported(2047, 1, 2048));
        assert!(!glm5_mtp_context_supported(2048, 1, 2048));
        assert!(glm5_mtp_context_supported(2047, 0, 2048));
        assert!(!glm5_mtp_context_supported(2048, 0, 2048));
        assert!(!glm5_mtp_context_supported(16384, 4, 2048));
        assert!(!glm5_mtp_context_supported(usize::MAX, 1, usize::MAX));
        assert!(!glm5_mtp_context_supported(1, usize::MAX, usize::MAX));
        assert!(!glm5_mtp_context_supported(0, 1, 0));
    }

    #[test]
    fn glm5_long_context_concurrency_requires_explicit_sparse_decode() {
        assert!(glm5_long_context_concurrency_supported(
            100_000, 2048, 1, 1, false, false
        ));
        assert!(!glm5_long_context_concurrency_supported(
            100_000, 2048, 2, 2, false, false
        ));
        assert!(glm5_long_context_concurrency_supported(
            2048, 2048, 3, 5, false, false
        ));
        assert!(glm5_long_context_concurrency_supported(
            16384, 2048, 3, 3, true, false
        ));
        assert!(glm5_long_context_concurrency_supported(
            16384, 2048, 2, 5, true, false
        ));
        assert!(!glm5_long_context_concurrency_supported(
            16384, 2048, 4, 4, true, false
        ));
        assert!(!glm5_long_context_concurrency_supported(
            16384, 2048, 1, 5, true, false
        ));
    }

    #[test]
    fn glm5_long_c4_requires_its_own_bounded_opt_in() {
        assert!(glm5_long_context_concurrency_supported(
            16384, 2048, 4, 4, true, true
        ));
        for (context, active, admitted, sparse) in [
            (16385, 4, 4, true),
            (16384, 4, 5, true),
            (16384, 5, 5, true),
            (16384, 4, 4, false),
        ] {
            assert!(!glm5_long_context_concurrency_supported(
                context, 2048, active, admitted, sparse, true
            ));
        }
    }
}

/// Initialize the GPU backend for the active feature.
///
/// Compile-time dispatch:
/// - `cuda` feature → `AtlasCudaBackend` loading PTX modules from `ptx_set`.
/// - `metal` feature → `MetalGpuBackend` loading metallib modules from
///   `ptx_set` as well. Both arms register the RESOLVED target's modules;
///   `metallib_modules()` is a plain alias of target 0, so registering from
///   it served another model's kernels in a multi-target build.
#[cfg(feature = "cuda")]
pub(crate) fn init_gpu_backend(
    args: &cli::ServeArgs,
    ptx_set: &atlas_kernels::TargetPtxSet,
) -> Result<(Box<dyn spark_runtime::gpu::GpuBackend>, usize)> {
    let backend =
        spark_runtime::cuda_backend::AtlasCudaBackend::new(args.gpu_ordinal, &ptx_set.modules)
            .context("Failed to initialize CUDA backend")?;

    let gpu: Box<dyn spark_runtime::gpu::GpuBackend> = Box::new(backend);
    let total_mem = gpu.total_memory()?;
    let free_mem = gpu.free_memory()?;
    // Baseline for self-relative KV budgeting: free memory now (post context +
    // PTX modules, pre weights) minus free-at-build = this process's own
    // footprint, co-tenants excluded. See gpu::baseline_free_bytes.
    spark_runtime::gpu::set_baseline_free_bytes(free_mem);
    tracing::info!(
        "GPU {}: {:.1} GB total, {:.1} GB free",
        args.gpu_ordinal,
        total_mem as f64 / (1024.0 * 1024.0 * 1024.0),
        free_mem as f64 / (1024.0 * 1024.0 * 1024.0),
    );
    Ok((gpu, free_mem))
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub(crate) fn init_gpu_backend(
    args: &cli::ServeArgs,
    ptx_set: &atlas_kernels::TargetPtxSet,
) -> Result<(Box<dyn spark_runtime::gpu::GpuBackend>, usize)> {
    // The RESOLVED target's modules, exactly like the CUDA arm above.
    // `metallib_modules()` is an alias of `ptx_modules()`, which build-codegen
    // emits as a plain alias of TARGET 0 in a multi-target build — so this
    // registered another model's kernels and every lookup for the model
    // actually being served failed.
    let gpu: Box<dyn spark_runtime::gpu::GpuBackend> = Box::new(
        spark_runtime::metal_backend::MetalGpuBackend::new(args.gpu_ordinal, &ptx_set.modules)
            .context("Failed to initialize Metal backend")?,
    );
    let total_mem = gpu.total_memory()?;
    let free_mem = gpu.free_memory()?;
    spark_runtime::gpu::set_baseline_free_bytes(free_mem);
    tracing::info!(
        "Metal device {}: {:.1} GB total, {:.1} GB free",
        args.gpu_ordinal,
        total_mem as f64 / (1024.0 * 1024.0 * 1024.0),
        free_mem as f64 / (1024.0 * 1024.0 * 1024.0),
    );
    Ok((gpu, free_mem))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn post_load_memory_audit(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    weight_bytes: usize,
    free_mem: usize,
    inference_reserve: usize,
    total_reserve: usize,
    gdn_two_phase_bytes: usize,
    max_batch_tokens_pre: usize,
) -> Result<()> {
    let estimated_free = free_mem.saturating_sub(weight_bytes);
    let actual_free = gpu.free_memory().unwrap_or(estimated_free);
    let available_free = if actual_free > 0 {
        actual_free
    } else {
        estimated_free
    };
    if available_free < total_reserve {
        let avail_gb = available_free as f64 / (1024.0 * 1024.0 * 1024.0);
        let need_gb = total_reserve as f64 / (1024.0 * 1024.0 * 1024.0);
        let hint = if args.max_batch_size > 1 {
            format!(
                " Reduce --max-batch-size (currently {}) or --max-seq-len (currently {}).",
                args.max_batch_size, args.max_seq_len
            )
        } else {
            format!(
                " Reduce --max-seq-len (currently {}) or use a smaller model.",
                args.max_seq_len
            )
        };
        anyhow::bail!(
            "Insufficient GPU memory for inference buffers. \
             After loading {:.2} GB of weights, only {:.2} GB remains \
             but {:.2} GB is needed for SSM state pool ({} slots × {} layers) + scratch buffers.{}",
            weight_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            avail_gb,
            need_gb,
            args.max_batch_size,
            config.num_ssm_layers(),
            hint,
        );
    }
    if gdn_two_phase_bytes > 0 {
        tracing::info!(
            "GDN chunked prefill reserve: {} MB (chunk_size={}, max_seq_len={})",
            gdn_two_phase_bytes / (1024 * 1024),
            max_batch_tokens_pre,
            args.max_seq_len,
        );
    }
    tracing::info!(
        "Weights: {:.2} GB, estimated free: {:.1} GB, actual free: {:.1} GB (reserve: {} MB)",
        weight_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        estimated_free as f64 / (1024.0 * 1024.0 * 1024.0),
        actual_free as f64 / (1024.0 * 1024.0 * 1024.0),
        inference_reserve / (1024 * 1024),
    );
    Ok(())
}
