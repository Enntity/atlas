// SPDX-License-Identifier: AGPL-3.0-only
//! Actual scheduler/Model/F0 dispatch, with real pooled SSM owners. Recorded
//! normalization launches are not CUDA clamp numerics or concurrent NCCL proof.
use super::super::{prefill_a_step::start_chunked_prefill, sched_ctx::SchedCtx, *};
use spark_model::model::glm_c2_test_support::{Event, Fixture, Wire};
use spark_model::traits::Model;

fn request() -> InferenceRequest {
    let (response_tx, _rx) = tokio::sync::oneshot::channel();
    InferenceRequest::Blocking {
        prompt_tokens: std::sync::Arc::new(vec![1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]),
        session_hash: 0,
        adapter_slot: -1,
        src_lang_id: 0,
        tgt_lang_id: 0,
        num_beams: 1,
        length_penalty: 1.0,
        early_stopping: false,
        image_pixels: vec![],
        max_tokens: 4,
        min_tokens: 0,
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        top_n_sigma: 0.0,
        min_p: 0.0,
        repetition_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        dry_multiplier: 0.0,
        dry_base: 1.75,
        dry_allowed_length: 2,
        lz_penalty: 0.0,
        logit_bias: vec![],
        stop_tokens: vec![],
        enable_thinking: false,
        thinking_budget: None,
        repetition_detection: None,
        require_tool_call: false,
        tools_present: false,
        suppress_tool_call: false,
        disable_mtp: true,
        grammar_spec: None,
        seed: Some(1),
        top_logprobs: None,
        prompt_logprobs: None,
        echo: false,
        timeout_at: None,
        response_tx,
    }
}

fn replay(
    worker: &impl Model,
    slots: &mut [Option<SequenceState>],
    tx: &Wire,
    rx: &Wire,
    commands: usize,
) {
    rx.queue(&tx.packets());
    for _ in 0..commands {
        assert!(worker.ep_worker_step(slots).unwrap());
    }
    rx.assert_drained();
}

fn run(first_chunk: bool) {
    const CHILD: &str = "ATLAS_GLM_NORMALIZATION_TEST";
    let name = if first_chunk {
        "actual_glm_first_chunk_normalizes_each_rank_on_default_stream"
    } else {
        "actual_glm_continuation_normalizes_each_rank_on_default_stream"
    };
    if std::env::var(CHILD).as_deref() != Ok("1") {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                &format!("scheduler::phase_continue_prefills::normalization_tests::{name}"),
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("ATLAS_EP_PROTOCOL", "v2")
            .env("ATLAS_GLM_INDEPENDENT_DECODE", "0")
            .env("ATLAS_GLM_MTP_DISTRIBUTED", "0")
            .env("ATLAS_GLM_MTP_BATCHED_PREFILL", "0")
            .env("ATLAS_GLM_MTP_REPAIR", "0")
            .env("ATLAS_SSM_TAIL_CKPT", "0")
            .env_remove("ATLAS_SSM_H_FP16")
            .status()
            .unwrap();
        assert!(
            status.success(),
            "actual normalization fixture child failed"
        );
        return;
    }
    let prepare = |rank| {
        let mut fixture = Fixture::legacy_ssm(rank);
        let wire = fixture.install_wire();
        wire.enable_cold_prefix();
        let (model, seqs, observer) = fixture.into_parts();
        assert!(model.glm_paired_execution().is_none());
        (model, seqs, observer, wire)
    };
    let (mut model, [mut retained, mut available], observer, tx) = prepare(0);
    let (mut worker, worker_seqs, peer_observer, rx) = prepare(1);
    let mut slots = worker_seqs.map(Some);
    // Keep slot0 occupied so this checks physical slot1, not row0 by accident.
    model.free_sequence(&mut available).unwrap();
    let stream = model.create_stream().unwrap();
    let event = model.create_event().unwrap();
    let default_stream = model.default_stream();
    assert_ne!(stream, default_stream);
    assert!(model.has_ssm_layers() && model.supports_chunked_mla());
    tx.clear();
    observer.clear();
    peer_observer.clear();
    let sched = SchedCtx::for_test();
    let started = start_chunked_prefill(
        &sched,
        None,
        None,
        None,
        None,
        &model,
        request(),
        &[0],
        4,
        stream,
        event,
        &mut None,
        0,
        false,
        None,
        None,
    )
    .unwrap();
    let StartPrefillResult::InProgress(mut p) = started else {
        panic!("expected actual first chunk, not complete prompt");
    };
    assert_eq!((p.seq.slot_idx, p.chunk_offset), (1, 4));
    // One worker step per (seq, cmd) preamble: native-only fence, vision
    // state clear, then the chunk itself.
    replay(&worker, &mut slots, &tx, &rx, 3);
    if !first_chunk {
        tx.clear();
        observer.clear();
        peer_observer.clear();
        let mut completed = vec![];
        let mut mixed = false;
        super::run_standard::run_standard_chunk_loop(
            &model,
            &mut p,
            0,
            &mut vec![],
            4,
            4,
            stream,
            event,
            false,
            false,
            false,
            None,
            None,
            None,
            None,
            None,
            false,
            &sched,
            &mut completed,
            &mut mixed,
            4,
            &super::SpecStep {
                num_drafts: 0,
                dflash_verify_raw_argmax: false,
            },
            &mut vec![],
        );
        assert_eq!(p.chunk_offset, 8);
        assert!(completed.is_empty() && !mixed);
        // Continuations re-send the fence but not vision state.
        replay(&worker, &mut slots, &tx, &rx, 2);
    }
    let offset = if first_chunk { 0 } else { 4 };
    for events in [observer.events(), peer_observer.events()] {
        assert!(events.contains(&Event::Target(4, offset, model.default_stream())));
    }
    let head_norm = observer.normalization_streams(&model, &p.seq).unwrap();
    let peer_norm = peer_observer
        .normalization_streams(&worker, slots[1].as_ref().unwrap())
        .unwrap();
    // Release genuine owners before reporting an expected behavioral RED.
    model.free_sequence(&mut p.seq).unwrap();
    model.free_sequence(&mut retained).unwrap();
    for seq in &mut slots {
        worker.free_sequence(seq.as_mut().unwrap()).unwrap();
    }
    // free_sequence returns slots, but the emptied guards retain their Arc
    // until the actual sequence containers are dropped. Teardown checks this.
    drop(p);
    drop(retained);
    drop(available);
    drop(slots);
    model.teardown().unwrap();
    worker.teardown().unwrap();
    assert_eq!(
        peer_norm,
        vec![default_stream],
        "worker must normalize once on default"
    );
    assert_eq!(
        head_norm,
        vec![default_stream],
        "head must normalize once on default"
    );
}

#[test]
fn actual_glm_first_chunk_normalizes_each_rank_on_default_stream() {
    run(true);
}

#[test]
fn actual_glm_continuation_normalizes_each_rank_on_default_stream() {
    run(false);
}
