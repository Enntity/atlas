// SPDX-License-Identifier: AGPL-3.0-only
//! Supervised fixed-two-owner serving; never enters the ordinary cleanup ladder.
use super::{ActiveSeq, InferenceRequest, LoraRotation, Model, sched_ctx::SchedCtx};
use crate::glm_terminal_session::SelectedModel;

#[path = "glm_c2_selected_admission.rs"]
mod admission;

pub(super) struct Tokens {
    pub eos: Vec<u32>,
    pub think_end: Option<u32>,
    pub think_start: Option<u32>,
    pub tool_start: Option<u32>,
    pub tool_end: Option<u32>,
    pub spontaneous_budget: u32,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_selected(
    owner: SelectedModel,
    mut request_rx: tokio::sync::mpsc::Receiver<InferenceRequest>,
    mut rotation_rx: tokio::sync::mpsc::Receiver<LoraRotation>,
    eos_tokens: Vec<u32>,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    spontaneous_think_budget: u32,
    sched: SchedCtx,
) -> ! {
    owner.bind_execution_thread();
    let tokens = Tokens {
        eos: eos_tokens,
        think_end: think_end_token,
        think_start: think_start_token,
        tool_start: tool_call_start_token,
        tool_end: tool_call_end_token,
        spontaneous_budget: spontaneous_think_budget,
    };
    let operation = owner.begin();
    operation.require(owner.model().bind_gpu_to_thread());
    operation.complete();
    let mut active: Vec<ActiveSeq> = Vec::with_capacity(2);
    loop {
        owner.check_health();
        if crate::tui::shutdown::requested() || request_rx.is_closed() {
            request_rx.close();
            while let Ok(req) = request_rx.try_recv() {
                admission::reject(req, "selected server is shutting down");
            }
            // `active` remains alive across this diverging call. No F1, drain,
            // Model teardown, or backend Drop follows matched shutdown/release.
            sched.stats.glm_c2.summary("shutdown");
            owner.shutdown_head();
        }
        for _ in 0..2 {
            let Ok((_, response)) = rotation_rx.try_recv() else {
                break;
            };
            let _ = response.send(Err("selected paired serving forbids adapter changes".into()));
        }
        for _ in 0..(2 - active.len()) {
            if crate::tui::shutdown::requested() || request_rx.is_closed() {
                break;
            }
            let Ok(req) = request_rx.try_recv() else {
                break;
            };
            if let Err(error) = admission::validate(&req, owner.model()) {
                admission::reject(req, &error.to_string());
                continue;
            }
            let operation = owner.begin();
            let a = admission::admit(&operation, owner.model(), req, &tokens, &sched);
            active.push(a);
            operation.complete();
        }
        if crate::tui::shutdown::requested() || request_rx.is_closed() {
            continue;
        }
        if !active.is_empty() {
            let operation = owner.begin();
            let verify = context(&sched, &tokens);
            operation.require(super::glm_c2_serial::step_selected_serial(
                owner.model(),
                &mut active,
                &sched,
                &verify,
            ));
            operation.complete();
            if crate::tui::shutdown::requested() || request_rx.is_closed() {
                continue;
            }
            let operation = owner.begin();
            operation.require(super::mod_helpers::retire_selected_finished_sequences(
                owner.model(),
                &mut active,
                sched.limits.max_seq_len,
            ));
            operation.complete();
            if active.is_empty() {
                sched.stats.glm_c2.summary("idle-wave");
            }
        } else {
            // The explicit ticket policy bounds idle communicator observations;
            // no detached watchdog or GPU thread is introduced.
            std::thread::sleep(std::time::Duration::from_millis(owner.poll_ms()));
        }
    }
}

fn context<'a>(sched: &'a SchedCtx, tokens: &Tokens) -> super::logit_processors::LogitsContext<'a> {
    super::logit_processors::LogitsContext {
        scratch: &sched.scratch,
        dumps: &sched.dumps,
        stats: sched.stats.clone(),
        watchdog: sched.watchdog,
        boundary_mask: sched.masks.boundary.clone(),
        mid_word_mask: sched.masks.mid_word.clone(),
        sampling: sched.levers.sampling(),
        timing: sched.timing.clone(),
        think_end_token: tokens.think_end,
        think_start_token: tokens.think_start,
        tool_call_start_token: tokens.tool_start,
        tool_call_end_token: tokens.tool_end,
    }
}
