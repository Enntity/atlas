// SPDX-License-Identifier: AGPL-3.0-only

//! Host-only, per-request native GLM K5 diagnostic. Never controls inference.

#[cfg(test)]
#[path = "verify_dflash_ledger_tests.rs"]
mod tests;

#[derive(Debug, Default, Clone, Copy)]
pub(in crate::scheduler) struct Ledger {
    emitted: u8,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct Prepared {
    position: usize,
    seed: u32,
    drafts: [u32; 4],
    raw: [u32; 5],
    vocab: usize,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct Record {
    pub ordinal: u8,
    pub position: usize,
    pub seed: u32,
    pub drafts: [u32; 4],
    pub raw: [u32; 5],
    pub selected: [u32; 5],
    pub accepted: usize,
}

impl Ledger {
    pub(super) fn prepare(
        &self,
        enabled: bool,
        position: usize,
        seed: u32,
        drafts: &[u32],
        raw: &[u32],
        vocab: usize,
    ) -> Option<Prepared> {
        if !enabled || self.emitted >= 8 || drafts.len() != 4 || raw.len() != 5 {
            return None;
        }
        position.checked_add(5)?;
        if seed as usize >= vocab || drafts.iter().chain(raw).any(|&id| id as usize >= vocab) {
            return None;
        }
        Some(Prepared {
            position,
            seed,
            drafts: drafts.try_into().ok()?,
            raw: raw.try_into().ok()?,
            vocab,
        })
    }

    pub(super) fn finish(
        &mut self,
        prepared: Option<Prepared>,
        selected: &[u32],
        accepted: usize,
    ) -> Option<Record> {
        let prepared = prepared?;
        if self.emitted >= 8
            || selected.len() != 5
            || selected.iter().any(|&id| id as usize >= prepared.vocab)
            || accepted > 4
        {
            return None;
        }
        let expected = prepared
            .drafts
            .iter()
            .zip(selected)
            .take_while(|(d, t)| d == t)
            .count();
        if accepted != expected {
            return None;
        }
        self.emitted += 1;
        Some(Record {
            ordinal: self.emitted,
            position: prepared.position,
            seed: prepared.seed,
            drafts: prepared.drafts,
            raw: prepared.raw,
            selected: selected.try_into().ok()?,
            accepted,
        })
    }

    #[cfg(test)]
    pub(super) fn emitted(&self) -> u8 {
        self.emitted
    }
}

fn flag_enabled(value: Option<&str>) -> bool {
    value == Some("1")
}

/// Configuration is cached, but no diagnostic counter is process-global.
pub(super) fn enabled_for(seq: &spark_model::traits::SequenceState) -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| flag_enabled(std::env::var("ATLAS_GLM_MTP_K5_LEDGER").ok().as_deref()))
        && native_glm(seq)
}

fn native_glm(seq: &spark_model::traits::SequenceState) -> bool {
    seq.proposer_state.as_ref().is_some_and(|state| {
        state
            .as_any()
            .is::<spark_model::layers::Glm5MtpProposerState>()
    })
}

pub(super) fn emit(slot: usize, record: &Record) {
    tracing::info!(
        "GLM MTP K5_LEDGER slot={} ordinal={} position={} seed={} drafts={:?} raw={:?} selected={:?} accepted={}",
        slot,
        record.ordinal,
        record.position,
        record.seed,
        record.drafts,
        record.raw,
        record.selected,
        record.accepted,
    );
}
